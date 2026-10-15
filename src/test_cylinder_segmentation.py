"""General local-support regressions; no filename/component-axis sample tuning."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

import cylinder_segmentation as segmentation
from geometry_analysis import _prepare, _cylinder_candidates, cylinder_candidates, inspect_mesh
from mesh_evidence import MeshBody, map_record_to_instance
from test_geometry_evidence import _strip, CandidateAssertions


def revolved(profile,sections=48,coverage=360.):
    full=coverage==360.
    count=sections if full else sections+1
    angles=np.linspace(0,np.deg2rad(coverage),count,endpoint=not full)
    vertices=np.vstack([np.column_stack([r*np.cos(angles),r*np.sin(angles),np.full(count,z)])
                        for r,z in profile])
    faces=[]
    for layer in range(len(profile)-1):
        for i in range(sections):
            j=(i+1)%count;a=layer*count+i;b=layer*count+j
            faces.extend([[a,b,b+count],[a,b+count,a+count]])
    return trimesh.Trimesh(vertices,faces,process=False)


def blended_wall(radius=3.,height=7.,blend=.8,sections=48,coverage=360.,cap_steps=8):
    # Smooth tangent quarter-circle transitions above and below a true axial run.
    upper=[(radius-blend+blend*np.cos(a),height/2+blend*np.sin(a))
           for a in np.linspace(0,np.pi/2,cap_steps+1)[1:]]
    lower=[(r,-z) for r,z in upper[::-1]]
    return revolved(lower+[(radius,-height/2),(radius,height/2)]+upper,sections,coverage)


class LocalCylinderTests(CandidateAssertions):
    def test_smooth_blends_do_not_hide_true_cylinder(self):
        for sections in [12,24,64]:
            with self.subTest(sections=sections):
                mesh=blended_wall(sections=sections)
                data=_prepare(mesh)
                self.assertEqual(_cylinder_candidates(data,.05,recover=False),[])
                candidates=cylinder_candidates(mesh)
                self.assertEqual(len(candidates),1)
                c=candidates[0]
                self.assertEqual(c['segmentation_method'],'local_normal_axis_consensus')
                self.assertAlmostEqual(c['radius_mm'],3.,places=9)
                self.assertAlmostEqual(c['axial_length_mm'],7.,places=9)
                self.assertEqual(c['triangle_count'],2*sections)
                self.check_candidate(mesh,c)

    def test_arbitrary_rotation_and_translation_are_equivariant(self):
        base=blended_wall(sections=32)
        for angle,axis,translation in [(.17,[2,3,7],[31,-23,87]),(1.31,[-7,1,3],[1e8,-2e8,3e8]),
                                       (2.77,[.4,-1,.3],[-2,4,7])]:
            with self.subTest(angle=angle):
                transform=trimesh.transformations.rotation_matrix(angle,axis)
                mesh=base.copy();mesh.apply_transform(transform);mesh.apply_translation(translation)
                c,=cylinder_candidates(mesh)
                self.assertAlmostEqual(c['radius_mm'],3.,delta=1e-5)
                self.assertAlmostEqual(c['axial_length_mm'],7.,delta=1e-5)
                self.assertGreater(abs(np.dot(c['axis'],transform[:3,2])),1-1e-9)
                self.assertLess(c['max_vertex_residual_mm'],1e-5)
                self.assertFalse(c['hole']['confirmed'])

    def test_partial_blended_arc_not_full_hole(self):
        c,=cylinder_candidates(blended_wall(sections=24,coverage=120.))
        self.assertAlmostEqual(c['angular_coverage_degrees'],120.,places=7)
        self.assertFalse(c['wraps_full_circumference'])
        self.assertFalse(c['hole']['confirmed'])

    def test_nonuniform_angular_tessellation_keeps_same_support(self):
        mesh=blended_wall(sections=40)
        angles=np.linspace(0,2*np.pi,40,endpoint=False)
        distorted=angles+.04*np.sin(angles)
        for i,(old,new) in enumerate(zip(angles,distorted)):
            ids=np.arange(i,len(mesh.vertices),40)
            radii=np.linalg.norm(mesh.vertices[ids,:2],axis=1)
            mesh.vertices[ids,0]=radii*np.cos(new);mesh.vertices[ids,1]=radii*np.sin(new)
        c,=cylinder_candidates(mesh)
        self.assertAlmostEqual(c['radius_mm'],3.,places=8)
        self.assertAlmostEqual(c['axial_length_mm'],7.,places=8)

    def test_face_order_permutation_preserves_support_without_sample_ids(self):
        mesh=blended_wall(sections=32)
        order=np.random.default_rng(278).permutation(len(mesh.faces))
        shuffled=trimesh.Trimesh(mesh.vertices,mesh.faces[order],process=False)
        c,=cylinder_candidates(shuffled)
        reference,=cylinder_candidates(mesh)
        self.assertEqual(set(order[c['source_triangle_ids']]),set(reference['source_triangle_ids']))
        self.assertAlmostEqual(c['radius_mm'],reference['radius_mm'],places=9)

    def test_inverted_source_winding_is_preserved_and_only_possible_role(self):
        mesh=blended_wall();mesh.faces=mesh.faces[:,::-1]
        before=mesh.faces.copy()
        c,=cylinder_candidates(mesh)
        self.assertEqual(c['hole']['classification'],'possible_hole')
        self.assertFalse(c['hole']['confirmed'])
        np.testing.assert_array_equal(mesh.faces,before)

    def test_disconnected_coaxial_blends_do_not_merge(self):
        first=blended_wall();second=first.copy();second.apply_translation([0,0,30])
        mesh=trimesh.util.concatenate([first,second])
        cs=cylinder_candidates(mesh)
        self.assertEqual(len(cs),2)
        self.assertFalse(set(cs[0]['source_triangle_ids'])&set(cs[1]['source_triangle_ids']))
        self.assertTrue(all(abs(c['axial_length_mm']-7.)<1e-8 for c in cs))

    def test_curved_belts_on_tori_are_not_promoted_to_cylinders(self):
        for major,minor,ms,ns in [(5.,1.,24,12),(5.,1.,48,24),(1.,.1,64,32),(.2,.1,32,12)]:
            with self.subTest(major=major,minor=minor,ms=ms,ns=ns):
                mesh=trimesh.creation.torus(major_radius=major,minor_radius=minor,
                                           major_sections=ms,minor_sections=ns)
                self.assertEqual(cylinder_candidates(mesh),[])

    def test_symmetric_curved_belts_and_unchanged_neighbor_subdivision_rejected(self):
        angles=np.deg2rad(15.+30.*np.arange(12))
        torus=revolved([(.2+.1*np.cos(a),.1*np.sin(a)) for a in angles],sections=32)
        # Close the cyclic profile with identical source ring indices.
        count=32
        closing=[]
        last=(len(angles)-1)*count
        for i in range(count):
            j=(i+1)%count
            closing.extend([[last+i,last+j,j],[last+i,j,i]])
        torus=trimesh.Trimesh(torus.vertices,np.vstack([torus.faces,closing]),process=False)
        sphere_angles=np.deg2rad(np.arange(-75,76,30))
        sphere=revolved([(np.cos(a),np.sin(a)) for a in sphere_angles],sections=32)
        for mesh in [torus,sphere]:
            with self.subTest(shape=len(mesh.faces)):
                self.assertEqual(cylinder_candidates(mesh),[])
                vertices=mesh.vertices.tolist();faces=[]
                for triangle in mesh.faces:
                    center=len(vertices);vertices.append(mesh.vertices[triangle].mean(axis=0).tolist())
                    a,b,c=triangle.tolist();faces.extend([[a,b,center],[b,c,center],[c,a,center]])
                refined=trimesh.Trimesh(vertices,faces,process=False)
                self.assertEqual(cylinder_candidates(refined),[])

    def test_sphere_ellipsoid_and_freeform_do_not_gain_false_cylinders(self):
        sphere=trimesh.creation.icosphere(subdivisions=2,radius=2.)
        ellipsoid=sphere.copy();ellipsoid.apply_scale([1.,.7,1.3])
        r=.1
        profile=[(r*(1+.15*np.sin(3*z)),z) for z in np.linspace(-1,1,17)]
        freeform=revolved(profile,sections=32)
        for mesh in [sphere,ellipsoid,freeform]:
            with self.subTest(mesh=len(mesh.faces)):
                self.assertEqual(cylinder_candidates(mesh),[])

    def test_shallow_cones_and_short_or_insufficient_arcs_remain_unresolved(self):
        for coverage in [60.,120.,360.]:
            for taper in [.0001,.002,.05]:
                mesh=_strip(radius=3,height=7,sections=24,coverage=coverage,top_radius=3+taper)
                self.assertEqual(cylinder_candidates(mesh),[])
        self.assertEqual(cylinder_candidates(blended_wall(coverage=40.)),[])
        self.assertEqual(cylinder_candidates(blended_wall(height=.05)),[])

    def test_coarse_new_fit_retains_whole_triangle_sagitta_failure(self):
        c,=cylinder_candidates(blended_wall(radius=3.,sections=12))
        self.assertTrue(c['vertex_fit_within_tolerance'])
        self.assertGreater(c['max_surface_residual_mm'],.05)
        self.assertFalse(c['surface_approximation_within_tolerance'])

    def test_budget_skips_are_explicit_and_no_fallback_forces_a_fit(self):
        data=_prepare(blended_wall())
        diagnostics={}
        with patch.object(segmentation,'MAX_AXES_PER_BODY',0):
            candidates=_cylinder_candidates(data,.05,diagnostics=diagnostics)
        self.assertEqual(candidates,[])
        self.assertTrue(diagnostics['body_axis_budget_reached'])
        self.assertGreater(diagnostics['axis_hypotheses_omitted'],0)
        self.assertFalse(diagnostics['search_exhaustive'])
        self.assertFalse(diagnostics['candidate_recall_certified'])

    def test_source_evidence_is_mapped_with_parent_band_and_no_mutation(self):
        mesh=blended_wall(sections=24)
        before_vertices,before_faces=mesh.vertices.copy(),mesh.faces.copy()
        record=inspect_mesh(mesh,0,0)
        n=len(mesh.faces);body=MeshBody(mesh,np.arange(n)+1000,np.arange(len(mesh.vertices))+7000)
        # Fake external index offsets are to test recursive mapper only.
        mapped=map_record_to_instance(record,body,{'components':[]},3)
        c,=mapped['cylinders']
        parent=c['local_segmentation_evidence']['parent_normal_band']
        self.assertTrue(all(i>=1000 for i in parent['source_triangle_ids']))
        self.assertEqual(parent['source_triangle_id_scope'],'build_instance_faces')
        self.assertTrue(all(i>=1000 for band in mapped['cylinder_segmentation_diagnostics']['bands']
                            for i in band['source_triangle_ids']))
        np.testing.assert_array_equal(mesh.vertices,before_vertices)
        np.testing.assert_array_equal(mesh.faces,before_faces)
        json.dumps(mapped,allow_nan=False)

    def test_deterministic_repeated_runs(self):
        mesh=blended_wall()
        a=cylinder_candidates(mesh);b=cylinder_candidates(mesh)
        self.assertEqual(a,b)


if __name__=='__main__':unittest.main(verbosity=2)
