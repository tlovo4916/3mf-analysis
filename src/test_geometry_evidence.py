"""Regression evidence for non-repairing, body-local geometric candidates."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from geometry_analysis import (
    _moments, _prepare, cylinder_candidates, inspect_mesh, planar_candidates,
    plate_frame,
)


def _strip(radius=5., height=20., sections=32, coverage=360., top_radius=None):
    """Oriented outward cylinder/frustum side, without fabricated cap faces."""
    full = coverage == 360.
    count = sections if full else sections + 1
    angles = np.linspace(0, np.deg2rad(coverage), count, endpoint=not full)
    top_radius = radius if top_radius is None else top_radius
    vertices = np.vstack([
        np.column_stack((r * np.cos(angles), r * np.sin(angles), np.full(count, z)))
        for r, z in [(radius, -height / 2), (top_radius, height / 2)]
    ])
    faces = []
    for i in range(sections):
        j = (i + 1) % count
        faces.extend([[i, j, j + count], [i, j + count, i + count]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _dense_one_face(mesh, iterations=5):
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    for _ in range(iterations):
        vertices, faces = trimesh.remesh.subdivide(
            vertices, faces, face_index=np.arange(min(4, len(faces))))
    return trimesh.Trimesh(vertices, faces, process=False)


class FrameEvidenceTests(unittest.TestCase):
    def test_rotated_plate_xy_rectangle_and_right_handed_round_trip(self):
        mesh = trimesh.creation.box(extents=[123, 37, 1.7])
        mesh.apply_transform(trimesh.transformations.rotation_matrix(1.09, [1, 2, -4]))
        mesh.apply_translation([17, -29, 83])
        rotation, origin, local = plate_frame(mesh)
        np.testing.assert_allclose(local.extents, [123, 37, 1.7], atol=1e-9)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        np.testing.assert_allclose(local.vertices @ rotation.T + origin, mesh.vertices,
                                   atol=1e-10)
        np.testing.assert_array_equal(local.faces, mesh.faces)

    def test_exact_surface_moments_survive_nonuniform_retessellation(self):
        # No vertex-cloud covariance can pass this: only a small surface patch
        # receives many new vertices while the physical surface is unchanged.
        mesh = trimesh.creation.box(extents=[100, 43, 2])
        dense = _dense_one_face(mesh, iterations=7)
        for current in (mesh, dense):
            center, covariance, scale = _moments(_prepare(current))
            if current is mesh:
                expected_center, expected_covariance = center, covariance * scale ** 2
            else:
                np.testing.assert_allclose(center, expected_center, atol=1e-10)
                np.testing.assert_allclose(covariance * scale ** 2, expected_covariance,
                                           atol=1e-9)
        rotation = trimesh.transformations.rotation_matrix(.91, [2, -3, 1])
        mesh.apply_transform(rotation)
        dense.apply_transform(rotation)
        r1, o1, l1 = plate_frame(mesh)
        r2, o2, l2 = plate_frame(dense)
        np.testing.assert_allclose(l2.extents, l1.extents, atol=1e-9)
        np.testing.assert_allclose(o2, o1, atol=1e-9)
        np.testing.assert_allclose(np.abs(r1.T @ r2), np.eye(3), atol=1e-9)

    def test_duplicate_vertex_storage_has_no_weight_and_unused_are_not_extents(self):
        original = trimesh.creation.box(extents=[60, 19, 1.5])
        original.apply_transform(trimesh.transformations.rotation_matrix(.72, [2, 5, 3]))
        triangles = original.vertices[original.faces]
        # Every face stores its own three vertices, plus unrelated unused data.
        vertices = np.vstack((triangles.reshape((-1, 3)), [[1e20, -1e20, 1e20],
                                                         [np.nan, np.inf, 0]]))
        faces = np.arange(len(original.faces) * 3).reshape((-1, 3))
        unshared = trimesh.Trimesh(vertices, faces, process=False)
        r0, o0, local0 = plate_frame(original)
        rotation, origin, local = plate_frame(unshared)
        np.testing.assert_allclose(local.extents, local0.extents, atol=1e-10)
        np.testing.assert_allclose(origin, o0, atol=1e-10)
        np.testing.assert_allclose(np.abs(r0.T @ rotation), np.eye(3), atol=1e-9)
        self.assertEqual(len(local.vertices), 3 * len(original.faces))
        ids = local.metadata['source_vertex_ids']
        np.testing.assert_allclose(local.vertices @ rotation.T + origin,
                                   unshared.vertices[ids], atol=1e-10)
        np.testing.assert_array_equal(local.faces, faces)
        self.assertFalse(local.metadata['mesh_repair_performed'])

    def test_large_translation_does_not_ill_condition_frame(self):
        mesh = trimesh.creation.box(extents=[100, 38, 2])
        mesh.apply_transform(trimesh.transformations.rotation_matrix(.93, [1, 2, 3]))
        mesh.apply_translation([1e11, -2e11, 3e11])
        rotation, origin, local = plate_frame(mesh)
        np.testing.assert_allclose(local.extents, [100, 38, 2], atol=2e-4)
        np.testing.assert_allclose(local.vertices @ rotation.T + origin, mesh.vertices,
                                   atol=1e-4)

    def test_no_normal_repair_and_negative_volume_is_blocking(self):
        mesh = trimesh.creation.box(extents=[20, 8, 1])
        mesh.faces = mesh.faces[:, ::-1]
        before_vertices, before_faces = mesh.vertices.copy(), mesh.faces.copy()
        with patch.object(trimesh.Trimesh, 'fix_normals',
                          side_effect=AssertionError('must never repair source or local mesh')):
            _, _, local = plate_frame(mesh)
            report = inspect_mesh(mesh, 3, 7)
        np.testing.assert_array_equal(mesh.vertices, before_vertices)
        np.testing.assert_array_equal(mesh.faces, before_faces)
        np.testing.assert_array_equal(local.faces, before_faces)
        self.assertLess(local.volume, 0)
        self.assertLess(report['signed_volume_mm3'], 0)
        self.assertLess(report['volume_mm3'], 0)
        self.assertEqual(report['strategy'], 'repair_required')
        self.assertFalse(report['mesh_repair_performed'])
        self.assertTrue(report['candidate_geometry_only'])

    def test_invalid_empty_nonfinite_and_degenerate_inputs_rejected(self):
        meshes = [
            trimesh.Trimesh(vertices=[], faces=[], process=False),
            trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                            faces=[[0, 1, 2]], process=False),
            trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [np.inf, 1, 0]],
                            faces=[[0, 1, 2]], process=False),
        ]
        for mesh in meshes:
            for function in (plate_frame, planar_candidates, cylinder_candidates):
                with self.subTest(function=function.__name__, vertices=len(mesh.vertices)):
                    with self.assertRaises(ValueError):
                        function(mesh)
            with self.assertRaises(ValueError):
                inspect_mesh(mesh, 0, 0)


class CandidateAssertions(unittest.TestCase):
    def check_candidate(self, mesh, candidate):
        self.assertEqual(candidate['source_triangle_id_scope'], 'body_local_faces')
        self.assertEqual(candidate['triangle_count'], len(candidate['source_triangle_ids']))
        ids = np.unique(mesh.faces[candidate['source_triangle_ids']])
        self.assertEqual(candidate['source_vertex_ids'], ids.tolist())
        np.testing.assert_allclose(candidate['points_mm'], mesh.vertices[ids])
        rotation = np.asarray(candidate['frame']['local_to_source_rotation'])
        transform = np.asarray(candidate['frame']['local_to_source_transform'])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        np.testing.assert_allclose(rotation[:, 0], candidate['ref_direction'])
        np.testing.assert_allclose(transform[:3, :3], rotation)
        np.testing.assert_allclose(transform[:3, 3], candidate['frame']['origin_mm'])
        self.assertTrue(candidate['candidate_geometry_only'])
        self.assertFalse(candidate['cad_reconstructed'])
        self.assertFalse(candidate['boundary']['is_cad_trim'])
        self.assertFalse(candidate['boundary']['self_intersection_checked'])
        for loop in candidate['boundary_loops']:
            vertex_ids = loop['vertex_ids']
            np.testing.assert_allclose(loop['points_mm'], mesh.vertices[vertex_ids])
            self.assertEqual(len(loop['edge_uses']), len(vertex_ids) - 1)
            for i, edge in enumerate(loop['edge_uses']):
                self.assertEqual(edge['start'], vertex_ids[i])
                self.assertEqual(edge['end'], vertex_ids[i + 1])
                face = mesh.faces[edge['source_triangle_id']]
                local_edge = edge['local_edge_index']
                self.assertEqual((edge['start'], edge['end']),
                                 (face[local_edge], face[(local_edge + 1) % 3]))
        json.dumps(candidate, allow_nan=False)


class PlaneEvidenceTests(CandidateAssertions):
    def test_isolated_large_triangle_has_frame_and_directed_closed_boundary(self):
        vertices = np.array([[0, 0, 0], [6, 0, 0], [0, 5, 0]], dtype=float)
        mesh = trimesh.Trimesh(vertices, [[0, 1, 2]], process=False)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(.59, [1, 2, 3]))
        mesh.apply_translation([1e9, -1e9, 3e9])
        candidates = planar_candidates(mesh)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.check_candidate(mesh, candidate)
        self.assertEqual(candidate['source_triangle_ids'], [0])
        self.assertAlmostEqual(candidate['area_mm2'], 15, places=5)
        self.assertLess(candidate['max_vertex_residual_mm'], 1e-7)
        self.assertLess(candidate['max_centroid_residual_mm'], 1e-7)
        self.assertTrue(candidate['surface_approximation_within_tolerance'])
        loop = candidate['boundary_loops'][0]
        self.assertEqual(loop['vertex_ids'], [0, 1, 2, 0])
        self.assertEqual(loop['topology_status'], 'closed')
        self.assertTrue(loop['is_closed'])

    def test_two_triangle_plane_internal_edge_not_in_boundary(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]],
                               faces=[[0, 1, 2], [0, 2, 3]], process=False)
        candidate, = planar_candidates(mesh)
        self.check_candidate(mesh, candidate)
        self.assertEqual(candidate['source_triangle_ids'], [0, 1])
        self.assertEqual(candidate['boundary']['boundary_edge_count'], 4)
        self.assertEqual(candidate['boundary_loops'][0]['vertex_ids'], [0, 1, 2, 3, 0])

    def test_inconsistent_winding_is_partial_boundary_not_repaired(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]],
                               faces=[[0, 1, 2], [3, 2, 0]], process=False)
        faces = mesh.faces.copy()
        candidate, = planar_candidates(mesh)
        self.check_candidate(mesh, candidate)
        self.assertEqual(candidate['boundary']['topology_status'], 'partial')
        self.assertTrue(candidate['boundary']['inconsistent_winding_edges'])
        self.assertTrue(any(not loop['is_closed'] for loop in candidate['boundary_loops']))
        self.assertGreater(candidate['max_oriented_normal_residual_degrees'], 179)
        np.testing.assert_array_equal(mesh.faces, faces)

    def test_disconnected_coplanar_regions_not_silently_merged(self):
        first = trimesh.Trimesh(vertices=[[0, 0, 0], [4, 0, 0], [0, 3, 0]],
                               faces=[[0, 1, 2]], process=False)
        second = first.copy()
        second.apply_translation([20, 0, 0])
        mesh = trimesh.util.concatenate([first, second])
        candidates = planar_candidates(mesh)
        self.assertEqual(len(candidates), 2)
        self.assertEqual({tuple(c['source_triangle_ids']) for c in candidates}, {(0,), (1,)})

    def test_real_bent_surface_has_nonzero_vertex_and_centroid_residuals(self):
        # Small enough dihedral for a plane proposal, but not a zero-error claim.
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [10, 0, 0], [10, 10, .01], [0, 10, 0]],
                               faces=[[0, 1, 2], [0, 2, 3]], process=False)
        candidate, = planar_candidates(mesh, tolerance=.05)
        self.assertGreater(candidate['max_vertex_residual_mm'], .001)
        self.assertGreater(candidate['max_centroid_residual_mm'], 0)
        self.assertGreater(candidate['max_normal_residual_degrees'], 0)
        self.assertEqual(candidate['max_vertex_residual_mm'], candidate['max_surface_residual_mm'])


class CylinderEvidenceTests(CandidateAssertions):
    def test_n12_n16_and_fine_cylinder_distinguish_vertex_from_surface_error(self):
        radius = 5.
        for sections in (12, 16, 128):
            with self.subTest(sections=sections):
                mesh = trimesh.creation.cylinder(radius=radius, height=20, sections=sections)
                candidate, = cylinder_candidates(mesh)
                self.check_candidate(mesh, candidate)
                self.assertEqual(candidate['triangle_count'], 2 * sections)
                self.assertAlmostEqual(candidate['radius_mm'], radius, places=10)
                self.assertLess(candidate['max_vertex_residual_mm'], 1e-10)
                self.assertGreater(candidate['max_centroid_residual_mm'], 0)
                sagitta = radius * (1 - np.cos(np.pi / sections))
                self.assertAlmostEqual(candidate['max_mesh_sagitta_mm'], sagitta, places=10)
                self.assertAlmostEqual(candidate['max_surface_residual_mm'], sagitta, places=10)
                self.assertEqual(candidate['surface_approximation_within_tolerance'], sagitta <= .05)
                self.assertAlmostEqual(candidate['max_normal_residual_degrees'], 180 / sections,
                                       places=8)
                self.assertAlmostEqual(candidate['angular_coverage_degrees'], 360., places=8)
                self.assertTrue(candidate['wraps_full_circumference'])

    def test_small_n16_cylinder_caps_excluded_in_huge_flat_assembly(self):
        plate = trimesh.creation.box(extents=[1e6, 1e6, 2])
        cylinder = trimesh.creation.cylinder(radius=1., height=3., sections=16)
        cylinder.apply_translation([2e6, 0, 0])
        mesh = trimesh.util.concatenate([plate, cylinder])
        candidate, = cylinder_candidates(mesh)
        self.check_candidate(mesh, candidate)
        ids = candidate['source_triangle_ids']
        self.assertEqual(len(ids), 32)
        self.assertTrue(all(i >= len(plate.faces) for i in ids))
        # Caps are tiny compared with assembly area, but still planar facets.
        self.assertTrue(np.all(np.abs(mesh.face_normals[ids, 2]) < 1e-10))
        self.assertAlmostEqual(candidate['radius_mm'], 1., places=9)

    def test_large_translation_and_rotated_circle_fit_are_conditioned(self):
        mesh = trimesh.creation.cylinder(radius=7.5, height=17., sections=64)
        transform = trimesh.transformations.rotation_matrix(.97, [3, 2, -1])
        mesh.apply_transform(transform)
        mesh.apply_translation([1e10, -2e10, 3e10])
        candidate, = cylinder_candidates(mesh)
        self.check_candidate(mesh, candidate)
        self.assertAlmostEqual(candidate['radius_mm'], 7.5, delta=2e-5)
        self.assertAlmostEqual(candidate['axial_length_mm'], 17., delta=2e-5)
        self.assertLess(candidate['max_vertex_residual_mm'], 1e-5)
        self.assertLess(candidate['fit_condition_number'], 3.)
        self.assertGreater(abs(np.dot(candidate['axis'], transform[:3, 2])), 1 - 1e-10)
        np.testing.assert_allclose(candidate['axis_point_mm'], [1e10, -2e10, 3e10], atol=1e-5)
        # Evaluate the documented frame/parameterization at a genuine vertex.
        rotation = np.asarray(candidate['frame']['local_to_source_rotation'])
        origin = np.asarray(candidate['frame']['origin_mm'])
        point = np.asarray(candidate['points_mm'][0])
        coordinates = (point - origin) @ rotation
        theta = np.arctan2(coordinates[1], coordinates[0])
        evaluated = origin + rotation @ [candidate['radius_mm'] * np.cos(theta),
                                         candidate['radius_mm'] * np.sin(theta), coordinates[2]]
        np.testing.assert_allclose(evaluated, point, atol=1e-5)

    def test_arcs_and_inward_winding_are_possible_holes_not_confirmed(self):
        mesh = _strip(radius=5, sections=12, coverage=120)
        mesh.faces = mesh.faces[:, ::-1]
        candidate, = cylinder_candidates(mesh)
        self.check_candidate(mesh, candidate)
        self.assertAlmostEqual(candidate['angular_coverage_degrees'], 120., places=8)
        self.assertFalse(candidate['wraps_full_circumference'])
        self.assertEqual(candidate['role_candidate'], 'hole')  # legacy report field
        self.assertEqual(candidate['role_status'], 'possible_hole')
        self.assertEqual(candidate['hole']['classification'], 'possible_hole')
        self.assertFalse(candidate['hole']['confirmed'])
        self.assertFalse(candidate['hole']['cavity_topology_verified'])
        self.assertAlmostEqual(candidate['hole']['inward_normal_area_fraction'], 1.)
        self.assertEqual(len(candidate['boundary_loops']), 1)
        self.assertEqual(candidate['boundary']['topology_status'], 'closed')
        self.assertTrue(candidate['boundary_loops'][0]['is_closed'])

    def test_minimum_60_degree_arc_survives_large_coordinate_roundoff(self):
        mesh = _strip(sections=32, coverage=60)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(.83, [2, -3, 1]))
        mesh.apply_translation([1e10, -2e10, 3e10])
        candidate, = cylinder_candidates(mesh)
        self.assertAlmostEqual(candidate['angular_coverage_degrees'], 60., delta=.001)
        self.assertLess(candidate['max_vertex_residual_mm'], 1e-4)
        self.assertFalse(candidate['hole']['confirmed'])

    def test_full_cylinder_has_two_source_winding_boundary_loops(self):
        mesh = _strip(sections=16)
        candidate, = cylinder_candidates(mesh)
        self.check_candidate(mesh, candidate)
        self.assertEqual(len(candidate['boundary_loops']), 2)
        self.assertEqual(candidate['boundary']['boundary_edge_count'], 32)
        signs = []
        for loop in candidate['boundary_loops']:
            self.assertTrue(loop['is_closed'])
            p = np.asarray(loop['points_mm'])
            signs.append(np.sum(p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1]))
        self.assertLess(signs[0] * signs[1], 0)  # preserve both directed orientations

    def test_cones_and_even_shallow_taper_are_not_flattened_to_cylinders(self):
        cone = trimesh.creation.cone(radius=5, height=20, sections=64)
        self.assertEqual(cylinder_candidates(cone), [])
        for difference in (.002, .03, .2, 3.):
            mesh = _strip(radius=5, height=20, sections=32, top_radius=5 + difference)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(.81, [1, 2, 3]))
            mesh.apply_translation([1e6, -1e6, 2e6])
            with self.subTest(top_radius=5 + difference):
                self.assertEqual(cylinder_candidates(mesh), [])

    def test_partial_shallow_cones_cannot_hide_in_a_tilted_normal_nullspace(self):
        for coverage in (60, 90, 120, 180, 270):
            for difference in (.0001, .002, .03):
                mesh = _strip(radius=5, height=20, sections=32, coverage=coverage,
                              top_radius=5 + difference)
                with self.subTest(coverage=coverage, difference=difference):
                    self.assertEqual(cylinder_candidates(mesh), [])

    def test_positive_tiny_radius_not_discarded_by_length_tolerance(self):
        candidate, = cylinder_candidates(_strip(radius=.01, height=.2, sections=16))
        self.assertAlmostEqual(candidate['radius_mm'], .01, places=10)
        self.assertTrue(candidate['vertex_fit_within_tolerance'])
        self.assertGreater(candidate['radius_mm'], 0)

    def test_limited_arc_and_box_not_a_cylinder(self):
        self.assertEqual(cylinder_candidates(_strip(sections=12, coverage=40)), [])
        self.assertEqual(cylinder_candidates(trimesh.creation.box()), [])

    def test_two_disconnected_coaxial_regions_do_not_merge(self):
        first = _strip(radius=4, height=5, sections=16)
        second = first.copy()
        second.apply_translation([0, 0, 10])
        mesh = trimesh.util.concatenate([first, second])
        candidates = cylinder_candidates(mesh)
        self.assertEqual(len(candidates), 2)
        self.assertNotEqual(candidates[0]['candidate_id'], candidates[1]['candidate_id'])
        sets = [set(c['source_triangle_ids']) for c in candidates]
        self.assertFalse(sets[0] & sets[1])
        self.assertEqual(len(sets[0] | sets[1]), len(mesh.faces))
        for candidate in candidates:
            self.assertAlmostEqual(candidate['axial_length_mm'], 5.)


class InspectionEvidenceTests(unittest.TestCase):
    def test_coverage_counts_unique_triangles_not_candidate_brep(self):
        mesh = trimesh.creation.cylinder(radius=5, height=20, sections=16)
        report = inspect_mesh(mesh, 2, 3)
        coverage = report['candidate_coverage']
        self.assertEqual(coverage['total_triangle_count'], len(mesh.faces))
        # Curved panels are retained in the cylinder but their duplicate plane
        # reports are suppressed; coverage still uses the union of source faces.
        self.assertEqual(coverage['overlap_triangle_count'], 0)
        self.assertEqual(coverage['covered_triangle_count'], len(mesh.faces))
        self.assertEqual(coverage['unresolved_triangle_count'], 0)
        self.assertAlmostEqual(coverage['covered_area_mm2'], mesh.area, places=7)
        self.assertFalse(report['cad_reconstructed'])
        self.assertFalse(report['brep_reconstructed'])
        self.assertFalse(report['design_intent_recovered'])
        self.assertTrue(report['candidate_geometry_only'])
        self.assertEqual(report['mesh_self_intersection_status'],
                         'not_certified_by_geometry_candidates')
        json.dumps(report, allow_nan=False)

    def test_unresolved_regions_preserve_triangle_ids_counts_and_area(self):
        # Unsupported toroidal supports remain unresolved, rather than being
        # forced into a sphere/cylinder simply because some points fit.
        mesh = trimesh.creation.torus(major_radius=5, minor_radius=1,
                                     major_sections=64, minor_sections=24)
        report = inspect_mesh(mesh, 0, 0)
        coverage = report['candidate_coverage']
        self.assertEqual(coverage['covered_triangle_count'], 0)
        self.assertEqual(coverage['unresolved_triangle_count'], len(mesh.faces))
        self.assertEqual(coverage['unresolved_source_triangle_ids'], list(range(len(mesh.faces))))
        self.assertAlmostEqual(coverage['unresolved_area_mm2'], mesh.area)
        self.assertEqual(len(report['unresolved_regions']), 1)
        self.assertEqual(report['strategy'], 'freeform_or_unresolved')

    def test_small_isolated_planar_faces_are_reported_with_adaptive_threshold(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                        [3, 0, 0], [4, 0, 0], [3, 1, 0]],
                               faces=[[0, 1, 2], [3, 4, 5]], process=False)
        report = inspect_mesh(mesh, 0, 0)
        self.assertEqual(len(report['planes']), 2)
        self.assertEqual(report['candidate_coverage']['covered_triangle_count'], 2)
        self.assertEqual(report['strategy'], 'repair_required')

    def test_unused_vertices_not_in_report_bounds(self):
        box = trimesh.creation.box(extents=[10, 8, .5])
        mesh = trimesh.Trimesh(np.vstack((box.vertices, [[1e20, 2e20, -3e20]])),
                               box.faces, process=False)
        report = inspect_mesh(mesh, 0, 0)
        self.assertEqual(report['unused_vertices'], 1)
        self.assertEqual(report['referenced_vertices'], len(box.vertices))
        np.testing.assert_allclose(report['bounds_mm'], box.bounds)
        self.assertEqual(report['strategy'], 'plate_profile_review')

    def test_large_translation_volume_is_signed_and_recentered(self):
        mesh = trimesh.creation.box(extents=[10, 8, .5])
        mesh.apply_translation([1e11, -2e11, 3e11])
        report = inspect_mesh(mesh, 0, 0)
        self.assertAlmostEqual(report['signed_volume_mm3'], 40., places=7)
        self.assertEqual(report['strategy'], 'plate_profile_review')

    def test_closed_zero_volume_shell_is_blocked_without_abs_or_repair(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [4, 0, 0], [0, 4, 0]],
                               faces=[[0, 1, 2], [2, 1, 0]], process=False)
        report = inspect_mesh(mesh, 0, 0)
        self.assertTrue(report['watertight'])
        self.assertEqual(report['signed_volume_mm3'], 0.)
        self.assertEqual(report['strategy'], 'repair_required')
        self.assertFalse(report['mesh_repair_performed'])

    def test_optional_mesh_quality_blockers_cannot_be_overridden_by_good_fits(self):
        import sys
        import types
        evidence = types.ModuleType('mesh_evidence')
        evidence.mesh_quality = lambda mesh: {'blocking_issues': ['test_intersection_evidence']}
        with patch.dict(sys.modules, {'mesh_evidence': evidence}):
            report = inspect_mesh(trimesh.creation.box(extents=[20, 10, 1]), 0, 0)
        self.assertEqual(report['strategy'], 'repair_required')
        self.assertIn('test_intersection_evidence', report['blocking_issues'])
        self.assertFalse(report['cad_reconstructed'])

    def test_invalid_comparison_tolerances_rejected(self):
        mesh = trimesh.creation.box()
        for tolerance in (np.nan, np.inf, -.01):
            for function in (planar_candidates, cylinder_candidates):
                with self.assertRaises(ValueError):
                    function(mesh, tolerance=tolerance)
            with self.assertRaises(ValueError):
                inspect_mesh(mesh, 0, 0, tolerance=tolerance)
        with self.assertRaises(ValueError):
            planar_candidates(mesh, minimum_area=np.inf)


if __name__ == '__main__':
    unittest.main()
