"""Connected analytic-support evidence; no assertions of reconstructed CAD."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from analytic_surfaces import additional_surface_candidates


def _profile(radii=(5., 2.), heights=(-10., 10.), sections=48, coverage=360.):
    """Outward, shared-edge conical/cylindrical side without caps."""
    full = coverage == 360.
    count = sections if full else sections + 1
    angles = np.linspace(0, np.deg2rad(coverage), count, endpoint=not full)
    vertices = np.vstack([np.column_stack((r * np.cos(angles), r * np.sin(angles),
                                           np.full(count, z)))
                          for r, z in zip(radii, heights)])
    faces = []
    for level in range(len(radii) - 1):
        for i in range(sections):
            j = (i + 1) % count
            a, b = level * count, (level + 1) * count
            faces.extend([[a + i, a + j, b + j], [a + i, b + j, b + i]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _dome(radius=5., sections=48, rings=12, lower=0., upper=90., cap=False):
    """Latitude/longitude partial sphere; poles are single vertices, not repairs."""
    angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    latitudes = np.deg2rad(np.linspace(lower, upper, rings + 1))
    vertices = []
    faces = []
    regular = latitudes[:-1] if upper == 90. else latitudes
    for latitude in regular:
        vertices.extend(np.column_stack((radius * np.cos(latitude) * np.cos(angles),
                                         radius * np.cos(latitude) * np.sin(angles),
                                         np.full(sections, radius * np.sin(latitude)))))
    for j in range(len(regular) - 1):
        for i in range(sections):
            k = (i + 1) % sections
            a, b = j * sections, (j + 1) * sections
            faces.extend([[a + i, a + k, b + k], [a + i, b + k, b + i]])
    if upper == 90.:
        pole = len(vertices)
        vertices.append([0, 0, radius])
        for i in range(sections):
            a = (len(regular) - 1) * sections
            faces.append([a + i, a + (i + 1) % sections, pole])
    if cap:
        center = len(vertices)
        vertices.append([0, 0, radius * np.sin(latitudes[0])])
        for i in range(sections):
            faces.append([i, center, (i + 1) % sections])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class AnalyticSurfaceTests(unittest.TestCase):
    def candidate(self, mesh, kind, tolerance=.05, exclude=()):
        result = additional_surface_candidates(mesh, tolerance, exclude)
        self.assertEqual(result['supported_surface_types'], ['cone', 'sphere'])
        self.assertTrue(result['candidate_geometry_only'])
        self.assertFalse(result['cad_reconstructed'])
        self.assertTrue(result['requires_review'])
        self.assertEqual(len(result[kind]), 1, result['rejected_regions'])
        self.assertFalse(result['spheres' if kind == 'cones' else 'cones'])
        candidate = result[kind][0]
        self.check_evidence(mesh, candidate)
        json.dumps(result, allow_nan=False)
        return candidate, result

    def check_evidence(self, mesh, candidate):
        ids = candidate['source_triangle_ids']
        self.assertEqual(candidate['source_triangle_id_scope'], 'body_local_faces')
        self.assertEqual(candidate['triangle_count'], len(ids))
        self.assertEqual(candidate['source_vertex_ids'], np.unique(mesh.faces[ids]).tolist())
        np.testing.assert_array_equal(candidate['points_mm'], mesh.vertices[candidate['source_vertex_ids']])
        frame = candidate['frame']
        rotation = np.asarray(frame['local_to_source_rotation'])
        transform = np.asarray(frame['local_to_source_transform'])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        np.testing.assert_allclose(rotation[:, 0], candidate['ref_direction'])
        np.testing.assert_allclose(transform[:3, 3], frame['origin_mm'])
        self.assertTrue(candidate['candidate_geometry_only'])
        self.assertFalse(candidate['cad_reconstructed'])
        self.assertTrue(candidate['vertex_fit_within_tolerance'])
        self.assertTrue(candidate['normal_consistency_within_resolution'])
        self.assertLessEqual(candidate['normal_rms_degrees'], candidate['normal_max_degrees'])
        self.assertFalse(candidate['boundary']['is_cad_trim'])
        self.assertFalse(candidate['boundary']['self_intersection_checked'])
        for loop in candidate['boundary_loops']:
            np.testing.assert_array_equal(loop['points_mm'], mesh.vertices[loop['vertex_ids']])
            for i, edge in enumerate(loop['edge_uses']):
                self.assertEqual(edge['start_vertex_id'], loop['vertex_ids'][i])
                self.assertEqual(edge['end_vertex_id'], loop['vertex_ids'][i + 1])
                source_face = mesh.faces[edge['source_triangle_id']]
                j = edge['local_edge_index']
                self.assertEqual((edge['start'], edge['end']), (source_face[j], source_face[(j + 1) % 3]))
        json.dumps(candidate, allow_nan=False)

    def assert_none(self, mesh, tolerance=.05):
        result = additional_surface_candidates(mesh, tolerance)
        self.assertFalse(result['cones'])
        self.assertFalse(result['spheres'])
        json.dumps(result, allow_nan=False)
        return result

    def test_full_cone_including_apex_excludes_caps_and_records_parameters(self):
        mesh = trimesh.creation.cone(radius=5, height=20, sections=48)
        c, result = self.candidate(mesh, 'cones')
        self.assertEqual(c['triangle_count'], 48)
        self.assertEqual(len(result['excluded_source_triangle_ids']), 48)
        np.testing.assert_allclose(c['axis'], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(c['axis_point_mm'], [0, 0, 10], atol=1e-12)
        np.testing.assert_allclose(c['apex_mm'], [0, 0, 20], atol=1e-12)
        self.assertAlmostEqual(c['radius_at_origin_mm'], 2.5)
        self.assertAlmostEqual(c['base_radius_mm'], 5.)
        self.assertAlmostEqual(c['radius_slope_mm_per_mm'], -.25)
        self.assertAlmostEqual(c['semi_angle_degrees'], np.rad2deg(np.arctan(.25)))
        np.testing.assert_allclose(c['axial_parameter_range_mm'], [-10., 10.], atol=1e-12)
        self.assertTrue(c['wraps_full_circumference'])
        self.assertTrue(c['surface_approximation_within_tolerance'])
        self.assertFalse(c['surface_residual_is_exact'])
        self.assertEqual(c['boundary']['boundary_edge_count'], 48)
        self.assertEqual(len(c['boundary_loops']), 1)
        self.assertLess(c['fit_condition_number'], 10.)

    def test_frustum_partial_and_full_angular_bands(self):
        for coverage in (60., 90., 180., 360.):
            with self.subTest(coverage=coverage):
                mesh = _profile(sections=24, coverage=coverage)
                c, _ = self.candidate(mesh, 'cones')
                self.assertEqual(c['source_triangle_ids'], list(range(len(mesh.faces))))
                self.assertAlmostEqual(c['angular_coverage_degrees'], coverage, places=6)
                self.assertAlmostEqual(c['radius_at_origin_mm'], 3.5, places=9)
                self.assertAlmostEqual(c['radius_slope_mm_per_mm'], -.15, places=9)
                self.assertAlmostEqual(c['base_radius_mm'], 5., places=9)
                self.assertAlmostEqual(c['semi_angle_degrees'], np.rad2deg(np.arctan(.15)), places=8)

    def test_cone_rotations_uniform_scales_and_translations(self):
        transform = trimesh.transformations.rotation_matrix(1.19, [2, -3, 4])
        translation = np.array([31., -27., 13.])
        for scale in (.01, 1., 100.):
            for radii in ((5., 2.), (5., 5.001)):
                with self.subTest(scale=scale, radii=radii):
                    mesh = _profile(radii=radii, sections=48)
                    mesh.apply_scale(scale)
                    mesh.apply_transform(transform)
                    mesh.apply_translation(translation)
                    c, _ = self.candidate(mesh, 'cones', tolerance=.05 * scale)
                    self.assertGreater(abs(c['radius_slope_mm_per_mm']), 0.)
                    self.assertAlmostEqual(abs(c['radius_slope_mm_per_mm']), abs(radii[1] - radii[0]) / 20., places=9)
                    self.assertAlmostEqual(c['radius_at_origin_mm'], np.mean(radii) * scale, places=8)
                    self.assertAlmostEqual(c['axial_length_mm'], 20. * scale, places=8)
                    np.testing.assert_allclose(c['axis_point_mm'], translation, atol=1e-8)
                    self.assertAlmostEqual(abs(np.dot(c['axis'], transform[:3, 2])), 1.)

    def test_shallow_cone_is_not_silently_flattened_even_at_large_translation(self):
        for delta in (-.005, -.0001, .0001, .005):
            with self.subTest(delta=delta):
                mesh = _profile(radii=(5., 5. + delta), sections=32)
                transform = trimesh.transformations.rotation_matrix(1.23, [3, -2, 1])
                mesh.apply_transform(transform)
                mesh.apply_translation([1e9, -2e9, 3e9])
                c, _ = self.candidate(mesh, 'cones', tolerance=.05)
                self.assertGreater(c['semi_angle_degrees'], 0.)
                self.assertAlmostEqual(abs(c['radius_slope_mm_per_mm']), abs(delta) / 20., delta=3e-8)
                self.assertLess(c['max_vertex_residual_mm'], 2e-6)
                # The extrapolated apex of a shallow cone is intentionally far
                # outside the observed region, not a confirmed physical feature.
                self.assertGreater(abs(c['apex_axial_parameter_mm']), c['axial_length_mm'])

    def test_full_cone_apex_remains_stable_under_rotation_and_large_translation(self):
        mesh = trimesh.creation.cone(radius=5., height=20., sections=48)
        transform = trimesh.transformations.rotation_matrix(1.23, [3, -2, 1])
        translation = np.array([1e9, -2e9, 3e9])
        mesh.apply_transform(transform)
        mesh.apply_translation(translation)
        c, _ = self.candidate(mesh, 'cones')
        self.assertAlmostEqual(abs(c['radius_slope_mm_per_mm']), .25, delta=1e-8)
        self.assertLess(c['max_vertex_residual_mm'], 1e-6)
        np.testing.assert_allclose(c['apex_mm'], transform[:3, :3] @ [0., 0., 20.] + translation,
                                   rtol=0, atol=1e-6)

    def test_cylinder_is_never_a_cone_or_sphere_even_with_two_cospherical_rings(self):
        for height in (.5, 1., 2., 5., 20.):
            with self.subTest(height=height):
                self.assert_none(_profile(radii=(5., 5.), heights=(-height / 2, height / 2)))
        mesh = _profile(radii=(5., 5.))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(1.23, [3, -2, 1]))
        mesh.apply_translation([1e9, -2e9, 3e9])
        self.assert_none(mesh)

    def test_mixed_cylinder_cone_band_and_elliptically_deformed_cone_rejected(self):
        mixed = _profile(radii=(5., 5., 3.), heights=(-10., 0., 10.), sections=48)
        r = self.assert_none(mixed)
        self.assertEqual(len(r['rejected_regions']), 1)
        self.assertEqual(r['rejected_regions'][0]['triangle_count'], len(mixed.faces))
        for scale_x in (1.001, 1.01, 1.1, 1.4):
            mesh = _profile()
            mesh.apply_scale([scale_x, 1, 1])
            # Even lenient point tolerance cannot suppress normal inconsistency.
            self.assert_none(mesh, tolerance=10.)

    def test_cone_requires_sixty_degrees_and_twelve_connected_triangles(self):
        r = self.assert_none(_profile(coverage=30., sections=16))
        self.assertEqual(r['rejected_regions'][0]['fit_failures']['cone']['reason'], 'cone_insufficient_angular_extent')
        r = self.assert_none(_profile(coverage=90., sections=4))
        self.assertEqual(r['rejected_regions'][0]['fit_failures']['cone']['reason'], 'insufficient_connected_triangles')

    def test_fine_closed_sphere_has_center_normal_evidence_and_empty_boundary(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=5)
        c, result = self.candidate(mesh, 'spheres')
        np.testing.assert_allclose(c['center_mm'], [0, 0, 0], atol=1e-12)
        self.assertAlmostEqual(c['radius_mm'], 5.)
        self.assertEqual(c['source_triangle_ids'], list(range(len(mesh.faces))))
        self.assertFalse(c['boundary_loops'])
        self.assertEqual(c['boundary']['boundary_edge_count'], 0)
        self.assertTrue(c['surface_residual_is_exact'])
        self.assertTrue(c['surface_approximation_within_tolerance'])
        self.assertLess(c['normal_max_degrees'], 1.)
        self.assertFalse(result['rejected_regions'])
        self.assertLess(c['fit_condition_number'], 2.)

    def test_hemisphere_open_and_capped_is_sphere_not_cone(self):
        for cap in (False, True):
            with self.subTest(cap=cap):
                mesh = _dome(cap=cap)
                c, result = self.candidate(mesh, 'spheres')
                self.assertAlmostEqual(c['radius_mm'], 5.)
                np.testing.assert_allclose(c['center_mm'], [0, 0, 0], atol=1e-12)
                self.assertEqual(c['boundary']['boundary_edge_count'], 48)
                self.assertEqual(len(c['boundary_loops']), 1)
                self.assertTrue(c['boundary_loops'][0]['is_closed'])
                self.assertEqual(len(result['excluded_source_triangle_ids']), 48 if cap else 0)
                self.assertEqual(c['triangle_count'], len(mesh.faces) - (48 if cap else 0))

    def test_spheres_rotated_scaled_and_large_translated_are_conditioned_locally(self):
        transform = trimesh.transformations.rotation_matrix(.73, [2, -3, 4])
        for maker in (lambda: trimesh.creation.icosphere(subdivisions=2, radius=5), _dome):
            for scale in (.01, 1., 100.):
                with self.subTest(maker=maker, scale=scale):
                    mesh = maker()
                    mesh.apply_scale(scale)
                    mesh.apply_transform(transform)
                    translation = [1e10, -2e10, 3e10] if scale == 1. else [19., -37., 23.]
                    mesh.apply_translation(translation)
                    c, _ = self.candidate(mesh, 'spheres', tolerance=.05 * scale)
                    np.testing.assert_allclose(c['center_mm'], translation, atol=1e-5 if scale == 1. else 1e-10)
                    self.assertAlmostEqual(c['radius_mm'], 5. * scale, delta=4e-6 if scale == 1. else 1e-9)
                    self.assertLess(c['fit_condition_number'], 3.)

    def test_tiny_spherical_patch_and_fillet_like_spherical_band_are_rejected(self):
        for mesh in (_dome(lower=80.), _dome(lower=0., upper=5.)):
            with self.subTest(faces=len(mesh.faces)):
                r = self.assert_none(mesh)
                self.assertTrue(r['rejected_regions'])
                reason = r['rejected_regions'][0]['fit_failures']['sphere']['reason']
                self.assertIn(reason, ('sphere_normals_lack_two_dimensional_curvature',
                                      'sphere_insufficient_two_dimensional_arc_extent',
                                      'sphere_fit_rank_or_condition_deficient'))

    def test_ellipsoid_and_torus_are_not_reported_as_supported_surfaces(self):
        ellipsoid = trimesh.creation.icosphere(subdivisions=3, radius=5)
        ellipsoid.apply_scale([1., 1.3, 1.7])
        torus = trimesh.creation.torus(major_radius=8, minor_radius=2,
                                       major_sections=48, minor_sections=24)
        for mesh in (ellipsoid, torus):
            with self.subTest(faces=len(mesh.faces)):
                r = self.assert_none(mesh)
                self.assertEqual(len(r['rejected_regions']), 1)
                self.assertEqual(r['rejected_regions'][0]['source_triangle_ids'], list(range(len(mesh.faces))))

    def test_boxes_single_plates_and_coplanar_cap_fans_never_become_curved_surfaces(self):
        box = trimesh.creation.box(extents=[20., 12., .3])
        plate = trimesh.Trimesh(vertices=[[0, 0, 0], [20, 0, 0], [20, 12, 0], [0, 12, 0]],
                                faces=[[0, 1, 2], [0, 2, 3]], process=False)
        cylinder = trimesh.creation.cylinder(radius=5., height=20., sections=48)
        cylinder.faces = cylinder.faces[np.abs(cylinder.face_normals[:, 2]) > .99]
        for mesh in (box, plate, cylinder):
            with self.subTest(faces=len(mesh.faces)):
                r = self.assert_none(mesh)
                self.assertEqual(r['excluded_source_triangle_ids'], list(range(len(mesh.faces))))

    def test_coarse_cone_vertices_fit_but_mesh_interior_exceeds_tolerance(self):
        for sections in (12, 16):
            c, _ = self.candidate(_profile(sections=sections), 'cones')
            self.assertLess(c['max_vertex_residual_mm'], 1e-10)
            self.assertGreater(c['max_centroid_residual_mm'], .05)
            self.assertGreater(c['max_sampled_surface_residual_mm'], .05)
            self.assertGreaterEqual(c['max_surface_residual_upper_bound_mm'], c['max_sampled_surface_residual_mm'])
            self.assertEqual(c['max_surface_residual_mm'], c['max_surface_residual_upper_bound_mm'])
            self.assertFalse(c['surface_approximation_within_tolerance'])
            self.assertFalse(c['surface_residual_is_exact'])
            self.assertEqual(c['surface_tolerance_assessment'], 'sampled_error_exceeds_tolerance')

    def test_coarse_sphere_exact_triangle_interior_distance_without_rtree(self):
        # Non-equilateral UV triangles have an interior radial minimum away
        # from the centroid, so a centroid-only implementation cannot pass.
        mesh = trimesh.creation.uv_sphere(radius=5, count=(8, 16))
        with patch('trimesh.proximity.closest_point', side_effect=AssertionError('must not query rtree')):
            c, _ = self.candidate(mesh, 'spheres')
        self.assertLess(c['max_vertex_residual_mm'], 1e-10)
        self.assertGreater(c['max_surface_residual_mm'], .05)
        self.assertFalse(c['surface_approximation_within_tolerance'])
        radial_triangles = mesh.triangles - np.asarray(c['center_mm'])
        closest = trimesh.triangles.closest_point(radial_triangles, np.zeros((len(mesh.faces), 3)))
        expected = c['radius_mm'] - np.linalg.norm(closest, axis=1).min()
        self.assertAlmostEqual(c['max_surface_residual_mm'], expected)
        self.assertTrue(c['surface_residual_is_exact'])
        self.assertGreater(c['max_surface_residual_mm'], c['max_centroid_residual_mm'])

    def test_cone_conservative_upper_bound_dominates_interior_grid_samples(self):
        mesh = _profile(sections=12, coverage=180.)
        c, _ = self.candidate(mesh, 'cones')
        rotation = np.asarray(c['frame']['local_to_source_rotation'])
        local = (mesh.triangles - np.asarray(c['axis_point_mm'])) @ rotation
        barycentric = np.asarray([[i / 12, j / 12, (12 - i - j) / 12]
                                  for i in range(13) for j in range(13 - i)])
        points = np.einsum('sk,fkj->fsj', barycentric, local)
        rho = np.linalg.norm(points[..., :2], axis=-1)
        error = np.abs(rho - c['radius_at_origin_mm'] - c['radius_slope_mm_per_mm'] * points[..., 2])
        error /= np.sqrt(1 + c['radius_slope_mm_per_mm'] ** 2)
        self.assertLessEqual(error.max(), c['max_surface_residual_upper_bound_mm'] + 1e-12)

    def test_exclusions_split_connected_bands_do_not_bridge_or_merge(self):
        mesh = _profile(sections=128)
        excluded = list(range(16)) + list(range(128, 144))
        r = additional_surface_candidates(mesh, exclude_triangle_ids=excluded + excluded[:2])
        self.assertEqual(len(r['cones']), 2, r['rejected_regions'])
        self.assertFalse(r['spheres'])
        actual = set()
        for c in r['cones']:
            self.check_evidence(mesh, c)
            self.assertFalse(set(c['source_triangle_ids']) & set(excluded))
            self.assertFalse(actual & set(c['source_triangle_ids']))
            actual.update(c['source_triangle_ids'])
            self.assertAlmostEqual(c['angular_coverage_degrees'], 157.5)
        self.assertEqual(actual, set(range(len(mesh.faces))) - set(excluded))
        self.assertEqual(r['caller_excluded_source_triangle_ids'], sorted(excluded))
        self.assertEqual(r['excluded_source_triangle_ids'], sorted(excluded))
        r = additional_surface_candidates(mesh, exclude_triangle_ids=range(len(mesh.faces)))
        self.assertFalse(r['cones'] + r['spheres'] + r['rejected_regions'])

    def test_disconnected_bodies_are_never_whole_shape_fit_or_coaxial_merge(self):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=5.)
        second = sphere.copy()
        second.apply_translation([30., 0., 0.])
        cone = trimesh.creation.cone(radius=5, height=20, sections=48)
        cone.apply_translation([60., 0., 0.])
        box = trimesh.creation.box(extents=[3, 4, 5])
        box.apply_translation([90., 0., 0.])
        mesh = trimesh.util.concatenate([sphere, second, cone, box])
        r = additional_surface_candidates(mesh)
        self.assertEqual(len(r['spheres']), 2)
        self.assertEqual(len(r['cones']), 1)
        used = set()
        for c in r['spheres'] + r['cones']:
            self.check_evidence(mesh, c)
            self.assertFalse(used & set(c['source_triangle_ids']))
            used.update(c['source_triangle_ids'])
        self.assertEqual(len(used), len(sphere.faces) * 2 + 48)
        self.assertEqual(len(r['excluded_source_triangle_ids']), 48 + len(box.faces))

    def test_small_curved_regions_beside_large_plate_use_local_conditioning(self):
        plate = trimesh.creation.box(extents=[1000., 500., 2.])
        for small, kind in ((_profile(radii=(.05, .02), heights=(-.1, .1)), 'cones'),
                            (trimesh.creation.icosphere(subdivisions=2, radius=.05), 'spheres')):
            with self.subTest(kind=kind):
                small.apply_translation([3000., 17., 23.])
                mesh = trimesh.util.concatenate([plate, small])
                c, _ = self.candidate(mesh, kind, tolerance=.005)
                self.assertEqual(c['source_triangle_ids'],
                                 list(range(len(plate.faces), len(mesh.faces))))
                self.assertLess(c['fit_condition_number'], 3.)
                self.assertLess(c['max_vertex_residual_mm'], 1e-10)

    def test_deterministic_no_random_sampling_normal_repair_or_input_mutation(self):
        for mesh, kind in ((trimesh.creation.icosphere(subdivisions=2, radius=5), 'spheres'),
                           (_profile(), 'cones')):
            mesh.faces = mesh.faces[:, ::-1]
            vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
            with patch.object(trimesh.Trimesh, 'fix_normals', side_effect=AssertionError('no repair')), \
                    patch('numpy.random.default_rng', side_effect=AssertionError('no random guesses')):
                c, first = self.candidate(mesh, kind)
                second = additional_surface_candidates(mesh)
            self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
            self.assertEqual(c['normal_orientation'], 'inward')
            self.assertGreater(c['inward_normal_area_fraction'], .99)
            np.testing.assert_array_equal(mesh.vertices, vertices)
            np.testing.assert_array_equal(mesh.faces, faces)

    def test_invalid_inputs_tolerances_and_exclusion_ids_raise_value_error(self):
        invalid_meshes = [trimesh.Trimesh(vertices=[], faces=[], process=False),
                          trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                                          faces=[[0, 1, 2]], process=False),
                          trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [np.inf, 1, 0]],
                                          faces=[[0, 1, 2]], process=False)]
        for mesh in invalid_meshes:
            with self.assertRaises(ValueError):
                additional_surface_candidates(mesh)
        mesh = _profile()
        for tolerance in (-1, np.nan, np.inf, 'invalid'):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                additional_surface_candidates(mesh, tolerance=tolerance)
        for ids in (None, 3, [-1], [len(mesh.faces)], [0.], [True], [np.nan]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                additional_surface_candidates(mesh, exclude_triangle_ids=ids)


if __name__ == '__main__':
    unittest.main()
