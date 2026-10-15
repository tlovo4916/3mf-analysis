"""Verifiable support relationships/direction hints, never design/CAD certificates."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from analytic_surfaces import additional_surface_candidates
import feature_relations
from feature_relations import feature_relationships, section_directions
from geometry_analysis import cylinder_candidates, planar_candidates, plate_frame


def _frame(origin=(0., 0., 0.), rotation=None, axis=None):
    origin = np.asarray(origin, dtype=float)
    if rotation is None:
        z = np.asarray(axis if axis is not None else (0., 0., 1.), dtype=float)
        z /= np.linalg.norm(z)
        seed = np.eye(3)[np.argmin(np.abs(z))]
        x = seed - z * np.dot(seed, z)
        x /= np.linalg.norm(x)
        rotation = np.column_stack((x, np.cross(z, x), z))
    rotation = np.asarray(rotation, dtype=float)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, origin
    return {'origin_mm': origin.tolist(), 'local_to_source_rotation': rotation.tolist(),
            'local_to_source_transform': transform.tolist(),
            'x_direction': rotation[:, 0].tolist(), 'y_direction': rotation[:, 1].tolist(),
            'z_direction': rotation[:, 2].tolist(), 'coordinate_frame': 'source_mesh_mm',
            'handedness': 'right'}


def _plane(name, normal=(0., 0., 1.), point=(0., 0., 0.), area=10.):
    return {'candidate_id': name, 'type': 'plane', 'normal': list(normal), 'point_mm': list(point),
            'frame': _frame(point, axis=normal), 'area_mm2': area, 'candidate_geometry_only': True,
            'source_triangle_ids': [], 'surface_approximation_within_tolerance': True}


def _surface(name, kind='cylinder', axis=(0., 0., 1.), origin=(0., 0., 0.),
             radius=5., interval=(-5., 5.), coverage=360., orientation='outward', area=100.):
    result = {'candidate_id': name, 'type': kind, 'axis': list(axis), 'axis_point_mm': list(origin),
              'frame': _frame(origin, axis=axis), 'axial_parameter_range_mm': list(interval),
              'axial_length_mm': interval[1] - interval[0], 'area_mm2': area,
              'angular_coverage_degrees': coverage,
              'angular_parameter_intervals_radians': [[0., math.radians(coverage)]],
              'wraps_full_circumference': coverage >= 360. - 1e-6,
              'normal_orientation': orientation, 'source_triangle_ids': [],
              'candidate_geometry_only': True, 'surface_approximation_within_tolerance': True}
    if kind == 'cylinder':
        result['radius_mm'] = radius
    else:
        result.update({'radius_at_origin_mm': radius, 'radius_slope_mm_per_mm': .1,
                       'semi_angle_degrees': math.degrees(math.atan(.1)),
                       'radius_range_mm': [radius + .1 * interval[0], radius + .1 * interval[1]],
                       'base_radius_mm': radius + .1 * interval[1],
                       'apex_axial_parameter_mm': -radius / .1,
                       'apex_mm': (np.asarray(origin) - radius / .1 * np.asarray(axis)).tolist()})
    return result


def _record(planes=(), cylinders=(), cones=(), spheres=(), rotation=None, origin=(0., 0., 0.),
            extents=(20., 10., 5.)):
    frame = _frame(origin, rotation=rotation if rotation is not None else np.eye(3))
    return {'planes': list(planes), 'cylinders': list(cylinders), 'cones': list(cones),
            'spheres': list(spheres), 'local_to_source_rotation': frame['local_to_source_rotation'],
            'source_origin_mm': frame['origin_mm'], 'oriented_extents_mm': list(extents),
            'thinness_ratio': extents[2] / min(extents[:2]),
            'strategy': 'repair_required', 'blocking_issues': ['source_review_blocker'],
            'requires_review': True, 'cad_reconstructed': False}


def _profile(radii=(5., 2.), interval=(-10., 10.), sections=48, coverage=360., inward=False):
    full = coverage == 360.
    count = sections if full else sections + 1
    angles = np.linspace(0, math.radians(coverage), count, endpoint=not full)
    vertices = np.vstack([np.column_stack((radius * np.cos(angles), radius * np.sin(angles),
                                          np.full(count, z))) for radius, z in zip(radii, interval)])
    faces = []
    for i in range(sections):
        j = (i + 1) % count
        faces.extend([[i, j, j + count], [i, j + count, i + count]])
    faces = np.asarray(faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces[:, ::-1] if inward else faces, process=False)


def _mesh_record(mesh, extra=False):
    rotation, origin, local = plate_frame(mesh)
    record = _record(rotation=rotation, origin=origin, extents=local.extents)
    record['planes'] = planar_candidates(mesh)
    record['cylinders'] = cylinder_candidates(mesh)
    if extra:
        excluded = {i for candidate in record['cylinders'] for i in candidate['source_triangle_ids']}
        additional = additional_surface_candidates(mesh, exclude_triangle_ids=excluded)
        record['cones'], record['spheres'] = additional['cones'], additional['spheres']
    return record


class FeatureRelationshipTests(unittest.TestCase):
    def check_result(self, result):
        json.dumps(result, allow_nan=False)
        self.assertTrue(result['candidate_geometry_only'])
        self.assertTrue(result['requires_review'])
        self.assertFalse(result['cad_reconstructed'])
        self.assertFalse(result['design_intent_recovered'])
        self.assertFalse(result['affects_review_gate'])
        self.assertIn(result['classification_candidate'], (
            'prismatic_sections_review', 'revolved_part_review', 'housing_or_shell_review',
            'mixed_mechanical_review', 'freeform_review'))
        self.assertNotIn('confirmed_feature', json.dumps(result))
        coverage = result['comparison_coverage']
        self.assertLessEqual(coverage['compared_pair_count'], 512)
        self.assertEqual(coverage['eligible_pair_count'], coverage['compared_pair_count'] + coverage['omitted_pair_count'])
        self.assertEqual(coverage['compared_pair_count'], sum(
            item['compared_pair_count'] for item in coverage['by_kind'].values()))

    def test_box_parallel_orthogonal_distances_and_no_gate_mutation(self):
        mesh = trimesh.creation.box(extents=[40., 15., 4.])
        record = _mesh_record(mesh)
        before, vertices, faces = copy.deepcopy(record), mesh.vertices.copy(), mesh.faces.copy()
        result = feature_relationships(mesh, record)
        self.check_result(result)
        parallel = [item for item in result['plane_relationships'] if item['type'] == 'parallel_plane_candidate']
        orthogonal = [item for item in result['plane_relationships'] if item['type'] == 'orthogonal_plane_candidate']
        self.assertEqual(len(parallel), 3)
        self.assertEqual(len(orthogonal), 12)
        np.testing.assert_allclose(sorted(item['distance_mm'] for item in parallel), [4., 15., 40.])
        self.assertTrue(all(item['distance_is_constant_on_supports'] for item in parallel))
        self.assertEqual(result['classification_candidate'], 'prismatic_sections_review')
        self.assertEqual(record, before)
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)

    def test_rotated_translated_box_keeps_world_plane_distances(self):
        mesh = trimesh.creation.box(extents=[30., 14., 3.])
        mesh.apply_transform(trimesh.transformations.rotation_matrix(1.13, [2, -3, 4]))
        mesh.apply_translation([1e6, -2e6, 3e6])
        record = _mesh_record(mesh)
        result = feature_relationships(mesh, record)
        parallel = [item for item in result['plane_relationships'] if item['type'] == 'parallel_plane_candidate']
        np.testing.assert_allclose(sorted(item['distance_mm'] for item in parallel), [3., 14., 30.], atol=1e-8)
        self.check_result(result)

    def test_near_parallel_planes_report_only_anchor_projections_not_constant_distance(self):
        angle = math.radians(.25)
        p = _plane('plane0')
        q = _plane('plane1', normal=(math.sin(angle), 0., math.cos(angle)), point=(0., 0., 2.))
        result = feature_relationships(None, _record(planes=[p, q]))
        relation, = result['plane_relationships']
        self.assertEqual(relation['type'], 'parallel_plane_candidate')
        self.assertAlmostEqual(relation['normal_angle_degrees'], .25)
        self.assertIsNone(relation['distance_mm'])
        self.assertFalse(relation['distance_is_constant_on_supports'])
        np.testing.assert_allclose(relation['anchor_projected_distances_mm'], [2., 2. * math.cos(angle)])

    def test_plane_angular_threshold_independent_of_length_tolerance(self):
        p = _plane('p')
        q = _plane('q', normal=(math.sin(math.radians(1)), 0., math.cos(math.radians(1))))
        result = feature_relationships(None, _record(planes=[p, q]), tolerance=1e6)
        self.assertFalse(result['plane_relationships'])

    def test_real_tube_radial_gap_preserves_independent_cylinder_candidates(self):
        inner = _profile(radii=(3., 3.), inward=True)
        outer = _profile(radii=(5., 5.))
        mesh = trimesh.util.concatenate([inner, outer])
        record = _mesh_record(mesh)
        self.assertEqual(len(record['cylinders']), 2)
        original = copy.deepcopy(record)
        result = feature_relationships(mesh, record)
        self.check_result(result)
        group, = result['coaxial_surface_groups']
        self.assertEqual(len(group['candidate_refs']), 2)
        self.assertFalse(group['surfaces_merged'])
        gap, = result['geometric_radial_gap_candidates']
        self.assertAlmostEqual(gap['radial_gap_mm'], 2.)
        self.assertAlmostEqual(gap['axial_overlap_length_mm'], 20.)
        self.assertFalse(gap['wall_thickness_certified'])
        self.assertEqual(gap['type'], 'geometric_radial_gap_candidate')
        self.assertEqual(result['classification_candidate'], 'housing_or_shell_review')
        self.assertEqual(record, original)
        self.assertEqual(result['thresholds']['axis_angular_threshold_degrees'], .5)
        self.assertEqual(result['thresholds']['full_circumference_shortfall_threshold_degrees'], 1e-6)

    def test_full_and_partial_cylinders_only_full_pair_gets_radial_gap(self):
        for angular_extent in (60., 180., 359.99, 360.):
            with self.subTest(angular_extent=angular_extent):
                mesh = trimesh.util.concatenate([_profile(radii=(3., 3.), inward=True),
                                                 _profile(radii=(5., 5.), coverage=angular_extent)])
                record = _mesh_record(mesh)
                self.assertEqual(len(record['cylinders']), 2)
                result = feature_relationships(mesh, record)
                self.assertEqual(len(result['coaxial_surface_groups']), 1)
                self.assertEqual(len(result['geometric_radial_gap_candidates']), int(angular_extent == 360.))
                self.check_result(result)

    def test_cone_and_cylinder_coaxial_but_not_a_cylindrical_wall_gap(self):
        for coverage in (90., 180., 360.):
            with self.subTest(coverage=coverage):
                cylinder = _profile(radii=(5., 5.))
                cone = _profile(radii=(3., 1.), coverage=coverage)
                mesh = trimesh.util.concatenate([cylinder, cone])
                record = _mesh_record(mesh, extra=True)
                self.assertEqual(len(record['cylinders']), 1)
                self.assertEqual(len(record['cones']), 1)
                result = feature_relationships(mesh, record)
                self.assertEqual(len(result['coaxial_surface_groups']), 1)
                self.assertFalse(result['geometric_radial_gap_candidates'])
                self.assertEqual(result['classification_candidate'], 'revolved_part_review')
                self.check_result(result)

    def test_angle_and_distance_are_separate_coaxial_tests(self):
        a = _surface('a')
        cases = [(.49, .01, 1), (.51, .01, 0), (0., .05, 1), (0., .050001, 0), (.49, .050001, 0)]
        for degrees, distance, matches in cases:
            with self.subTest(degrees=degrees, distance=distance):
                b = _surface('b', axis=(math.sin(math.radians(degrees)), 0., math.cos(math.radians(degrees))),
                             origin=(0., distance, 20.))
                result = feature_relationships(None, _record(cylinders=[a, b]))
                self.assertEqual(len(result['coaxial_surface_groups']), matches)
        b = _surface('b', axis=(math.sin(math.radians(1)), 0., math.cos(math.radians(1))))
        self.assertFalse(feature_relationships(None, _record(cylinders=[a, b]), tolerance=1e9)['coaxial_surface_groups'])

    def test_opposite_axis_and_translated_parameter_intervals_overlap_correctly(self):
        a = _surface('inner', radius=3., interval=(0., 10.), orientation='inward')
        b = _surface('outer', radius=5., axis=(0., 0., -1.), origin=(0., 0., 15.), interval=(8., 12.))
        result = feature_relationships(None, _record(cylinders=[a, b]))
        group, = result['coaxial_surface_groups']
        self.assertAlmostEqual(group['axis_angle_degrees'], 0.)
        self.assertAlmostEqual(group['line_distance_mm'], 0.)
        gap, = result['geometric_radial_gap_candidates']
        np.testing.assert_allclose(gap['axial_overlap_parameter_range_mm'], [3., 7.])
        self.assertAlmostEqual(gap['axial_overlap_length_mm'], 4.)
        self.assertFalse(gap['wall_thickness_certified'])

    def test_rotated_tube_gap_uses_source_frame_and_is_translation_invariant(self):
        rotation = trimesh.transformations.rotation_matrix(.97, [3., -1., 2.])[:3, :3]
        for translation in (np.array([17., -30., 5.]), np.array([1e9, -2e9, 3e9])):
            with self.subTest(translation=translation):
                mesh = trimesh.util.concatenate([_profile(radii=(3., 3.), inward=True), _profile(radii=(5., 5.))])
                mesh.apply_transform(trimesh.transformations.rotation_matrix(.97, [3., -1., 2.]))
                mesh.apply_translation(translation)
                record = _mesh_record(mesh)
                result = feature_relationships(mesh, record)
                self.assertEqual(len(result['coaxial_surface_groups']), 1)
                gap, = result['geometric_radial_gap_candidates']
                self.assertAlmostEqual(gap['radial_gap_mm'], 2., delta=2e-6)
                self.assertAlmostEqual(gap['axial_overlap_length_mm'], 20., delta=2e-6)
                self.assertAlmostEqual(abs(np.dot(gap['reference_axis'], rotation[:, 2])), 1., delta=1e-10)
                self.check_result(result)

    def test_disjoint_and_endpoint_only_intervals_do_not_establish_gap(self):
        a = _surface('inner', radius=3., orientation='inward', interval=(0., 10.))
        for origin in (10., 10.0001, 100.):
            with self.subTest(origin=origin):
                b = _surface('outer', origin=(0., 0., origin), interval=(0., 5.))
                result = feature_relationships(None, _record(cylinders=[a, b]))
                self.assertEqual(len(result['coaxial_surface_groups']), 1)
                self.assertFalse(result['geometric_radial_gap_candidates'])

    def test_winding_role_and_radius_evidence_cannot_be_assumed_or_repaired(self):
        a = _surface('inner', radius=3., orientation='inward')
        b = _surface('outer')
        cases = []
        for label in ('mixed', 'unknown', 'inward'):
            candidate = copy.deepcopy(b)
            candidate['normal_orientation'] = label
            cases.append(candidate)
        candidate = copy.deepcopy(b)
        del candidate['normal_orientation']
        candidate['role_candidate'] = 'boss_or_outer_surface'
        cases.append(candidate)
        candidate = copy.deepcopy(b)
        candidate['hole'] = {'inward_normal_area_fraction': 1.}
        cases.append(candidate)
        for radius in (2., 3.):
            candidate = copy.deepcopy(b)
            candidate['radius_mm'] = radius
            cases.append(candidate)
        for index, candidate in enumerate(cases):
            with self.subTest(case=index):
                result = feature_relationships(None, _record(cylinders=[a, candidate]))
                self.assertFalse(result['geometric_radial_gap_candidates'])
        classic = copy.deepcopy(a)
        del classic['normal_orientation']
        classic['hole'] = {'inward_normal_area_fraction': 1., 'confirmed': False}
        self.assertEqual(len(feature_relationships(None, _record(cylinders=[classic, b]))['geometric_radial_gap_candidates']), 1)

    def test_full_angular_threshold_must_also_agree_with_observed_intervals(self):
        inner = _surface('inner', radius=3., orientation='inward')
        for extent, wraps, expect in ((360. - 5e-7, True, 1), (360. - 2e-6, True, 0), (360., False, 0)):
            outer = _surface('outer', coverage=extent)
            outer['wraps_full_circumference'] = wraps
            result = feature_relationships(None, _record(cylinders=[inner, outer]))
            self.assertEqual(len(result['geometric_radial_gap_candidates']), expect)
        outer = _surface('outer')
        # A mistaken full label cannot overwrite partial interval evidence.
        outer['angular_parameter_intervals_radians'] = [[0., math.pi], [0., math.pi]]
        self.assertFalse(feature_relationships(None, _record(cylinders=[inner, outer]))['geometric_radial_gap_candidates'])
        del outer['angular_parameter_intervals_radians']
        del outer['angular_coverage_degrees']
        self.assertFalse(feature_relationships(None, _record(cylinders=[inner, outer]))['geometric_radial_gap_candidates'])

    def test_pairwise_coaxial_groups_do_not_create_transitive_distance_chain(self):
        candidates = [_surface(name, origin=(offset, 0., 0.))
                      for name, offset in (('a', 0.), ('b', .04), ('c', .08))]
        record = _record(cylinders=candidates)
        result = feature_relationships(None, record)
        members = {tuple(group['candidate_ids']) for group in result['coaxial_surface_groups']}
        self.assertEqual(members, {('a', 'b'), ('b', 'c')})
        self.assertEqual(result['classification_candidate'], 'mixed_mechanical_review')
        self.assertTrue(all(len(group['candidate_refs']) == 2 for group in result['coaxial_surface_groups']))
        self.assertEqual(record['cylinders'], candidates)

    def test_pairwise_coaxial_groups_do_not_create_transitive_angle_chain(self):
        candidates = [_surface(name, axis=(math.sin(math.radians(degrees)), 0., math.cos(math.radians(degrees))))
                      for name, degrees in (('a', 0.), ('b', .4), ('c', .8))]
        result = feature_relationships(None, _record(cylinders=candidates))
        self.assertEqual({tuple(group['candidate_ids']) for group in result['coaxial_surface_groups']}, {('a', 'b'), ('b', 'c')})

    def test_three_compatible_surfaces_are_independent_pairs_not_union(self):
        values = [_surface('a'), _surface('b'), _surface('c', kind='cone')]
        result = feature_relationships(None, _record(cylinders=values[:2], cones=values[2:]))
        self.assertEqual(len(result['coaxial_surface_groups']), 3)
        self.assertTrue(all(group['all_members_pairwise_compatible'] for group in result['coaxial_surface_groups']))
        self.assertTrue(all(not group['surfaces_merged'] for group in result['coaxial_surface_groups']))
        self.assertEqual(result['classification_candidate'], 'revolved_part_review')

    def test_spheres_have_no_inferred_axis_and_mixed_axes_have_review_only_class(self):
        spheres = [{'candidate_id': 'sphere', 'type': 'sphere', 'center_mm': [0., 0., 0.],
                    'radius_mm': 5., 'frame': _frame(), 'area_mm2': 100.}]
        result = feature_relationships(None, _record(spheres=spheres))
        self.assertFalse(result['coaxial_surface_groups'])
        self.assertEqual(result['comparison_coverage']['axisless_sphere_candidate_count'], 1)
        self.assertEqual(result['classification_candidate'], 'freeform_review')
        cylinders = [_surface('z'), _surface('x', axis=(1., 0., 0.))]
        result = feature_relationships(None, _record(cylinders=cylinders, spheres=spheres))
        self.assertFalse(result['coaxial_surface_groups'])
        self.assertEqual(result['classification_candidate'], 'mixed_mechanical_review')
        self.check_result(result)

    def test_empty_mesh_record_and_single_candidate_coverage(self):
        for mesh in (None, trimesh.Trimesh()):
            result = feature_relationships(mesh, {})
            self.check_result(result)
            self.assertEqual(result['classification_candidate'], 'freeform_review')
            self.assertFalse(result['plane_relationships'] + result['coaxial_surface_groups'] + result['geometric_radial_gap_candidates'])
            self.assertTrue(result['comparison_coverage']['complete'])
            self.assertEqual(result['comparison_coverage']['pair_coverage_fraction'], 1.)
        result = feature_relationships(None, _record(cylinders=[_surface('one')]))
        self.assertEqual(result['comparison_coverage']['compared_pair_count'], 0)
        self.assertEqual(result['comparison_coverage']['uncompared_eligible_candidate_count'], 1)
        self.assertEqual(result['classification_candidate'], 'revolved_part_review')

    def test_tiny_geometry_uses_absolute_length_tolerance_not_length_as_angle(self):
        inner = _surface('inner', radius=.0002, interval=(-.001, .001), orientation='inward')
        outer = _surface('outer', radius=.0003, interval=(-.001, .001))
        result = feature_relationships(None, _record(cylinders=[inner, outer]), tolerance=1e-8)
        gap, = result['geometric_radial_gap_candidates']
        self.assertAlmostEqual(gap['radial_gap_mm'], .0001, places=12)
        self.assertAlmostEqual(gap['axial_overlap_length_mm'], .002, places=12)
        shifted = _surface('shifted', radius=.0003, origin=(2e-8, 0., 0.))
        self.assertFalse(feature_relationships(None, _record(cylinders=[inner, shifted]), tolerance=1e-8)['coaxial_surface_groups'])
        planes = [_plane('a', point=(0., 0., 0.)), _plane('b', normal=(0., 0., -1.), point=(0., 0., 1e-9))]
        relation, = feature_relationships(None, _record(planes=planes), tolerance=1e-12)['plane_relationships']
        self.assertAlmostEqual(relation['distance_mm'], 1e-9, delta=1e-20)
        self.check_result(result)

    def test_shared_pair_budget_is_bounded_reports_omissions_and_is_deterministic(self):
        for plane_count, axis_count in ((2000, 0), (0, 2000), (1000, 1000)):
            with self.subTest(plane_count=plane_count, axis_count=axis_count):
                record = _record(planes=[_plane(f'p_{i:05d}', point=(0., 0., float(i))) for i in range(plane_count)],
                                 cylinders=[_surface(f'c_{i:05d}') for i in range(axis_count)])
                result = feature_relationships(None, record)
                coverage = result['comparison_coverage']
                self.assertEqual(coverage['compared_pair_count'], 512)
                expected = plane_count * (plane_count - 1) // 2 + axis_count * (axis_count - 1) // 2
                self.assertEqual(coverage['eligible_pair_count'], expected)
                self.assertEqual(coverage['omitted_pair_count'], expected - 512)
                self.assertFalse(coverage['complete'])
                self.assertLess(coverage['pair_coverage_fraction'], 1.)
                self.assertLessEqual(len(result['plane_relationships']) + len(result['coaxial_surface_groups']), 512)
                if axis_count:
                    self.assertEqual(result['classification_candidate'], 'mixed_mechanical_review')
                if plane_count and axis_count:
                    self.assertEqual(coverage['by_kind']['plane_pairs']['compared_pair_count'], 256)
                    self.assertEqual(coverage['by_kind']['axial_surface_pairs']['compared_pair_count'], 256)
                self.assertEqual(feature_relationships(None, record), result)
                self.check_result(result)

    def test_pair_budget_exact_boundary_and_no_repeated_geometry_comparison(self):
        # C(32, 2)=496; C(7, 2)=21 -> 517 available pairs, five omitted.
        record = _record(planes=[_plane(f'p{i}') for i in range(32)],
                         cylinders=[_surface(f'c{i}') for i in range(7)])
        with patch('feature_relations._angle', wraps=feature_relations._angle) as angle:
            result = feature_relationships(None, record)
        self.assertEqual(angle.call_count, 512)
        coverage = result['comparison_coverage']
        self.assertEqual(coverage['omitted_pair_count'], 5)
        self.assertEqual(coverage['by_kind']['axial_surface_pairs']['compared_pair_count'], 21)
        self.assertEqual(coverage['by_kind']['plane_pairs']['compared_pair_count'], 491)

    def test_large_finite_coordinates_do_not_return_infinite_derived_measurements(self):
        inner = _surface('inner', radius=3., orientation='inward', interval=(-1e308, 1e308))
        outer = _surface('outer', interval=(-1e308, 1e308))
        with self.assertRaises(ValueError):
            feature_relationships(None, _record(cylinders=[inner, outer]))

    def test_invalid_tolerance_rejected_before_empty_shortcut(self):
        for tolerance in (-.01, 0., float('nan'), float('inf'), -float('inf'), None, True, '0.05', complex(.05, 0)):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    feature_relationships(None, {}, tolerance=tolerance)

    def test_nonfinite_values_rejected_including_uncompared_candidates_and_mesh(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            for path in ('axis', 'area', 'frame', 'radius', 'fit', 'body', 'sphere'):
                with self.subTest(value=value, path=path):
                    record = _record(cylinders=[_surface('one')])
                    if path == 'axis':
                        record['cylinders'][0]['axis'][0] = value
                    elif path == 'area':
                        record['cylinders'][0]['area_mm2'] = value
                    elif path == 'frame':
                        record['cylinders'][0]['frame']['origin_mm'][0] = value
                    elif path == 'radius':
                        record['cylinders'][0]['radius_mm'] = value
                    elif path == 'fit':
                        record['cylinders'][0]['max_vertex_residual_mm'] = value
                    elif path == 'body':
                        record['source_origin_mm'][0] = value
                    else:
                        record['spheres'] = [{'candidate_id': 'sphere', 'radius_mm': value}]
                    with self.assertRaises(ValueError):
                        feature_relationships(None, record)
                    with self.assertRaises(ValueError):
                        section_directions(record)
            mesh = trimesh.creation.box()
            mesh.vertices[0, 0] = value
            with self.assertRaises(ValueError):
                feature_relationships(mesh, {})
        record = _record(cylinders=[_surface(f'c{i}') for i in range(600)])
        record['cylinders'][-1]['radius_mm'] = float('nan')
        with self.assertRaises(ValueError):
            feature_relationships(None, record)

    def test_malformed_directions_frames_and_parameters_are_not_silently_used(self):
        record = _record(cylinders=[_surface('one')])
        changes = [('axis', [0., 0., 0.]), ('axis', [0., 1.]), ('radius_mm', -1.),
                   ('radius_mm', 0.), ('area_mm2', -1.), ('axial_parameter_range_mm', [2., 1.]),
                   ('angular_coverage_degrees', 361.), ('wraps_full_circumference', 1),
                   ('angular_parameter_intervals_radians', [[1., 0.]]),
                   ('angular_parameter_intervals_radians', [[0., 7.]]), ('normal_orientation', 'certified')]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                invalid = copy.deepcopy(record)
                invalid['cylinders'][0][key] = value
                with self.assertRaises(ValueError):
                    feature_relationships(None, invalid)
                with self.assertRaises(ValueError):
                    section_directions(invalid)
        record['local_to_source_rotation'][0][0] = 2.
        with self.assertRaises(ValueError):
            section_directions(record)
        candidate = _surface('one')
        candidate['frame']['local_to_source_transform'][0][3] = 1.
        with self.assertRaises(ValueError):
            feature_relationships(None, _record(cylinders=[candidate]))


class SectionDirectionTests(unittest.TestCase):
    def check_directions(self, values):
        self.assertLessEqual(len(values), 3)
        self.assertEqual(len({item['id'] for item in values}), len(values))
        for item in values:
            self.assertIsInstance(item['axis_source'], str)
            self.assertTrue(item['reason'])
            self.assertIsInstance(item['is_plate_direction'], bool)
            self.assertTrue(item['candidate_geometry_only'])
            axis = np.asarray(item['axis'])
            self.assertAlmostEqual(np.linalg.norm(axis), 1.)
            frame = item['frame']
            rotation = np.asarray(frame['local_to_source_rotation'])
            transform = np.asarray(frame['local_to_source_transform'])
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)
            np.testing.assert_allclose(rotation[:, 2], axis, atol=1e-10)
            np.testing.assert_allclose(transform[:3, 3], frame['origin_mm'], atol=1e-10)
            np.testing.assert_allclose(transform[:3, :3], rotation, atol=1e-10)
        for i, item in enumerate(values):
            for other in values[i + 1:]:
                self.assertLess(abs(np.dot(item['axis'], other['axis'])), 1 - 1e-8)
        if len(values) == 3:
            self.assertGreater(abs(np.linalg.det(np.asarray([item['axis'] for item in values]))), 1e-4)
        json.dumps(values, allow_nan=False)

    def test_thin_plate_column_two_first_even_if_curved_axis_has_huge_area(self):
        rotation = trimesh.transformations.rotation_matrix(1.09, [1., 2., -4.])[:3, :3]
        origin = np.array([17., -29., 83.])
        record = _record(cylinders=[_surface('large_curved', axis=rotation[:, 0], area=1e10)],
                         rotation=rotation, origin=origin, extents=(123., 37., 1.7))
        before = copy.deepcopy(record)
        directions = section_directions(record)
        self.check_directions(directions)
        self.assertEqual([item['axis_source'] for item in directions],
                         ['local_frame_column_2', 'local_frame_column_0', 'local_frame_column_1'])
        self.assertEqual([item['is_plate_direction'] for item in directions], [True, False, False])
        for item, column in zip(directions, (2, 0, 1)):
            self.assertAlmostEqual(abs(np.dot(item['axis'], rotation[:, column])), 1.)
            np.testing.assert_array_equal(item['frame']['origin_mm'], origin)
        self.assertEqual(record, before)

    def test_real_rotated_translated_plate_record_selects_known_thinnest_axis(self):
        mesh = trimesh.creation.box(extents=[123., 37., 1.7])
        transform = trimesh.transformations.rotation_matrix(1.09, [1., 2., -4.])
        mesh.apply_transform(transform)
        mesh.apply_translation([17., -29., 83.])
        record = _mesh_record(mesh)
        first = section_directions(record)[0]
        self.assertTrue(first['is_plate_direction'])
        self.assertAlmostEqual(abs(np.dot(first['axis'], transform[:3, 2])), 1.)
        self.assertEqual(first['axis_source'], 'local_frame_column_2')

    def test_nonplate_dominant_curved_axis_uses_world_axis_not_body_axis(self):
        rotation = trimesh.transformations.rotation_matrix(.97, [3., -1., 2.])[:3, :3]
        translation = [1e6, -2e6, 3e6]
        cone = _surface('cone', kind='cone', axis=-rotation[:, 1], origin=translation, area=500.)
        record = _record(cones=[cone], rotation=rotation, origin=translation)
        directions = section_directions(record)
        self.check_directions(directions)
        first = directions[0]
        self.assertEqual(first['axis_source'], 'cone_axis:cone')
        self.assertFalse(first['is_plate_direction'])
        self.assertAlmostEqual(abs(np.dot(first['axis'], rotation[:, 1])), 1.)
        self.assertEqual([item['axis_source'] for item in directions[1:]], ['local_frame_column_0', 'local_frame_column_2'])
        np.testing.assert_array_equal(first['frame']['origin_mm'], translation)

    def test_real_cone_partial_and_full_axis_precedes_local_longest_axis(self):
        transform = trimesh.transformations.rotation_matrix(1.19, [2., -3., 4.])
        for coverage in (90., 180., 360.):
            with self.subTest(coverage=coverage):
                mesh = _profile(coverage=coverage)
                mesh.apply_transform(transform)
                mesh.apply_translation([31., -27., 13.])
                record = _mesh_record(mesh, extra=True)
                self.assertEqual(len(record['cones']), 1)
                directions = section_directions(record)
                self.check_directions(directions)
                self.assertTrue(directions[0]['axis_source'].startswith('cone_axis:'))
                self.assertAlmostEqual(abs(np.dot(directions[0]['axis'], transform[:3, 2])), 1.)

    def test_area_times_coverage_ranking_duplicate_and_opposite_axis_dedup(self):
        candidates = [_surface('partial_large', axis=(1., 0., 0.), coverage=90., area=1000.),
                      _surface('full_dominant', axis=(0., 0., 1.), area=400.),
                      _surface('opposite_duplicate', axis=(0., 0., -1.), area=390.)]
        cone = _surface('second', kind='cone', axis=(0., 1., 0.), area=300.)
        record = _record(cylinders=candidates, cones=[cone])
        result = section_directions(record)
        self.assertEqual([item['axis_source'] for item in result],
                         ['cylinder_axis:full_dominant', 'cone_axis:second', 'cylinder_axis:partial_large'])
        self.check_directions(result)
        shuffled = copy.deepcopy(record)
        shuffled['cylinders'] = list(reversed(shuffled['cylinders']))
        self.assertEqual([item['axis_source'] for item in result],
                         [item['axis_source'] for item in section_directions(shuffled)])
        self.assertEqual(section_directions(record), result)

    def test_nearly_parallel_frame_seed_still_defines_an_orthonormal_section_frame(self):
        axis = (math.cos(2e-8), math.sin(2e-8), 0.)
        record = _record(cylinders=[_surface('near_local_x', axis=axis)])
        self.check_directions(section_directions(record))

    def test_tied_candidates_have_deterministic_id_order(self):
        record = _record(cylinders=[_surface('b', axis=(0., 1., 0.)), _surface('a', axis=(1., 0., 0.))])
        result = section_directions(record)
        self.assertEqual([item['axis_source'] for item in result],
                         ['cylinder_axis:a', 'cylinder_axis:b', 'local_frame_column_2'])
        self.check_directions(result)

    def test_coplanar_curved_axes_cannot_consume_all_three_directions(self):
        candidates = [_surface('a', axis=(1., 0., 0.), area=300.),
                      _surface('b', axis=(0., 1., 0.), area=200.),
                      _surface('c', axis=(1., 1., 0.), area=100.)]
        record = _record(cylinders=candidates)
        result = section_directions(record)
        self.assertEqual([item['axis_source'] for item in result],
                         ['cylinder_axis:a', 'cylinder_axis:b', 'local_frame_column_2'])
        self.check_directions(result)

    def test_absdot_near_duplicate_threshold_uses_orthogonal_local_fallback(self):
        angle = math.acos(1. - .5e-8)
        candidates = [_surface('a', axis=(1., 0., 0.), area=200.),
                      _surface('near', axis=(-math.cos(angle), math.sin(angle), 0.), area=100.)]
        result = section_directions(_record(cylinders=candidates))
        self.assertEqual([item['axis_source'] for item in result],
                         ['cylinder_axis:a', 'local_frame_column_1', 'local_frame_column_2'])
        self.check_directions(result)

    def test_nonplate_without_curved_axes_prefers_column_zero_and_supports_frame_dict(self):
        record = _record()
        frame = _frame([12., -14., 9.], rotation=trimesh.transformations.rotation_matrix(.87, [1, 2, 3])[:3, :3])
        del record['local_to_source_rotation'], record['source_origin_mm']
        record['frame'] = frame
        result = section_directions(record)
        self.assertEqual([item['axis_source'] for item in result],
                         ['local_frame_column_0', 'local_frame_column_1', 'local_frame_column_2'])
        self.assertTrue(all(not item['is_plate_direction'] for item in result))
        self.check_directions(result)

    def test_candidate_frame_only_axis_fallback_keeps_full_source_frame(self):
        record = _record(cones=[_surface('cone', kind='cone', origin=(17., 29., 83.), axis=(1., 2., 3.))])
        candidate = record['cones'][0]
        expected = np.asarray(candidate['frame']['z_direction'])
        del candidate['axis'], candidate['axis_point_mm']
        del record['local_to_source_rotation'], record['source_origin_mm']
        result = section_directions(record)
        self.assertEqual(len(result), 1)
        self.check_directions(result)
        self.assertAlmostEqual(abs(np.dot(result[0]['axis'], expected)), 1.)
        np.testing.assert_array_equal(result[0]['frame']['origin_mm'], [17., 29., 83.])

    def test_plate_inference_from_extents_only_and_tiny_valid_frame(self):
        record = _record(extents=(.001, .0005, .00001))
        del record['thinness_ratio']
        result = section_directions(record)
        self.assertTrue(result[0]['is_plate_direction'])
        self.assertEqual(result[0]['axis_source'], 'local_frame_column_2')
        self.check_directions(result)

    def test_empty_limits_and_nonfinite_or_invalid_limit_rejection(self):
        self.assertEqual(section_directions({}), [])
        record = _record()
        for limit in (0, 1, 2, 3, 20):
            with self.subTest(limit=limit):
                self.assertEqual(len(section_directions(record, limit)), min(limit, 3))
        for limit in (-1, .5, 3., '3', None, True, float('nan'), float('inf')):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    section_directions(record, limit)
        record['source_origin_mm'][0] = float('nan')
        with self.assertRaises(ValueError):
            section_directions(record, 0)


if __name__ == '__main__':
    unittest.main()
