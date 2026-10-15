"""Synthetic finite-offset event evidence tests; no CAD execution/acceptance.

Fixtures deliberately keep original source indices and do not use mesh repair,
Boolean operations or a particular production 3MF.
"""
from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from mesh_evidence import map_source_indices
import section_analysis as sections
from section_analysis import SectionTopologyError, analyze_sections, section_at


def layered_prism(levels, scales=None, profile=None):
    """Convex polygon rings, including optional ledges at repeated heights."""
    profile = np.asarray(profile if profile is not None else
                         [[-7, -5], [7, -5], [7, 5], [-7, 5]], dtype=float)
    scales = scales if scales is not None else [1.] * len(levels)
    count = len(profile)
    vertices = np.vstack([np.column_stack((profile * scale, np.full(count, z)))
                          for z, scale in zip(levels, scales)])
    faces = [[0, i + 1, i] for i in range(1, count - 1)]
    for layer in range(len(levels) - 1):
        lower, upper = layer * count, (layer + 1) * count
        for i in range(count):
            following = (i + 1) % count
            faces.extend([[lower + i, lower + following, upper + following],
                          [lower + i, upper + following, upper + i]])
    top = (len(levels) - 1) * count
    faces.extend([[top, top + i, top + i + 1] for i in range(1, count - 1)])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def exact_failure(mesh, z, attempt_index=0):
    try:
        section_at(mesh, z)
    except SectionTopologyError as exc:
        return {"attempt_index": attempt_index, "z_mm": z, "stage": "section",
                "error_code": exc.code, "evidence": exc.metadata}
    raise AssertionError("Fixture must fail at the exact event")


def injected_ambiguity(z):
    return SectionTopologyError("ambiguous_section_event", "Injected probe ambiguity, not repaired.",
                                z_mm=z, raw_paths=[[[1., 2.], [3., 4.]]],
                                details={"synthetic_retry_test": True})


class SectionEventProbeTests(unittest.TestCase):
    def assert_json(self, stack):
        json.dumps(stack.metadata, allow_nan=False)
        for sample in stack.samples:
            json.dumps(sample.metadata, allow_nan=False)

    def assert_probe_bounds(self, mesh, stack):
        vertices = mesh.vertices[np.unique(mesh.faces)]
        zs = np.unique(vertices[:, 2])
        policy = stack.metadata['event_probe_policy']
        count = 0
        for event in stack.metadata['event_probes']:
            parent = stack.metadata['failed_samples'][event['parent_failed_sample_index']]
            self.assertEqual(parent['error_code'], 'ambiguous_section_event')
            self.assertEqual(event['parent_attempt_index'], parent['attempt_index'])
            self.assertEqual(event['original_z_mm'], parent['z_mm'])
            self.assertEqual(event['parent_evidence_json_pointer'],
                             f"/failed_samples/{event['parent_failed_sample_index']}/evidence")
            self.assertEqual([s['side'] for s in event['sides']], ['below', 'above'])
            for side in event['sides']:
                self.assertIn(side['status'], ['success', 'failed', 'unavailable'])
                self.assertTrue(side['stop_reason'])
                self.assertEqual(side['original_z_mm'], parent['z_mm'])
                self.assertLessEqual(len(side['attempts']), policy['max_attempts_per_side'])
                for attempt in side['attempts']:
                    count += 1
                    actual, original = attempt['actual_z_mm'], attempt['original_z_mm']
                    delta = actual - original
                    self.assertTrue(math.isfinite(actual))
                    self.assertEqual(attempt['delta_mm'], delta)
                    self.assertGreater(abs(delta), 0.)
                    self.assertLessEqual(abs(delta), side['max_abs_delta_mm'])
                    self.assertLessEqual(abs(delta), float(zs[-1] - zs[0]) * .001)
                    self.assertEqual(delta < 0, side['side'] == 'below')
                    lo, hi = side['open_interval_z_mm']
                    self.assertLess(lo, actual)
                    self.assertLess(actual, hi)
                    self.assertFalse(np.any((zs > min(original, actual)) & (zs < max(original, actual))))
                    self.assertGreater(attempt['nearest_vertex_z_clearance_mm'],
                                       side['numeric_on_plane_epsilon_mm'])
                    self.assertFalse(attempt['included_in_primary_sampling'])
                    if attempt.get('evidence'):
                        self.assertEqual(attempt['evidence']['z_mm'], actual)
                if not side['attempts']:
                    self.assertIsNone(side['actual_z_mm'])
                    self.assertIsNone(side['delta_mm'])
                else:
                    self.assertEqual(side['actual_z_mm'], side['attempts'][-1]['actual_z_mm'])
                    self.assertEqual(side['delta_mm'], side['attempts'][-1]['delta_mm'])
        self.assertEqual(policy['attempted_section_count'], count)
        self.assertLessEqual(count, policy['additional_section_attempt_budget'])
        self.assertFalse(stack.metadata['sampling_complete'])
        self.assertFalse(stack.metadata['automatic_full_thickness_suggestion_allowed'])
        self.assert_json(stack)

    def test_exact_failure_and_all_primary_outputs_are_unchanged(self):
        for event_z in (.25, .5):
            with self.subTest(event_z=event_z):
                mesh = layered_prism([0, event_z, event_z, 1], [1, 1, 1.2, 1.2])
                original = exact_failure(mesh, event_z)['evidence']
                with patch('section_analysis._probe_section_events', return_value=([], {})):
                    baseline = analyze_sections(mesh)
                stack = analyze_sections(mesh)
                for key in baseline.metadata:
                    if key not in {'event_probes', 'event_probe_policy', 'warnings'}:
                        self.assertEqual(stack.metadata[key], baseline.metadata[key], key)
                self.assertEqual([s.metadata for s in stack.samples],
                                 [s.metadata for s in baseline.samples])
                self.assertEqual(stack.metadata['failed_samples'][0]['evidence'], original)
                self.assertEqual(len(stack.metadata['event_probes']), 1)
                event = stack.metadata['event_probes'][0]
                self.assertEqual([s['status'] for s in event['sides']], ['success', 'success'])
                areas = [s['attempts'][0]['evidence']['net_area_mm2'] for s in event['sides']]
                self.assertAlmostEqual(areas[0], 140.)
                self.assertAlmostEqual(areas[1], 201.6)
                for side in event['sides']:
                    evidence = side['attempts'][0]['evidence']
                    self.assertIsNone(evidence['comparison_to_reference'])
                    self.assertEqual(evidence['sampling']['parent_attempt_index'], event['parent_attempt_index'])
                if event_z == .25:
                    self.assertIsNone(stack.metadata['reference_sample_index'])
                    self.assertIsNone(stack.metadata['profile_varies'])
                else:
                    self.assertTrue(stack.metadata['profile_varies'])
                self.assert_probe_bounds(mesh, stack)

    def test_successful_probes_cannot_clear_plate_reconstruction_veto(self):
        from reverse_assistant import make_plate_plan
        mesh = layered_prism([0, .25, 1])
        stack = analyze_sections(mesh)
        self.assertTrue(all(side['status'] == 'success'
                            for side in stack.metadata['event_probes'][0]['sides']))
        with self.assertRaisesRegex(ValueError, 'Section evidence is incomplete/invalid'):
            make_plate_plan(mesh, mesh, np.eye(3), np.zeros(3), stack, 'synthetic.3mf', '0' * 64)
        self.assertIsNone(stack.metadata['reference_sample_index'])
        self.assertEqual(stack.metadata['profile_status'], 'indeterminate')
        self.assertTrue(all(s.metadata['comparison_to_reference'] is None for s in stack.samples))
        failed = stack.metadata['failed_samples'][0]
        attempt = stack.metadata['sampling_schedule'][failed['attempt_index']]
        self.assertEqual(attempt['status'], 'failed')
        self.assertIsNone(attempt['sample_index'])

    def test_rotated_source_recovered_to_mesh_local_frame_preserves_lineage(self):
        profile = [[-7, -4], [4, -5], [8, 1], [2, 6], [-6, 3]]
        original = layered_prism([0, .5, 1], profile=profile)
        # A nontrivial tilted frame with exactly recoverable local Z; no frame
        # assumption in the sectioner or hidden mesh.section recentering.
        rotation = np.array([[.6, -.8, 0.], [0., 0., -1.], [.8, .6, 0.]])
        origin = np.array([71., -23., 14.])
        source_vertices = original.vertices @ rotation.T + origin
        local = trimesh.Trimesh(vertices=(source_vertices - origin) @ rotation,
                                faces=original.faces.copy(), process=False)
        with patch.object(local, 'section', side_effect=AssertionError('No Path3D backend')):
            stack = analyze_sections(local)
        self.assertEqual(len(stack.metadata['event_probes']), 1)
        for side in stack.metadata['event_probes'][0]['sides']:
            self.assertEqual(side['status'], 'success')
            evidence = side['attempts'][0]['evidence']
            self.assertEqual(evidence['projection'], 'local_xy_without_transform')
            for node in evidence['intersection_nodes']:
                self.assertEqual(node['point_xyz_mm'][2], side['actual_z_mm'])
                key = node['source_key']
                self.assertEqual(key[0], 'edge')
                a, b = source_vertices[key[1:]]
                point = rotation @ np.asarray(node['point_xyz_mm']) + origin
                fraction = np.dot(point - a, b - a) / np.dot(b - a, b - a)
                np.testing.assert_allclose(point, a + fraction * (b - a), atol=1e-11)
        self.assert_probe_bounds(local, stack)

    def test_arbitrary_rotation_roundoff_is_not_permission_to_cross_raw_events(self):
        original = layered_prism([0, .5, 1])
        rotation = trimesh.transformations.rotation_matrix(.713, [1, 2, 3])[:3, :3]
        origin = np.array([71.23, -23.17, 14.31])
        transformed = original.vertices @ rotation.T + origin
        mesh = trimesh.Trimesh(vertices=(transformed - origin) @ rotation,
                               faces=original.faces.copy(), process=False)
        stack = analyze_sections(mesh)
        self.assertTrue(stack.metadata['event_probes'])
        self.assertTrue(any(side['status'] == 'unavailable' for event in stack.metadata['event_probes']
                            for side in event['sides']))
        self.assert_probe_bounds(mesh, stack)

    def test_large_displacement_uses_numeric_epsilon_but_never_large_offsets(self):
        for translation, expected in ((1e9, 'success'), (1e12, 'unavailable')):
            with self.subTest(translation=translation):
                mesh = layered_prism([0, .5, 1])
                mesh.apply_translation([translation, -translation, translation])
                stack = analyze_sections(mesh)
                self.assertEqual(len(stack.metadata['event_probes']), 1)
                for side in stack.metadata['event_probes'][0]['sides']:
                    self.assertEqual(side['status'], expected)
                    if expected == 'success':
                        self.assertGreater(abs(side['delta_mm']), side['numeric_on_plane_epsilon_mm'])
                    else:
                        self.assertEqual(side['stop_reason'], 'no_numeric_clearance_before_adjacent_event')
                        self.assertEqual(side['attempts'], [])
                self.assert_probe_bounds(mesh, stack)

    def test_adjacent_events_inside_schedule_cluster_still_bound_each_probe(self):
        gap = 2e-10  # Below schedule coalescing epsilon, above source arithmetic epsilon.
        mesh = layered_prism([0, .5 - gap, .5, .5 + gap, 1])
        stack = analyze_sections(mesh)
        self.assertEqual(stack.metadata['sampling_policy']['event_layer_count'], 3)
        self.assertEqual(stack.metadata['sampling_policy']['raw_unique_vertex_z_count'], 5)
        event = stack.metadata['event_probes'][0]
        self.assertEqual([s['status'] for s in event['sides']], ['success', 'success'])
        self.assertEqual([s['adjacent_event_z_mm'] for s in event['sides']], [.5 - gap, .5 + gap])
        self.assert_probe_bounds(mesh, stack)

    def test_sub_epsilon_gaps_are_explicitly_unavailable_independent_of_user_tolerance(self):
        mesh = layered_prism([0, .5 - 1e-14, .5, .5 + 1e-14, 1])
        with patch('section_analysis.section_at', wraps=section_at) as extract:
            stack = analyze_sections(mesh, tolerance=0)
            self.assertEqual(extract.call_count, len(stack.metadata['sampling_schedule']))
        relaxed = analyze_sections(mesh, tolerance=100)
        self.assertEqual(stack.metadata['event_probes'], relaxed.metadata['event_probes'])
        for side in stack.metadata['event_probes'][0]['sides']:
            self.assertEqual(side['status'], 'unavailable')
            self.assertEqual(side['stop_reason'], 'no_numeric_clearance_before_adjacent_event')
        self.assert_probe_bounds(mesh, stack)

    def test_endpoint_event_never_probes_outside_mesh_and_interior_side_is_explicit(self):
        mesh = layered_prism([0, 1])
        failures = [exact_failure(mesh, z, i) for i, z in enumerate((0., 1.))]
        events, policy = sections._probe_section_events(mesh, mesh.vertices, failures, .05, 7)
        for event, outside in zip(events, ('below', 'above')):
            self.assertEqual(len(event['sides']), 2)
            for side in event['sides']:
                if side['side'] == outside:
                    self.assertEqual(side['status'], 'unavailable')
                    self.assertEqual(side['stop_reason'], 'outside_mesh_z_bounds')
                    self.assertIsNone(side['adjacent_event_z_mm'])
                    self.assertIsNone(side['actual_z_mm'])
                    self.assertEqual(side['attempts'], [])
                else:
                    self.assertEqual(side['status'], 'success')
                    self.assertLess(0., side['actual_z_mm'])
                    self.assertLess(side['actual_z_mm'], 1.)
        self.assertEqual(policy['attempted_section_count'], 2)
        self.assertEqual(policy['unavailable_side_count'], 2)
        json.dumps(events, allow_nan=False)

    def test_fraction_rounding_to_endpoint_is_not_silently_moved(self):
        mesh = layered_prism([1e9, 1e9 + 1e-6])
        with patch('section_analysis.section_at', wraps=section_at) as extract:
            stack = analyze_sections(mesh)
        self.assertEqual(extract.call_count, len(stack.metadata['sampling_schedule']))
        bounds = stack.metadata['z_bounds_mm']
        self.assertTrue(any(a['z_mm'] in bounds for a in stack.metadata['sampling_schedule']))
        self.assertEqual(stack.samples, [])
        self.assertTrue(all(a['status'] == 'failed' for a in stack.metadata['sampling_schedule']))
        self.assertTrue(all(s['status'] == 'unavailable' for e in stack.metadata['event_probes'] for s in e['sides']))
        self.assert_probe_bounds(mesh, stack)

    def test_non_event_primary_failures_never_trigger_extra_extractions_or_repair(self):
        mesh = layered_prism([0, 1])
        mesh.update_faces(np.arange(len(mesh.faces)) != 2)
        vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
        with patch('section_analysis.section_at', wraps=section_at) as extract, \
                patch.object(trimesh.Trimesh, 'fill_holes', side_effect=AssertionError('No repair')):
            stack = analyze_sections(mesh)
        self.assertEqual(extract.call_count, len(stack.metadata['sampling_schedule']))
        self.assertEqual(stack.metadata['event_probes'], [])
        self.assertEqual(stack.metadata['event_probe_policy']['attempted_section_count'], 0)
        self.assertEqual({f['error_code'] for f in stack.metadata['failed_samples']}, {'open_or_branched_section'})
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)
        self.assert_probe_bounds(mesh, stack)

    def test_open_one_sided_probe_failure_is_retained_and_never_retried(self):
        mesh = layered_prism([0, .5, 1])
        mesh.update_faces(np.arange(len(mesh.faces)) != 2)
        vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
        stack = analyze_sections(mesh)
        self.assertEqual(len(stack.metadata['event_probes']), 1)
        below, above = stack.metadata['event_probes'][0]['sides']
        self.assertEqual(below['status'], 'failed')
        self.assertEqual(below['stop_reason'], 'non_event_probe_failure_not_retried')
        self.assertEqual(len(below['attempts']), 1)
        failure = below['attempts'][0]
        self.assertEqual(failure['error_code'], 'open_or_branched_section')
        self.assertTrue(failure['evidence']['details']['bad_degree_nodes'])
        self.assertEqual(failure['evidence']['traversed_segment_count'], 0)
        self.assertEqual(above['status'], 'success')
        self.assertEqual(stack.metadata['event_probe_policy']['attempted_section_count'], 2)
        self.assertTrue(all(f['z_mm'] != below['actual_z_mm'] for f in stack.metadata['failed_samples']))
        # Failed probes retain the same exportable source provenance as exact
        # failures; generated degree-node/segment IDs must stay untouched.
        original_failure = deepcopy(failure)
        face_map, vertex_map = np.arange(len(faces)) + 100, np.arange(len(vertices)) + 60
        mapped = map_source_indices(failure, face_map, vertex_map, copy_data=True)
        self.assertEqual(failure, original_failure)
        self.assertEqual(mapped['source_triangle_id_scope'], 'build_instance_faces')
        self.assertEqual(mapped['evidence']['source_vertex_id_scope'], 'build_instance_vertices')
        for old, new in zip(failure['evidence']['details']['bad_degree_nodes'],
                            mapped['evidence']['details']['bad_degree_nodes']):
            self.assertEqual(old['node_id'], new['node_id'])
            self.assertEqual(old['incident_segment_ids'], new['incident_segment_ids'])
            self.assertEqual(new['source_key'], [old['source_key'][0], *vertex_map[old['source_key'][1:]].tolist()])
        for old, new in zip(failure['evidence']['segments'], mapped['evidence']['segments']):
            self.assertEqual(new['source_triangle_id'], int(face_map[old['source_triangle_id']]))
            self.assertEqual(old['endpoint_ids'], new['endpoint_ids'])
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)
        self.assert_probe_bounds(mesh, stack)

    def test_tangent_vertex_probe_does_not_discard_the_open_component_above(self):
        body = layered_prism([-1, 1])
        tangent = trimesh.Trimesh(vertices=[[15, 0, 0], [16, 0, 1], [15, 1, 1]],
                                  faces=[[0, 1, 2]], process=False)
        mesh = trimesh.util.concatenate([body, tangent])
        stack = analyze_sections(mesh)
        event = stack.metadata['event_probes'][0]
        exact = stack.metadata['failed_samples'][event['parent_failed_sample_index']]['evidence']
        self.assertEqual(exact['ambiguity_kinds'], ['vertex_on_plane_event'])
        self.assertEqual(exact['on_plane_vertex_ids'], [len(body.vertices)])
        self.assertEqual(exact['event_triangle_ids'], [len(body.faces)])
        below, above = event['sides']
        self.assertEqual(below['status'], 'success')
        self.assertEqual(below['attempts'][0]['evidence']['loop_count'], 1)
        self.assertEqual(above['status'], 'failed')
        self.assertEqual(above['attempts'][0]['error_code'], 'open_or_branched_section')
        self.assertGreater(above['attempts'][0]['evidence']['segment_count'],
                           below['attempts'][0]['evidence']['segment_count'])
        self.assertEqual(len(above['attempts']), 1)
        self.assert_probe_bounds(mesh, stack)

    def test_event_retries_are_exponential_bounded_and_keep_prior_failures(self):
        mesh = layered_prism([0, .5, 1])
        primary = set(sections.DEFAULT_FRACTIONS)
        def extract(m, z, tolerance):
            if z not in primary and abs(z - .5) < 3e-6:
                raise injected_ambiguity(z)
            return section_at(m, z, tolerance)
        with patch('section_analysis.section_at', side_effect=extract):
            stack = analyze_sections(mesh, max_samples=7)
        for side in stack.metadata['event_probes'][0]['sides']:
            self.assertEqual(side['status'], 'success')
            self.assertEqual([a['status'] for a in side['attempts']], ['failed', 'failed', 'success'])
            self.assertEqual([a['requested_abs_delta_mm'] for a in side['attempts']], [1e-6, 2e-6, 4e-6])
            self.assertEqual(side['selected_probe_attempt_index'], side['attempts'][-1]['probe_attempt_index'])
            for attempt in side['attempts'][:-1]:
                self.assertTrue(attempt['evidence']['details']['synthetic_retry_test'])
                self.assertEqual(attempt['evidence']['raw_paths_mm'], [[[1., 2.], [3., 4.]]])
        self.assertEqual(len(stack.metadata['failed_samples']), 1)
        self.assertEqual(stack.metadata['event_probe_policy']['retry_section_count'], 4)
        self.assert_probe_bounds(mesh, stack)

    def test_exponential_retry_ceiling_does_not_run_until_success(self):
        mesh = layered_prism([0, .5, 1])
        def extract(m, z, tolerance):
            if z not in sections.DEFAULT_FRACTIONS:
                raise injected_ambiguity(z)
            return section_at(m, z, tolerance)
        with patch('section_analysis.section_at', side_effect=extract) as calls:
            stack = analyze_sections(mesh, max_samples=7)
        self.assertEqual(calls.call_count, 15)  # Seven primary + four for each side.
        for side in stack.metadata['event_probes'][0]['sides']:
            self.assertEqual(side['status'], 'failed')
            self.assertEqual(side['stop_reason'], 'per_side_attempt_limit_reached')
            self.assertEqual([a['requested_abs_delta_mm'] for a in side['attempts']],
                             [1e-6, 2e-6, 4e-6, 8e-6])
        self.assert_probe_bounds(mesh, stack)

    def test_offset_ceiling_stops_retries_before_repeating_a_slice(self):
        mesh = layered_prism([0, .5 - 1e-6, .5, .5 + 1e-6, 1])
        def extract(m, z, tolerance):
            if z not in sections.DEFAULT_FRACTIONS:
                raise injected_ambiguity(z)
            return section_at(m, z, tolerance)
        with patch('section_analysis.section_at', side_effect=extract):
            stack = analyze_sections(mesh, max_samples=7)
        for side in stack.metadata['event_probes'][0]['sides']:
            self.assertEqual(side['status'], 'failed')
            self.assertEqual(side['stop_reason'], 'offset_bound_reached')
            self.assertEqual(len(side['attempts']), 1)
        self.assert_probe_bounds(mesh, stack)

    def test_each_direction_budget_is_local_fair_and_does_not_consume_primary_slots(self):
        mesh = layered_prism([0, *sections.DEFAULT_FRACTIONS, 1])
        def extract(m, z, tolerance):
            if z not in sections.DEFAULT_FRACTIONS:
                raise injected_ambiguity(z)
            return section_at(m, z, tolerance)
        with patch('section_analysis.section_at', side_effect=extract) as calls:
            first = analyze_sections(mesh, max_samples=7)
            second = analyze_sections(mesh, max_samples=7)
        self.assertEqual(calls.call_count, 42)  # Independent 7 + 14 budgets.
        self.assertEqual(first.metadata, second.metadata)
        policy = first.metadata['event_probe_policy']
        self.assertEqual(policy['attempted_section_count'], 14)
        self.assertEqual(policy['maximum_total_section_attempts'], 21)
        self.assertTrue(policy['budget_limited'])
        self.assertEqual(len(first.metadata['sampling_schedule']), 7)
        self.assertEqual(len(first.metadata['failed_samples']), 7)
        for event in first.metadata['event_probes']:
            for side in event['sides']:
                self.assertEqual(side['status'], 'failed')
                self.assertEqual(side['stop_reason'], 'direction_probe_budget_exhausted')
                self.assertEqual(len(side['attempts']), 1)
        self.assert_probe_bounds(mesh, first)

    def test_source_mapping_covers_exact_event_and_probes_without_mutation_or_generated_id_mapping(self):
        base = layered_prism([0, .5, .5, 1], [1, 1, 1.2, 1.2])
        # An unused vertex must not create a fake neighboring Z or inflate the
        # numeric epsilon; face/vertex keys still address the original arrays.
        mesh = trimesh.Trimesh(vertices=np.vstack([[[1e15, -1e15, .5 + 1e-15]], base.vertices]),
                               faces=base.faces + 1, process=False)
        vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
        stack = analyze_sections(mesh)
        original = {**stack.metadata, 'samples': [s.metadata for s in stack.samples]}
        before = deepcopy(original)
        face_map = np.arange(len(mesh.faces)) * 3 + 200
        vertex_map = np.arange(len(mesh.vertices)) * 7 + 100
        mapped = map_source_indices(original, face_map, vertex_map, copy_data=True)
        self.assertEqual(original, before)
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)
        original_event, mapped_event = original['event_probes'][0], mapped['event_probes'][0]
        for a, b in [(original_event, mapped_event), *zip(original_event['sides'], mapped_event['sides'])]:
            self.assertEqual(b['source_triangle_id_scope'], 'build_instance_faces')
            self.assertEqual(b['source_vertex_id_scope'], 'build_instance_vertices')
            self.assertEqual(a['parent_attempt_index'], b['parent_attempt_index'])
        evidence_pairs = [(original['failed_samples'][0]['evidence'], mapped['failed_samples'][0]['evidence'])]
        evidence_pairs.extend(zip(original['samples'], mapped['samples']))
        for a, b in zip(original_event['sides'], mapped_event['sides']):
            self.assertEqual(a['status'], 'success')
            self.assertEqual(len(a['attempts']), 1)
            self.assertEqual(b['attempts'][0]['source_triangle_id_scope'], 'build_instance_faces')
            evidence_pairs.append((a['attempts'][0]['evidence'], b['attempts'][0]['evidence']))
        for a, b in evidence_pairs:
            self.assertEqual(a['unused_vertex_count'], 1)
            self.assertEqual(b['source_vertex_id_scope'], 'build_instance_vertices')
            self.assertEqual(a['z_mm'], b['z_mm'])
            for key in ('event_triangle_ids', 'coplanar_triangle_ids'):
                self.assertEqual(b[key], face_map[a[key]].tolist())
            self.assertEqual(b['on_plane_vertex_ids'], vertex_map[a['on_plane_vertex_ids']].tolist())
            for old, new in zip(a['intersection_nodes'], b['intersection_nodes']):
                self.assertEqual(new['source_key'], [old['source_key'][0], *vertex_map[old['source_key'][1:]].tolist()])
                self.assertEqual(old['id'], new['id'])
                self.assertEqual(old['incident_segment_ids'], new['incident_segment_ids'])
                self.assertEqual(old['point_xyz_mm'], new['point_xyz_mm'])
            for old, new in zip(a['segments'], b['segments']):
                self.assertEqual(new['source_triangle_id'], int(face_map[old['source_triangle_id']]))
                self.assertEqual(old['id'], new['id'])
                self.assertEqual(old['endpoint_ids'], new['endpoint_ids'])
            for old, new in zip(a['coplanar_edges'], b['coplanar_edges']):
                self.assertEqual(new['source_vertex_ids'], vertex_map[old['source_vertex_ids']].tolist())
                self.assertEqual(new['source_triangle_ids'], face_map[old['source_triangle_ids']].tolist())
            for old, new in zip(a.get('loops', []), b.get('loops', [])):
                self.assertEqual(new['source_triangle_ids'], face_map[old['source_triangle_ids']].tolist())
                for key in ('id', 'points_mm', 'intersection_node_ids', 'segment_ids'):
                    self.assertEqual(old[key], new[key])
        json.dumps(mapped, allow_nan=False)
        self.assert_probe_bounds(mesh, stack)

    def test_missing_source_provenance_is_not_guessed_or_repaired(self):
        class Fixture:
            vertices = np.array([[0., 0., 0.], [0., 0., .5], [0., 0., 1.]])
            def section(self, plane_origin, plane_normal):
                if plane_origin[2] == .5:
                    raise injected_ambiguity(.5)
                return None
        with patch('section_analysis.section_at', wraps=section_at) as extract:
            stack = analyze_sections(Fixture())
        self.assertEqual(extract.call_count, len(stack.metadata['sampling_schedule']))
        self.assertEqual(len(stack.metadata['event_probes']), 1)
        for side in stack.metadata['event_probes'][0]['sides']:
            self.assertEqual(side['status'], 'unavailable')
            self.assertEqual(side['stop_reason'], 'source_event_provenance_unavailable')
        self.assertIsNone(stack.metadata['profile_varies'])
        self.assert_json(stack)


if __name__ == '__main__':
    unittest.main()
