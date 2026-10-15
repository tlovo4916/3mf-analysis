"""Independent continuation regressions: whole-pipeline claims and retained lineage.

These are local synthetic tests, not physical CAD or exporter-population proof.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import trimesh

from assistant import process, _stack_with_source_ids
from mesh_evidence import map_source_indices, split_bodies
from section_analysis import analyze_sections
from test_threemf_io import model_xml, mesh_xml, assembly


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'arbitrary-part.3mf'

    def package(self, mesh=None, *, objects=None, items='<item objectid="1"/>', attributes=''):
        if objects is None:
            mesh = trimesh.creation.box(extents=[30, 20, 1]) if mesh is None else mesh
            objects = '<object id="1">' + mesh_xml(mesh.vertices, mesh.faces) + '</object>'
        with zipfile.ZipFile(self.source, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('3D/3dmodel.model', model_xml(objects, items, attributes=attributes))
        return self.source

    def run_part(self, **options):
        return process(self.source, self.root / 'output', **options)

    def test_candidate_cavity_must_not_clear_aggregate_mesh_blocker(self):
        outer = trimesh.creation.box(extents=[30, 20, 10])
        inner = trimesh.creation.box(extents=[15, 10, 5])
        inner.faces = inner.faces[:, ::-1]
        self.package(trimesh.util.concatenate([outer, inner]))
        report = self.run_part(max_section_directions=1)
        self.assertEqual(report['bodies'][1]['shell_role_candidate'],
                         'possible_cavity_or_inverted_nested_shell')
        self.assertIn('nonpositive_or_nonfinite_signed_volume', report['bodies'][1]['blocking_issues'])
        self.assertTrue(report['has_mesh_blockers'])
        self.assertTrue(report['has_reconstruction_blockers'])
        self.assertTrue(report['requires_shell_relationship_review'])
        self.assertFalse(report['instances'][0]['shell_relationships']['containment_certified'])
        self.assertTrue(all('plate_plan' not in body for body in report['bodies']))

    def test_failed_nonplate_sections_are_partial_not_complete(self):
        box = trimesh.creation.box(extents=[10, 8, 6])
        # Remove the two -X side triangles, so all primary slices are open.
        box.update_faces(np.flatnonzero(box.face_normals[:, 0] > -.5))
        self.package(box)
        report = self.run_part()
        self.assertTrue(any(entry['summary']['failed_sample_count']
                            for body in report['bodies'] for entry in body['section_analyses']))
        self.assertEqual(report['status'], 'analysis_partial_not_cad_complete')
        self.assertFalse(report['section_analysis_attempts_complete'])
        self.assertTrue(report['has_section_analysis_errors'])
        self.assertTrue(report['has_body_analysis_errors'])
        self.assertFalse(report['has_candidate_analysis_errors'])
        self.assertFalse(report['geometry_analysis_coverage_complete'])
        self.assertTrue(report['candidate_analysis_coverage_complete'])

    def test_nonplate_sections_have_recoverable_source_frame(self):
        mesh = trimesh.creation.cylinder(radius=5, height=25, sections=32)
        matrix = trimesh.transformations.rotation_matrix(.37, [1, 2, 3])
        matrix[:3, 3] = [75, -20, 12]
        mesh.apply_transform(matrix)
        self.package(mesh)
        report = self.run_part()
        body = report['bodies'][0]
        self.assertGreater(len(body['cylinders']), 0)
        self.assertEqual(len(body['section_analyses']), 3)
        self.assertNotIn('plate_plan', body)
        for entry in body['section_analyses']:
            self.assertIn('full_section_evidence', entry)
            with gzip.open(self.root / 'output' / entry['full_section_evidence'], 'rt') as stream:
                evidence = json.load(stream)
            rotation = np.asarray(evidence['section_frame']['local_to_source_rotation'])
            origin = np.asarray(evidence['section_frame']['origin_mm'])
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)
            for sample in evidence['samples']:
                self.assertEqual(sample['source_vertex_id_scope'], 'build_instance_vertices')
                for node in sample['intersection_nodes']:
                    key = node['source_key']
                    if key[0] != 'edge':
                        continue
                    world = rotation @ np.asarray(node['point_xyz_mm']) + origin
                    a, b = mesh.vertices[key[1:]]
                    vector = b - a
                    fraction = np.dot(world - a, vector) / np.dot(vector, vector)
                    np.testing.assert_allclose(world, a + fraction * vector, atol=1e-9)

    def test_curve_identity_and_coverage_include_sphere_and_cone(self):
        for mesh, collection in ((trimesh.creation.icosphere(subdivisions=2, radius=5), 'spheres'),
                                 (trimesh.creation.cone(radius=5, height=15, sections=32), 'cones')):
            with self.subTest(collection=collection):
                self.package(mesh)
                report = process(self.source, self.root / collection, max_section_directions=1)
                body = report['bodies'][0]
                self.assertGreater(len(body[collection]), 0)
                for candidate in body[collection]:
                    self.assertTrue(candidate['candidate_id'].startswith('instance_0000_shell_0000_'))
                    self.assertEqual(candidate['source_triangle_id_scope'], 'build_instance_faces')
                    self.assertFalse(candidate['cad_reconstructed'])
                    self.assertTrue(set(candidate['source_triangle_ids']).issubset(
                        body['candidate_coverage']['covered_source_triangle_ids']))

    def test_partial_build_cannot_offer_plate_plan(self):
        self.package(items='<item objectid="1"/><item objectid="999"/>')
        report = self.run_part(max_section_directions=1)
        self.assertEqual(len(report['package']['failed_build_items']), 1)
        self.assertFalse(report['package']['document_coverage_complete'])
        self.assertEqual(report['status'], 'analysis_partial_not_cad_complete')
        self.assertEqual(report['bodies'][0]['plate_plan_status'], 'blocked')
        self.assertNotIn('plate_plan', report['bodies'][0])

    def test_partial_nonplate_document_has_reconstruction_veto(self):
        self.package(trimesh.creation.box(extents=[10, 8, 6]),
                     items='<item objectid="1"/><item objectid="999"/>')
        report = self.run_part(max_section_directions=1)
        self.assertTrue(report['has_reconstruction_blockers'])
        self.assertFalse(report['package']['document_coverage_complete'])
        self.assertNotIn('plate_plan', report['bodies'][0])

    def test_support_can_be_analyzed_but_not_rebuilt(self):
        mesh = trimesh.creation.box(extents=[30, 20, 1])
        obj = '<object id="1" type="support">' + mesh_xml(mesh.vertices, mesh.faces) + '</object>'
        self.package(objects=obj)
        report = self.run_part(max_section_directions=1)
        self.assertEqual(report['bodies'][0]['strategy'], 'source_purpose_review')
        self.assertEqual(report['bodies'][0]['plate_plan_status'], 'blocked')
        self.assertNotIn('plate_plan', report['bodies'][0])

    def test_budget_retains_all_shells_and_entire_source_mesh(self):
        first = trimesh.creation.box(extents=[20, 15, 1])
        second = first.copy()
        second.apply_translation([50, 0, 0])
        self.package(trimesh.util.concatenate([first, second]))
        report = self.run_part(max_bodies=1, max_section_directions=1)
        self.assertEqual(report['body_count'], 2)
        self.assertEqual(report['analysis_budget']['unexamined_shell_count'], 1)
        self.assertEqual(report['bodies'][1]['analysis_status'], 'not_analyzed_budget')
        self.assertEqual(report['status'], 'analysis_partial_not_cad_complete')
        self.assertEqual(sorted(i for b in report['bodies'] for i in b['instance_triangle_ids']), list(range(24)))
        with np.load(self.root / 'output' / report['instances'][0]['mesh_arrays']) as arrays:
            self.assertEqual(len(arrays['faces']), 24)

    def test_same_location_components_remain_distinct(self):
        mesh = trimesh.creation.box(extents=[10, 8, 6])
        obj = '<object id="1">' + mesh_xml(mesh.vertices, mesh.faces) + '</object>'
        group = assembly(2, '<component objectid="1"/><component objectid="1"/>')
        self.package(objects=obj + group, items='<item objectid="2"/>')
        report = self.run_part(max_section_directions=1)
        self.assertEqual(report['body_count'], 2)
        self.assertFalse(report['instances'][0]['mesh_cleanup'][0]['cross_component_welding'])
        self.assertTrue(all('plate_plan' not in b for b in report['bodies']))

    def test_export_does_not_mutate_stack_and_maps_vertices_and_failed_events(self):
        box = trimesh.creation.box(extents=[30, 20, 1])
        stack = analyze_sections(box)
        before = deepcopy([s.metadata for s in stack.samples])
        face_map, vertex_map = np.arange(12) + 120, np.arange(8) + 80
        evidence = _stack_with_source_ids(stack, face_map, vertex_map)
        self.assertEqual(before, [s.metadata for s in stack.samples])
        for sample in evidence['samples']:
            for node in sample['intersection_nodes']:
                self.assertTrue(all(i >= 80 for i in node['source_key'][1:]))
            self.assertTrue(all(segment['source_triangle_id'] >= 120 for segment in sample['segments']))
        aliases = {'source_triangle_ids': [1, 2], 'vertex_ids': [3, 4]}
        failure = {'event_triangle_ids': [0, 1], 'on_plane_vertex_ids': [0],
                   'source_key': ['vertex', 0]}
        original = {'first': aliases, 'again': aliases, 'failure': failure}
        mapped = map_source_indices(original, face_map, vertex_map, copy_data=True)
        self.assertEqual(mapped['first']['source_triangle_ids'], [121, 122])
        self.assertIs(mapped['first'], mapped['again'])
        self.assertEqual(mapped['failure']['event_triangle_ids'], [120, 121])
        self.assertEqual(mapped['failure']['on_plane_vertex_ids'], [80])
        self.assertEqual(mapped['failure']['source_key'], ['vertex', 80])
        self.assertEqual(original['first']['source_triangle_ids'], [1, 2])

    def test_relationship_failure_does_not_erase_independent_sections(self):
        self.package(trimesh.creation.cylinder(radius=5, height=20, sections=32))
        with patch('assistant.feature_relationships', side_effect=ValueError('advisory unavailable')):
            report = self.run_part(max_section_directions=1)
        body = report['bodies'][0]
        self.assertIn('relationship_analysis_error', body)
        self.assertEqual(body['section_analyses'][0]['status'], 'sampled')
        self.assertIn('full_section_evidence', body)
        self.assertEqual(report['status'], 'analysis_partial_not_cad_complete')

    def test_generated_winding_fraction_never_exceeds_one(self):
        from geometry_analysis import cylinder_candidates
        from analytic_surfaces import additional_surface_candidates
        from feature_relations import feature_relationships
        for count in (12, 17, 31, 63, 97, 128):
            mesh = trimesh.creation.cylinder(radius=4.13, height=2.73, sections=count)
            mesh.faces = mesh.faces[:, ::-1]
            candidates = cylinder_candidates(mesh)
            for candidate in candidates:
                fraction = candidate['hole']['inward_normal_area_fraction']
                self.assertGreaterEqual(fraction, 0.)
                self.assertLessEqual(fraction, 1.)
            feature_relationships(mesh, {'cylinders': candidates})
        sphere = trimesh.creation.icosphere(subdivisions=2)
        sphere.faces = sphere.faces[:, ::-1]
        for candidate in additional_surface_candidates(sphere)['spheres']:
            self.assertLessEqual(candidate['inward_normal_area_fraction'], 1.)

    def test_direct_plate_analyzer_cannot_hide_skipped_build_items(self):
        from reverse_assistant import analyze
        self.package(items='<item objectid="1"/><item objectid="1" printable="0"/>')
        with self.assertRaisesRegex(ValueError, 'document_coverage_complete'):
            analyze(self.source, self.root / 'direct')
        self.assertFalse((self.root / 'direct').exists())

    def test_direct_entry_maps_compacted_vertices_to_snapshot(self):
        from reverse_assistant import analyze
        box = trimesh.creation.box(extents=[30, 20, 1])
        mesh = trimesh.Trimesh(np.vstack([[999, 999, 999], box.vertices]), box.faces + 1, process=False)
        self.package(mesh)
        plan = analyze(self.source, self.root / 'direct')
        self.assertEqual(plan['section_source_vertex_id_scope'], 'build_instance_vertices')
        for sample in plan['section_checks']:
            for node in sample['intersection_nodes']:
                if node['source_key'][0] != 'edge':
                    continue
                ids = node['source_key'][1:]
                self.assertTrue(all(i >= 1 for i in ids))
                rotation = np.asarray(plan['section_frame']['local_to_source_rotation'])
                origin = np.asarray(plan['section_frame']['origin_mm'])
                world = rotation @ np.asarray(node['point_xyz_mm']) + origin
                a, b = mesh.vertices[ids]
                fraction = np.dot(world - a, b - a) / np.dot(b - a, b - a)
                np.testing.assert_allclose(world, a + fraction * (b - a), atol=1e-9)

    def test_direct_entry_hash_matches_snapshot_despite_source_replacement(self):
        import hashlib
        import reverse_assistant as direct
        self.package()
        original = self.source.read_bytes()
        real_loader = direct.load_document
        def replacing_source(snapshot, *args, **kwargs):
            result = real_loader(snapshot, *args, **kwargs)
            self.source.write_bytes(b'changed-after-snapshot-read')
            return result
        with patch('reverse_assistant.load_document', side_effect=replacing_source):
            plan = direct.analyze(self.source, self.root / 'direct')
        self.assertEqual(plan['source_sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual((self.root / 'direct' / 'source.3mf').read_bytes(), original)
        self.assertEqual(plan['source_provenance']['source_path'],
                         str((self.root / 'direct' / 'source.3mf').resolve()))

    def test_sparse_shell_partitioning_preserves_order_and_every_face(self):
        count = 1200
        vertices = np.zeros((count, 3, 3))
        vertices[:, :, 0] = np.arange(count)[:, None] * 3
        vertices[:, 1, 0] += 1
        vertices[:, 2, 1] = 1
        mesh = trimesh.Trimesh(vertices.reshape(-1, 3), np.arange(count * 3).reshape(-1, 3), process=False)
        parts = split_bodies(mesh)
        self.assertEqual(len(parts), count)
        self.assertEqual([p.instance_triangle_ids.tolist() for p in parts], [[i] for i in range(count)])
        for i, partition in enumerate(parts):
            np.testing.assert_array_equal(partition.mesh.vertices, mesh.vertices[3 * i:3 * i + 3])

    def test_transitive_normal_band_is_not_a_plane_on_a_torus(self):
        from geometry_analysis import inspect_mesh
        mesh = trimesh.creation.torus(major_radius=5, minor_radius=1,
                                     major_sections=64, minor_sections=24)
        record = inspect_mesh(mesh, 0, 0)
        self.assertEqual(record['planes'], [])
        self.assertEqual(record['cones'], [])
        self.assertEqual(record['spheres'], [])
        self.assertEqual(record['candidate_coverage']['unresolved_triangle_count'], len(mesh.faces))

    def test_invalid_section_face_indices_rejected_before_sampling(self):
        for faces in (np.array([[-1, 1, 2]]), np.array([[0, 1, 999]])):
            mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 1], [0, 1, 2]], faces=faces, process=False)
            with self.subTest(faces=faces.tolist()), self.assertRaises(ValueError):
                analyze_sections(mesh)

    def test_source_name_and_section_plan_scope_agree_with_full_evidence(self):
        self.package()
        report = self.run_part(max_section_directions=1)
        body = report['bodies'][0]
        plan = json.loads((self.root / 'output' / body['plate_plan']).read_text())
        self.assertEqual(plan['source_name'], self.source.name)
        self.assertEqual(plan['section_source_triangle_id_scope'], 'build_instance_faces')
        self.assertEqual(plan['section_source_vertex_id_scope'], 'build_instance_vertices')
        self.assertFalse(plan['approved_for_build'])
        self.assertFalse(plan['accepted_approximation'])


if __name__ == '__main__':
    unittest.main()
