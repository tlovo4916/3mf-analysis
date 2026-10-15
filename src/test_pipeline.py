"""End-to-end LOCAL evidence tests; not actual SolidWorks or platform E2E."""
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
from shapely.geometry import Polygon

from assistant import process, _next_folder
from build_solidworks import validate
from geometry_analysis import plate_frame
from reverse_assistant import analyze, fit_circle, make_plate_plan
from section_analysis import analyze_sections, _bounded_boundary_hausdorff
from test_threemf_io import model_xml, box_object, assembly, mesh_xml


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'part.3mf'

    def package(self, objects=None, items='<item objectid="1"/>', mesh=None):
        if mesh is not None:
            objects = '<object id="1">' + mesh_xml(mesh.vertices, mesh.faces) + '</object>'
        if objects is None:
            mesh = trimesh.creation.box(extents=[30, 20, 1])
            objects = '<object id="1">' + mesh_xml(mesh.vertices, mesh.faces) + '</object>'
        with zipfile.ZipFile(self.source, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('3D/3dmodel.model', model_xml(objects, items))
        return self.source

    def test_source_snapshot_plan_and_mesh_arrays_are_consistent(self):
        self.package()
        out = self.root / 'output'
        with patch('assistant.load_document', wraps=__import__('threemf_io').load_document) as loader:
            report = process(self.source, out)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(report['instance_count'], 1)
        self.assertEqual(report['body_count'], 1)
        self.assertFalse(report['has_mesh_blockers'])
        self.assertFalse(report['has_body_analysis_errors'])
        self.assertEqual((out / 'source.3mf').read_bytes(), self.source.read_bytes())
        body = report['bodies'][0]
        self.assertEqual(body['plate_plan_status'], 'requires_human_review')
        plan = json.loads((out / body['plate_plan']).read_text())
        self.assertEqual(plan['schema_version'], 2)
        self.assertAlmostEqual(plan['volume_difference_percent'], 0)
        with self.assertRaises(ValueError):
            validate(plan)
        reviewed = deepcopy(plan)
        reviewed.update(approved_for_build=True, accepted_approximation=True)
        validate(reviewed)
        self.assertFalse(plan['source_placement_applied'])
        with np.load(out / report['instances'][0]['mesh_arrays'], allow_pickle=False) as arrays:
            self.assertEqual(len(arrays['faces']), 12)
            np.testing.assert_allclose(np.ptp(arrays['vertices_mm'], axis=0), [30, 20, 1])
        json.dumps(report, allow_nan=False)

    def test_sections_summary_and_lossless_full_evidence_match(self):
        self.package()
        out = self.root / 'output'
        report = process(self.source, out)
        body = report['bodies'][0]
        summary = json.loads((out / body['section_evidence']).read_text())
        with gzip.open(out / body['full_section_evidence'], 'rt') as stream:
            full = json.load(stream)
        self.assertEqual(summary['sample_count'], len(full['samples']))
        self.assertTrue(summary['full_evidence']['geometry_and_lineage_lossless'])
        for entry, sample in zip(summary['samples'], full['samples']):
            self.assertEqual(entry['net_area_mm2'], sample['net_area_mm2'])
            self.assertIn('points_mm', sample['loops'][0])
            self.assertNotIn('points_mm', entry['loops'][0])
        self.assertEqual(full['source_triangle_id_scope'], 'build_instance_faces')

    def test_two_components_are_one_instance_and_no_union_is_inferred(self):
        box = trimesh.creation.box(extents=[30, 20, 1])
        obj = '<object id="1">' + mesh_xml(box.vertices, box.faces) + '</object>'
        components = '<component objectid="1"/><component objectid="1" transform="1 0 0 0 1 0 0 0 1 60 0 0"/>'
        self.package(obj + assembly(2, components), '<item objectid="2"/>')
        report = process(self.source, self.root / 'output')
        self.assertEqual(report['instance_count'], 1)
        self.assertEqual(report['body_count'], 2)
        self.assertFalse(report['instances'][0]['boolean_union_of_components_performed'])
        for body in report['bodies']:
            self.assertEqual(body['plate_plan_status'], 'blocked')
            self.assertNotIn('plate_plan', body)
        self.assertTrue(all(i >= 12 for i in report['bodies'][1]['planes'][0]['source_triangle_ids']))

    def test_inverted_mesh_is_blocked_not_abs_volume_repaired(self):
        mesh = trimesh.creation.box(extents=[30, 20, 1])
        mesh.faces = mesh.faces[:, ::-1]
        self.package(mesh=mesh)
        report = process(self.source, self.root / 'output')
        self.assertTrue(report['has_mesh_blockers'])
        body = report['bodies'][0]
        self.assertLess(body['volume_mm3'], 0)
        self.assertEqual(body['strategy'], 'repair_required')
        self.assertEqual(body['plate_plan_status'], 'blocked')
        self.assertFalse(body['mesh_repair_performed'])

    def test_degenerate_source_face_retained_and_excluded_explicitly(self):
        box = trimesh.creation.box(extents=[30, 20, 1])
        mesh = trimesh.Trimesh(box.vertices, np.vstack([box.faces, [0, 1, 1]]), process=False)
        self.package(mesh=mesh)
        report = process(self.source, self.root / 'output')
        body = report['bodies'][0]
        self.assertEqual(body['source_triangle_count'], 13)
        self.assertEqual(body['analyzed_triangle_count'], 12)
        self.assertEqual(body['candidate_analysis_view']['skipped_degenerate_instance_triangle_ids'], [12])
        self.assertEqual(body['plate_plan_status'], 'blocked')
        self.assertIn('degenerate_triangles', body['blocking_issues'])

    def test_open_source_is_not_closed_by_body_split(self):
        mesh = trimesh.creation.box(extents=[30, 20, 1])
        mesh.update_faces(np.arange(11))
        self.package(mesh=mesh)
        report = process(self.source, self.root / 'output')
        body = report['bodies'][0]
        self.assertEqual(body['triangles'], 11)
        self.assertFalse(body['watertight'])
        self.assertTrue(report['has_mesh_blockers'])
        self.assertEqual(body['plate_plan_status'], 'blocked')

    def test_invalid_sibling_retained_without_losing_valid_body(self):
        mesh = trimesh.creation.box(extents=[30, 20, 1])
        vertices = np.vstack([mesh.vertices, [[80, 0, 0], [81, 0, 0], [82, 0, 0]]])
        faces = np.vstack([mesh.faces, [8, 9, 10]])
        self.package(mesh=trimesh.Trimesh(vertices, faces, process=False))
        report = process(self.source, self.root / 'output')
        self.assertEqual(report['body_count'], 2)
        self.assertTrue(report['has_body_analysis_errors'])
        self.assertEqual(len(report['bodies'][0]['planes']), 6)
        self.assertEqual(report['bodies'][1]['triangles'], 1)
        self.assertIn('analysis_error', report['bodies'][1])

    def test_malformed_source_batch_does_not_hide_failure(self):
        from assistant import main
        import sys
        self.source.write_bytes(b'not a zip')
        out = self.root / 'output'
        with patch.object(sys, 'argv', ['assistant.py', str(self.source), '--out', str(out)]):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 1)
        error = json.loads((out / '000-part' / 'error.json').read_text())
        self.assertFalse(error['ok'])
        self.assertIn('ZIP', error['error'])

    def test_reused_output_refused_and_cli_gets_new_folder(self):
        self.package()
        out = self.root / 'output'
        process(self.source, out)
        original = (out / 'analysis.json').read_bytes()
        with self.assertRaises(FileExistsError):
            process(self.source, out)
        self.assertEqual((out / 'analysis.json').read_bytes(), original)
        self.assertEqual(_next_folder(out), self.root / 'output-run-001')

    def test_tolerance_validation_happens_before_reading_or_creating_output(self):
        for value in (0, -.1, .6, True, float('nan'), float('inf'), '0.05'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                process(self.source, self.root / 'output', value)
        self.assertFalse((self.root / 'output').exists())

    def test_direct_analyze_refuses_inverted_or_existing_output(self):
        self.package()
        plan = analyze(self.source, self.root / 'direct')
        self.assertFalse(plan['approved_for_build'])
        with self.assertRaises(FileExistsError):
            analyze(self.source, self.root / 'direct')
        mesh = trimesh.creation.box(extents=[30, 20, 1])
        mesh.faces = mesh.faces[:, ::-1]
        self.package(mesh=mesh)
        with self.assertRaises(ValueError):
            analyze(self.source, self.root / 'inverted')
        self.assertFalse((self.root / 'inverted').exists())

    def test_untrusted_source_name_is_html_escaped(self):
        self.package()
        source = self.root / '<script>alert.3mf'
        self.source.rename(source)
        out = self.root / 'output'
        process(source, out)
        content = (out / 'index.html').read_text()
        self.assertIn('&lt;script&gt;alert.3mf', content)
        self.assertNotIn('<script>', content)

    def test_fitted_circle_translation_and_chord_error(self):
        angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        points = np.c_[3 + 5 * np.cos(angles), -2 + 5 * np.sin(angles)]
        fit = fit_circle(points + [1e9, -1e9], .01)
        self.assertAlmostEqual(fit['diameter_mm'], 10, places=6)
        self.assertGreater(fit['max_boundary_radial_error_mm'], fit['radial_error_mm'])
        coarse = np.c_[5 * np.cos(angles[::8]), 5 * np.sin(angles[::8])]
        self.assertIsNone(fit_circle(coarse, .01))
        self.assertIsNone(fit_circle([[0, 0], [1, 0], [1, 1], [0, 1]], .05))

    def test_bounded_boundary_distances_enclose_dense_vertex_distance(self):
        from shapely import hausdorff_distance
        a = Polygon([[0, 0], [5, 0], [5, 3], [0, 3]])
        b = Polygon([[0, 0], [5, 0], [5, 3], [2.5, 3.3], [0, 3]])
        lower, upper, error = _bounded_boundary_hausdorff(a, b, .05)
        reference = float(hausdorff_distance(a.boundary, b.boundary, densify=.02))
        self.assertLessEqual(lower, reference + 1e-10)
        self.assertGreaterEqual(upper, reference - 1e-10)
        self.assertLessEqual(error, .005 + 1e-12)

    def test_tube_plan_uses_inner_loop_but_not_design_hole_claim(self):
        mesh = trimesh.creation.annulus(r_min=3, r_max=15, height=1, sections=128)
        rotation, origin, local = plate_frame(mesh)
        stack = analyze_sections(local)
        plan = make_plate_plan(mesh, local, rotation, origin, stack, 'tube.3mf', '0' * 64)
        self.assertEqual(len(plan['cuts']), 1)
        self.assertEqual(plan['cuts'][0]['type'], 'circle')
        self.assertEqual(plan['cuts'][0]['fit_scope'], 'closed_section_polyline_not_design_hole')
        self.assertFalse(plan['cad_reconstructed'])
        plan.update(approved_for_build=True, accepted_approximation=True)
        validate(plan)


if __name__ == '__main__':
    unittest.main()
