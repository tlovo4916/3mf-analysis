import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from mesh_evidence import analysis_view, map_record_to_instance, mesh_quality, split_bodies
from geometry_analysis import inspect_mesh


class MeshEvidenceTests(unittest.TestCase):
    def test_open_mesh_is_not_filled_by_partitioning(self):
        mesh = trimesh.creation.box(extents=[20, 10, 1])
        mesh.update_faces(np.arange(len(mesh.faces) - 1))
        original = mesh.faces.copy()
        with patch.object(trimesh.Trimesh, 'fill_holes', side_effect=AssertionError('No repair')):
            parts = split_bodies(mesh)
        self.assertEqual(len(parts), 1)
        self.assertEqual(len(parts[0].mesh.faces), len(original))
        self.assertFalse(parts[0].mesh.is_watertight)
        np.testing.assert_array_equal(mesh.faces, original)
        self.assertIn('open_boundary_edges', mesh_quality(parts[0].mesh)['blocking_issues'])

    def test_all_faces_and_source_indices_survive_multiple_shells(self):
        first = trimesh.creation.box()
        second = first.copy()
        second.apply_translation([5, 0, 0])
        mesh = trimesh.util.concatenate([first, second])
        parts = split_bodies(mesh)
        self.assertEqual(len(parts), 2)
        ids = np.concatenate([part.instance_triangle_ids for part in parts])
        np.testing.assert_array_equal(np.sort(ids), np.arange(len(mesh.faces)))
        for part in parts:
            np.testing.assert_array_equal(part.mesh.triangles, mesh.triangles[part.instance_triangle_ids])

    def test_degenerate_duplicate_and_isolated_faces_not_silently_deleted(self):
        box = trimesh.creation.box()
        mesh = trimesh.Trimesh(np.vstack([box.vertices, [[8, 0, 0], [9, 0, 0], [10, 0, 0]]]),
                               np.vstack([box.faces, box.faces[0], [0, 0, 0], [8, 9, 10]]), process=False)
        parts = split_bodies(mesh)
        self.assertEqual(sum(len(part.mesh.faces) for part in parts), 15)
        quality = mesh_quality(mesh)
        self.assertEqual(quality['duplicate_triangle_count'], 1)
        self.assertEqual(quality['degenerate_triangle_ids'], [13, 14])
        self.assertIn('nonmanifold_edges', quality['blocking_issues'])

    def test_inward_and_inconsistent_winding_are_distinct_blockers(self):
        inward = trimesh.creation.box()
        inward.faces = inward.faces[:, ::-1]
        quality = mesh_quality(inward)
        self.assertTrue(quality['watertight'])
        self.assertTrue(quality['winding_consistent'])
        self.assertEqual(quality['signed_volume_mm3'], -1)
        self.assertIn('nonpositive_or_nonfinite_signed_volume', quality['blocking_issues'])
        inconsistent = trimesh.creation.box()
        faces = inconsistent.faces.copy()
        faces[0] = faces[0, ::-1]
        inconsistent.faces = faces
        self.assertIn('inconsistent_face_winding', mesh_quality(inconsistent)['blocking_issues'])

    def test_unused_vertices_do_not_distort_volume(self):
        box = trimesh.creation.box(extents=[5, 8, 2])
        mesh = trimesh.Trimesh(np.vstack([box.vertices, [[1e15, -1e15, 0]]]), box.faces, process=False)
        mesh.apply_translation([1e9, -1e9, 1e9])
        quality = mesh_quality(mesh)
        self.assertAlmostEqual(quality['signed_volume_mm3'], 80)
        self.assertEqual(quality['unused_vertices'], 1)
        self.assertFalse(quality['solid_validity_certified'])
        self.assertEqual(quality['self_intersection_status'], 'not_checked')

    def test_analysis_view_skips_only_flagged_faces_and_keeps_source(self):
        box = trimesh.creation.box(extents=[20, 10, 1])
        mesh = trimesh.Trimesh(box.vertices, np.vstack([box.faces, [0, 1, 1]]), process=False)
        part = split_bodies(mesh)[0]
        view, metadata = analysis_view(part)
        self.assertEqual(len(mesh.faces), 13)
        self.assertEqual(len(part.mesh.faces), 13)
        self.assertEqual(len(view.mesh.faces), 12)
        self.assertEqual(metadata['skipped_degenerate_instance_triangle_ids'], [12])
        self.assertEqual(metadata['faces_removed_from_source'], 0)
        self.assertIn('degenerate_triangles', metadata['source_quality_blockers_preserved'])

    def test_source_mapping_is_not_double_applied_to_shared_boundary_alias(self):
        first = trimesh.creation.box(extents=[20, 10, 1])
        second = first.copy()
        second.apply_translation([60, 0, 0])
        mesh = trimesh.util.concatenate([first, second])
        part = split_bodies(mesh)[1]
        view, _ = analysis_view(part)
        record = inspect_mesh(view.mesh, 0, 1)
        component = {'component_index': 0, 'model_path': '3D/part.model', 'object_id': 5,
                     'component_path': [], 'face_start': 0, 'face_end': 24,
                     'source_to_world_mm': np.eye(4).tolist(), 'original_unit': 'millimeter'}
        mapped = map_record_to_instance(record, view, {'components': [component]}, 0)
        for candidate in mapped['planes']:
            self.assertTrue(all(12 <= i < 24 for i in candidate['source_triangle_ids']))
            self.assertEqual(candidate['source_triangle_id_scope'], 'build_instance_faces')
            self.assertTrue(candidate['candidate_id'].startswith('instance_0000_shell_0001_'))
            for loop in candidate['boundary_loops']:
                np.testing.assert_array_equal(loop['points_mm'], mesh.vertices[loop['vertex_ids']])
                for edge_use in loop['edge_uses']:
                    self.assertTrue(12 <= edge_use['source_triangle_id'] < 24)
                    self.assertTrue(8 <= edge_use['start_vertex_id'] < 16)


if __name__ == '__main__':
    unittest.main()
