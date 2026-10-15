"""Regression tests for actual multi-layer section geometry and safety limits."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon

from section_analysis import (
    SectionEvidence, SectionStack, SectionTopologyError,
    analyze_sections, polygons_to_section, section_at,
)


def rectangle(x0, y0, x1, y1):
    """Explicitly closed, without rounding or discarding vertices."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def layered_rectangular_mesh(levels, sizes):
    """Watertight capped mesh with one rectangle per Z event.

    Repeated event Zs with different sizes create a horizontal ledge; distinct
    event Zs create a side wall. No Boolean/triangulation extra dependency.
    """
    vertices = []
    faces = []
    for z, (width, depth) in zip(levels, sizes):
        vertices.extend([[-width / 2, -depth / 2, z], [width / 2, -depth / 2, z],
                         [width / 2, depth / 2, z], [-width / 2, depth / 2, z]])
    faces.extend([[0, 2, 1], [0, 3, 2]])
    for layer in range(len(levels) - 1):
        lower, upper = 4 * layer, 4 * (layer + 1)
        for corner in range(4):
            following = (corner + 1) % 4
            faces.extend([[lower + corner, lower + following, upper + following],
                          [lower + corner, upper + following, upper + corner]])
    last = 4 * (len(levels) - 1)
    faces.extend([[last, last + 1, last + 2], [last, last + 2, last + 3]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class CoordinateSectionMesh:
    """A stack fixture whose slices still pass through the public loop helper."""
    def __init__(self, paths_at_z, events=(0.0, 1.0)):
        self.paths_at_z = paths_at_z
        self.vertices = np.array([[0.0, 0.0, z] for z in events])

    def section(self, plane_origin, plane_normal):
        z = plane_origin[2]
        paths = self.paths_at_z(z)
        if paths is None:
            return None
        entities = []
        offset = 0
        for points in paths:
            indices = np.arange(len(points)) + offset
            if np.array_equal(points[0], points[-1]):
                indices[-1] = indices[0]
            entities.append(trimesh.path.entities.Line(indices))
            offset += len(points)
        return trimesh.path.Path3D(
            entities=entities,
            vertices=np.vstack([np.column_stack((p, np.full(len(p), z))) for p in paths]),
            process=False,
        )


class LoopEvidenceTests(unittest.TestCase):
    def assert_json(self, metadata):
        json.dumps(metadata, allow_nan=False)

    def test_single_rectangle_preserves_all_coordinates(self):
        points = [[123.125, -71.5], [126.75, -71.5], [131.125, -71.5],
                  [131.125, -65.5], [123.125, -65.5], [123.125, -71.5]]
        evidence = polygons_to_section([points], z=2.75)
        self.assertIsInstance(evidence, SectionEvidence)
        self.assertIsInstance(evidence.geometry, Polygon)
        metadata = evidence.metadata
        self.assertEqual(metadata['z_mm'], 2.75)
        self.assertEqual(metadata['loops'][0]['points_mm'], points)
        self.assertEqual(list(evidence.geometry.exterior.coords), [tuple(p) for p in points])
        self.assertEqual(metadata['topology_status'], 'valid')
        self.assertEqual(metadata['loop_count'], 1)
        self.assertEqual(metadata['outer_region_count'], 1)
        self.assertEqual(metadata['material_region_count'], 1)
        self.assertEqual(metadata['void_loop_count'], 0)
        self.assertEqual(metadata['max_nesting_depth'], 0)
        self.assertEqual(metadata['net_area_mm2'], 48.0)
        self.assert_json(metadata)

    def test_smallest_parent_and_parity_for_shuffled_nested_islands(self):
        # Hole, depth-4 island, separate exterior, depth-2 island, exterior,
        # depth-3 void: winding and input order must not determine topology.
        paths = [rectangle(-6, -6, 6, 6)[::-1], rectangle(-1, -1, 1, 1),
                 rectangle(20, 0, 22, 2), rectangle(-4, -4, 4, 4)[::-1],
                 rectangle(-10, -10, 10, 10), rectangle(-2, -2, 2, 2)]
        evidence = polygons_to_section(paths, z=8)
        loops = evidence.metadata['loops']
        expected_parents = [4, 5, None, 0, None, 3]
        expected_depths = [1, 4, 0, 2, 0, 3]
        self.assertEqual([p['parent_id'] for p in loops], expected_parents)
        self.assertEqual([p['depth'] for p in loops], expected_depths)
        self.assertEqual([p['id'] for p in loops], list(range(6)))
        self.assertEqual([p['role'] for p in loops],
                         ['void', 'material_island', 'material_outer',
                          'material_island', 'material_outer', 'void'])
        self.assertEqual([p['points_mm'] for p in loops], paths)
        self.assertEqual([p['area_mm2'] for p in loops], [144, 4, 4, 64, 400, 16])
        self.assertEqual(evidence.metadata['outer_region_count'], 2)
        self.assertEqual(evidence.metadata['material_region_count'], 4)
        self.assertEqual(evidence.metadata['void_loop_count'], 2)
        self.assertEqual(evidence.metadata['max_nesting_depth'], 4)
        self.assertEqual(evidence.metadata['loop_count'], 6)
        self.assertEqual(evidence.metadata['net_area_mm2'], 400 - 144 + 64 - 16 + 4 + 4)
        self.assertIsInstance(evidence.geometry, MultiPolygon)
        self.assertEqual(len(evidence.geometry.geoms), 4)
        self.assertTrue(evidence.geometry.is_valid)
        self.assertEqual(sorted(p.area for p in evidence.geometry.geoms), [4, 4, 48, 256])
        self.assert_json(evidence.metadata)

    def test_two_outers_each_with_hole(self):
        paths = [rectangle(0, 0, 10, 10), rectangle(2, 2, 4, 4),
                 rectangle(20, 0, 30, 10)[::-1], rectangle(22, 2, 24, 4)[::-1]]
        evidence = polygons_to_section(paths)
        self.assertEqual(evidence.metadata['outer_region_count'], 2)
        self.assertEqual(evidence.metadata['material_region_count'], 2)
        self.assertEqual(evidence.metadata['void_loop_count'], 2)
        self.assertEqual(evidence.metadata['net_area_mm2'], 192)
        self.assertEqual([len(p.interiors) for p in evidence.geometry.geoms], [1, 1])

    def test_empty_section_is_explicit(self):
        evidence = polygons_to_section([], z=-12)
        self.assertIsInstance(evidence.geometry, Polygon)
        self.assertTrue(evidence.geometry.is_empty)
        self.assertEqual(evidence.metadata['topology_status'], 'empty')
        self.assertEqual(evidence.metadata['loop_count'], 0)
        self.assertEqual(evidence.metadata['net_area_mm2'], 0)
        self.assertEqual(evidence.metadata['topology_signature'], [])
        self.assert_json(evidence.metadata)

    def test_open_path_is_not_implicitly_closed_even_with_large_tolerance(self):
        for points in (rectangle(0, 0, 2, 2)[:-1],
                       [[0, 0], [2, 0], [2, 2], [0, 2], [0.000001, 0]]):
            with self.subTest(points=points):
                with self.assertRaises(SectionTopologyError) as caught:
                    polygons_to_section([points], tolerance=10)
                self.assertEqual(caught.exception.code, 'open_path')
                self.assertEqual(caught.exception.metadata['raw_paths_mm'], [points])
                self.assertEqual(caught.exception.metadata['topology_status'], 'invalid')
                self.assert_json(caught.exception.metadata)

    def test_cross_overlap_touch_and_duplicate_loops_are_errors(self):
        cases = [
            ([rectangle(0, 0, 4, 4), rectangle(2, -1, 5, 3)], 'crossing_loops'),
            ([rectangle(0, 0, 4, 4), rectangle(2, 0, 5, 4)], 'overlapping_loop_boundaries'),
            ([rectangle(0, 0, 4, 4), rectangle(4, 4, 6, 6)], 'touching_loops'),
            ([rectangle(0, 0, 4, 4), rectangle(0, 0, 4, 4)[::-1]], 'overlapping_loop_boundaries'),
            # An otherwise contained void may not touch its parent's boundary.
            ([rectangle(0, 0, 4, 4), [[0, 2], [1, 1], [1, 3], [0, 2]]], 'touching_loops'),
        ]
        for paths, code in cases:
            with self.subTest(code=code, paths=paths):
                with self.assertRaises(SectionTopologyError) as caught:
                    polygons_to_section(paths)
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(caught.exception.metadata['raw_paths_mm'], paths)
                self.assert_json(caught.exception.metadata)

    def test_self_crossing_self_touch_nonfinite_and_degenerate_are_rejected(self):
        cases = [
            [[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]],
            [[0, 0], [2, 0], [1, 1], [2, 2], [0, 2], [1, 1], [0, 0]],
            [[0, 0], [float('nan'), 0], [2, 2], [0, 0]],
            [[0, 0], [float('inf'), 0], [2, 2], [0, 0]],
            [[0, 0], [1, 0], [2, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]],
        ]
        for path in cases:
            with self.subTest(path=path):
                with self.assertRaises(SectionTopologyError) as caught:
                    polygons_to_section([path])
                self.assert_json(caught.exception.metadata)

    def test_invalid_tolerance_rejected_by_all_apis_before_mesh_access(self):
        for tolerance in (-0.01, float('nan'), float('inf'), -float('inf')):
            for invoke in (lambda: polygons_to_section([], tolerance=tolerance),
                           lambda: section_at(None, 0, tolerance=tolerance),
                           lambda: analyze_sections(None, tolerance=tolerance)):
                with self.subTest(tolerance=tolerance):
                    with self.assertRaisesRegex(ValueError, 'tolerance'):
                        invoke()
        self.assertEqual(polygons_to_section([], tolerance=0).metadata['tolerance_mm'], 0)

    def test_nonfinite_z_is_rejected(self):
        for z in (float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'z'):
                polygons_to_section([], z=z)
            with self.assertRaisesRegex(ValueError, 'z'):
                section_at(None, z)


class MeshSectionTests(unittest.TestCase):
    def test_box_preserves_actual_local_xy_coordinates(self):
        mesh = trimesh.creation.box(extents=[10, 8, 2])
        mesh.apply_translation([23, -15, 1])
        z = 0.37
        # Retain every source-edge intersection, but do not depend on Path3D's
        # cycle ordering or its default coordinate-merging tolerance.
        edges = {tuple(sorted((int(a), int(b)))) for face in mesh.faces
                 for a, b in zip(face, np.roll(face, -1))}
        expected = set()
        for a, b in sorted(edges):
            da, db = mesh.vertices[a, 2] - z, mesh.vertices[b, 2] - z
            if da * db < 0:
                xy = mesh.vertices[a, :2] + da / (da - db) * (mesh.vertices[b, :2] - mesh.vertices[a, :2])
                expected.add(tuple(xy))
        evidence = section_at(mesh, z)
        self.assertEqual(evidence.metadata['source'], 'source_mesh_edge_intersections')
        self.assertEqual({tuple(p) for loop in evidence.metadata['loops'] for p in loop['points_mm']}, expected)
        self.assertEqual(evidence.metadata['segment_count'], len(expected))
        self.assertEqual(evidence.metadata['traversed_segment_count'], len(expected))
        self.assertTrue(evidence.metadata['all_segments_traversed_once'])
        self.assertEqual(evidence.geometry.bounds, (18, -19, 28, -11))
        self.assertAlmostEqual(evidence.metadata['net_area_mm2'], 80)
        self.assertEqual(evidence.metadata['z_mm'], z)

    def test_tube_with_void(self):
        mesh = trimesh.creation.annulus(r_min=2, r_max=5, height=2, sections=64)
        evidence = section_at(mesh, 0.17)
        self.assertEqual(evidence.metadata['loop_count'], 2)
        self.assertEqual(evidence.metadata['void_loop_count'], 1)
        self.assertEqual(evidence.metadata['material_region_count'], 1)
        self.assertEqual(evidence.metadata['max_nesting_depth'], 1)
        self.assertEqual(len(evidence.geometry.interiors), 1)
        polygon_area = 64 / 2 * np.sin(2 * np.pi / 64) * (25 - 4)
        self.assertAlmostEqual(evidence.geometry.area, polygon_area)

    def test_nonplanar_or_nonfinite_discrete_paths_are_errors(self):
        for xyz, code in (
            (np.column_stack((rectangle(0, 0, 2, 2), np.full(5, .02))), 'nonplanar_section_path'),
            (np.column_stack((rectangle(0, 0, 2, 2), np.full(5, np.nan))), 'nonfinite_coordinates'),
        ):
            path = type('Path', (), {'discrete': [xyz], 'dangling': [], 'is_closed': True})()
            mesh = type('Mesh', (), {'section': lambda self, **kwargs: path})()
            with self.subTest(code=code):
                with self.assertRaises(SectionTopologyError) as caught:
                    section_at(mesh, 0, tolerance=1)
                self.assertEqual(caught.exception.code, code)
                json.dumps(caught.exception.metadata, allow_nan=False)

    def test_huge_z_extent_allows_only_interpolation_roundoff(self):
        mesh = trimesh.creation.box(extents=[10, 8, 1e6])
        self.assertAlmostEqual(section_at(mesh, .1).metadata['net_area_mm2'], 80)

    def test_multiple_mesh_bodies_with_an_island_inside_a_tube(self):
        tube = trimesh.creation.annulus(r_min=3, r_max=5, height=2, sections=32)
        island = trimesh.creation.box(extents=[2, 2, 2])
        outside = trimesh.creation.box(extents=[2, 2, 2])
        outside.apply_translation([12, 0, 0])
        evidence = section_at(trimesh.util.concatenate([tube, island, outside]), 0)
        self.assertEqual(evidence.metadata['loop_count'], 4)
        self.assertEqual(evidence.metadata['outer_region_count'], 2)
        self.assertEqual(evidence.metadata['material_region_count'], 3)
        self.assertEqual(evidence.metadata['void_loop_count'], 1)
        self.assertEqual(evidence.metadata['max_nesting_depth'], 2)
        self.assertEqual(len(evidence.geometry.geoms), 3)
        self.assertTrue(evidence.geometry.is_valid)

    def test_no_intersection(self):
        evidence = section_at(trimesh.creation.box(), 100)
        self.assertTrue(evidence.geometry.is_empty)
        self.assertEqual(evidence.metadata['topology_status'], 'empty')

    def test_actual_open_mesh_does_not_silently_use_only_closed_body(self):
        open_mesh = trimesh.creation.box(extents=[4, 4, 2])
        index = next(i for i, t in enumerate(open_mesh.triangles) if t[:, 2].min() < 0 < t[:, 2].max())
        open_mesh.update_faces([i for i in range(len(open_mesh.faces)) if i != index])
        closed_mesh = trimesh.creation.box(extents=[2, 2, 2])
        closed_mesh.apply_translation([10, 0, 0])
        mesh = trimesh.util.concatenate([open_mesh, closed_mesh])
        with self.assertRaises(SectionTopologyError) as caught:
            section_at(mesh, 0)
        self.assertEqual(caught.exception.code, 'open_or_branched_section')
        metadata = caught.exception.metadata
        # Keep both the open and closed body's segments; do not accept a subset.
        self.assertEqual(len(metadata['raw_paths_mm']), metadata['segment_count'])
        self.assertEqual(len(metadata['details']['bad_degree_nodes']), 2)
        self.assertEqual({n['degree'] for n in metadata['details']['bad_degree_nodes']}, {1})
        self.assertGreater(metadata['segment_count'], 8)
        self.assertEqual(metadata['traversed_segment_count'], 0)
        self.assertFalse(metadata['all_segments_traversed_once'])
        self.assertEqual(metadata['extraction_method'], 'source_edge_id_plane_intersection')
        json.dumps(metadata, allow_nan=False)
        # Duplicate a real crossing face: its shared endpoints now have degree
        # three and must fail, rather than deduplicating source segments.
        branched = trimesh.Trimesh(vertices=closed_mesh.vertices.copy(),
                                   faces=np.vstack([closed_mesh.faces, closed_mesh.faces[index]]), process=False)
        with self.assertRaises(SectionTopologyError) as caught:
            section_at(branched, 0)
        self.assertEqual(caught.exception.code, 'open_or_branched_section')
        self.assertEqual({n['degree'] for n in caught.exception.metadata['details']['bad_degree_nodes']}, {3})

    def test_sub_merge_tolerance_thin_and_nearby_loops_stay_separate(self):
        # Both wall width and gap are below Trimesh tol_path.merge=1e-5.
        thin = trimesh.creation.box(extents=[2, 4e-6, 2])
        nearby = thin.copy()
        nearby.apply_translation([0, 6e-6, 0])
        mesh = trimesh.util.concatenate([thin, nearby])
        a = section_at(mesh, .137, tolerance=10)
        b = section_at(mesh, .137, tolerance=0)
        self.assertEqual(a.metadata['loop_count'], 2)
        self.assertEqual(a.metadata['material_region_count'], 2)
        self.assertAlmostEqual(a.geometry.area, 16e-6, places=18)
        self.assertAlmostEqual(a.geometry.geoms[0].distance(a.geometry.geoms[1]), 2e-6, places=18)
        self.assertEqual(a.metadata['intersection_nodes'], b.metadata['intersection_nodes'])
        self.assertEqual(a.metadata['loops'], b.metadata['loops'])
        self.assertEqual(a.metadata['segment_count'], 16)
        self.assertTrue(a.metadata['all_segments_traversed_once'])
        for loop in a.metadata['loops']:
            self.assertEqual(loop['intersection_node_ids'][0], loop['intersection_node_ids'][-1])
            self.assertEqual(loop['points_mm'][0], loop['points_mm'][-1])
            self.assertEqual(len(loop['source_triangle_ids']), len(loop['segment_ids']))
        json.dumps(a.metadata, allow_nan=False)

    def test_real_backend_never_calls_mesh_section_or_changes_source(self):
        mesh = trimesh.creation.box(extents=[10, 8, 2])
        # Reverse a face deliberately: ID slicing must neither rely on nor fix winding.
        mesh.faces[0] = mesh.faces[0, ::-1]
        vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
        with patch.object(mesh, 'section', side_effect=AssertionError('Path3D must not be used')):
            a = section_at(mesh, .137)
            b = section_at(mesh, .137)
        self.assertEqual(a.metadata, b.metadata)
        self.assertEqual(a.metadata['backend'], 'numpy_source_triangle_edges')
        self.assertEqual(a.metadata['numeric_on_plane_epsilon'],
                         128 * np.finfo(float).eps * np.max(np.abs(vertices)))
        self.assertEqual(a.metadata['self_intersection_status'], 'not_checked')
        self.assertFalse(a.metadata['solid_validity_certified'])
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)
        nodes = {n['id']: n for n in a.metadata['intersection_nodes']}
        self.assertEqual(len({tuple(n['source_key']) for n in nodes.values()}), len(nodes))
        self.assertTrue(all(len(n['incident_segment_ids']) == 2 for n in nodes.values()))
        for segment in a.metadata['segments']:
            face = mesh.faces[segment['source_triangle_id']]
            for endpoint in segment['endpoint_ids']:
                key = nodes[endpoint]['source_key']
                self.assertEqual(key[0], 'edge')
                self.assertTrue(all(v in face for v in key[1:]))

    def test_event_vertices_coplanar_edges_and_triangles_are_ambiguities(self):
        cube = trimesh.creation.box(extents=[2, 2, 2])
        tangent = trimesh.Trimesh(vertices=[[5, 0, 0], [6, 0, 1], [5, 1, 1]],
                                  faces=[[0, 1, 2]], process=False)
        cases = [(trimesh.util.concatenate([cube, tangent]), 0, ['vertex_on_plane_event']),
                 (layered_rectangular_mesh([0, .5, 1], [(2, 2)] * 3), .5,
                  ['vertex_on_plane_event', 'coplanar_edges']),
                 (cube, 1, ['vertex_on_plane_event', 'coplanar_edges', 'coplanar_triangles'])]
        for mesh, z, kinds in cases:
            with self.subTest(kinds=kinds):
                with self.assertRaises(SectionTopologyError) as caught:
                    section_at(mesh, z, tolerance=10)
                error = caught.exception
                self.assertEqual(error.code, 'ambiguous_section_event')
                self.assertEqual(error.metadata['ambiguity_kinds'], kinds)
                self.assertTrue(error.metadata['on_plane_vertex_ids'])
                self.assertTrue(any(n['source_key'][0] == 'vertex' for n in error.metadata['intersection_nodes']))
                self.assertEqual(error.metadata['traversed_segment_count'], 0)
                self.assertFalse(error.metadata['all_segments_traversed_once'])
                json.dumps(error.metadata, allow_nan=False)
        near_event = trimesh.Trimesh(vertices=[[0, 0, 1e-15], [1, 0, 1], [0, 1, -1]],
                                     faces=[[0, 1, 2]], process=False)
        with self.assertRaises(SectionTopologyError) as caught:
            section_at(near_event, 0, tolerance=0)
        self.assertEqual(caught.exception.code, 'ambiguous_section_event')
        source_node = next(n for n in caught.exception.metadata['intersection_nodes']
                           if n['source_key'] == ['vertex', 0])
        self.assertEqual(source_node['point_xyz_mm'], [0, 0, 1e-15])  # No event-plane snap.
        stack = analyze_sections(cases[1][0])
        self.assertEqual(len(stack.samples), 6)
        self.assertEqual([f['error_code'] for f in stack.metadata['failed_samples']], ['ambiguous_section_event'])
        self.assertIsNone(stack.metadata['profile_varies'])
        self.assertFalse(stack.metadata['automatic_full_thickness_suggestion_allowed'])

    def test_zero_length_degenerate_segment_is_not_a_loop_or_endpoint_weld(self):
        # A zero-area triangle outside a closed body cuts to a point: ignore it,
        # with face evidence, rather than generating a spurious closed loop.
        cube = trimesh.creation.box(extents=[2, 2, 2])
        degenerate = trimesh.Trimesh(vertices=[[5, 0, -1], [5, 0, 1], [5, 0, 1]],
                                     faces=[[0, 1, 2]], process=False)
        combined = trimesh.util.concatenate([cube, degenerate])
        evidence = section_at(combined, 0)
        self.assertEqual(evidence.metadata['loop_count'], 1)
        self.assertEqual(evidence.metadata['ignored_degenerate_segment_count'], 1)
        self.assertEqual(evidence.metadata['ignored_degenerate_segments'][0]['source_triangle_id'], 12)
        self.assertEqual(evidence.metadata['segment_count'], 8)
        self.assertTrue(evidence.metadata['all_segments_traversed_once'])
        self.assertTrue(section_at(degenerate, 0).geometry.is_empty)
        # Collinear triangles may bridge distinct edge IDs at the same XY in
        # the source. Omitting their zero-length slices must NOT auto-snap the
        # remaining degree-one ends into a closed section.
        collapsed_wall = layered_rectangular_mesh([0, 1], [(2, 2), (2, 2)])
        collapsed_wall.vertices[1] = collapsed_wall.vertices[0]
        collapsed_wall.vertices[5] = collapsed_wall.vertices[4]
        with self.assertRaises(SectionTopologyError) as caught:
            section_at(collapsed_wall, .137)
        self.assertEqual(caught.exception.code, 'open_or_branched_section')
        self.assertEqual(caught.exception.metadata['ignored_degenerate_segment_count'], 2)
        self.assertTrue(caught.exception.metadata['details']['bad_degree_nodes'])
        json.dumps(caught.exception.metadata, allow_nan=False)

    def test_overlapping_closed_bodies_are_not_unioned(self):
        a = trimesh.creation.box(extents=[4, 4, 2])
        b = trimesh.creation.box(extents=[4, 4, 2])
        b.apply_translation([2, 1, 0])
        with self.assertRaises(SectionTopologyError):
            section_at(trimesh.util.concatenate([a, b]), 0)


class SectionStackTests(unittest.TestCase):
    def assert_stack_json(self, stack):
        json.dumps(stack.metadata, allow_nan=False)
        for sample in stack.samples:
            json.dumps(sample.metadata, allow_nan=False)

    def test_single_box_default_schedule_and_reference(self):
        mesh = trimesh.creation.box(extents=[10, 8, 2])
        mesh.apply_translation([23, -15, 1])
        stack = analyze_sections(mesh)
        self.assertIsInstance(stack, SectionStack)
        fractions = [s.metadata['sampling']['fraction'] for s in stack.samples]
        np.testing.assert_allclose(fractions, [.05, .1, .25, .5, .75, .9, .95])
        self.assertEqual(stack.metadata['reference_sample_index'], 2)
        self.assertEqual(stack.samples[2].metadata['z_mm'], .5)
        self.assertFalse(stack.metadata['profile_varies'])
        self.assertFalse(stack.metadata['sampling_complete'])
        self.assertFalse(stack.metadata['automatic_full_thickness_suggestion_allowed'])
        self.assertEqual(stack.metadata['failed_samples'], [])
        self.assertLess(stack.metadata['max_boundary_hausdorff_mm'], 1e-12)
        self.assertLess(stack.metadata['max_symmetric_difference_area_mm2'], 1e-12)
        self.assertFalse(stack.metadata['sampling_policy']['coverage_limited'])
        self.assertEqual(stack.metadata['sampling_policy']['event_layer_count'], 2)
        self.assert_stack_json(stack)

    def test_tube_is_constant_despite_slice_tessellation_changes(self):
        mesh = trimesh.creation.annulus(r_min=2, r_max=5, height=2, sections=64)
        stack = analyze_sections(mesh)
        self.assertFalse(stack.metadata['profile_varies'])
        self.assertFalse(stack.metadata['topology_changed'])
        self.assertLess(stack.metadata['max_boundary_hausdorff_mm'], 1e-10)
        self.assertLess(stack.metadata['max_symmetric_difference_area_mm2'], 1e-10)
        self.assertEqual(stack.metadata['failed_samples'], [])
        self.assert_stack_json(stack)

    def test_thin_step_is_found_between_default_layers_not_merged_by_user_tolerance(self):
        # Only .6% of the height: none of the seven defaults sees this protrusion.
        mesh = layered_rectangular_mesh(
            [0, .333, .333, .339, .339, 1],
            [(10, 8), (10, 8), (12, 8), (12, 8), (10, 8), (10, 8)])
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        stack = analyze_sections(mesh, tolerance=.05)
        metadata = stack.metadata
        self.assertTrue(metadata['profile_varies'])
        self.assertFalse(metadata['topology_changed'])
        self.assertEqual(metadata['failed_samples'], [])
        step_sample = next(s for s in stack.samples if .333 < s.metadata['z_mm'] < .339)
        self.assertAlmostEqual(step_sample.metadata['z_mm'], .336)
        self.assertAlmostEqual(step_sample.metadata['net_area_mm2'], 96)
        self.assertTrue(step_sample.metadata['comparison_to_reference']['profile_differs'])
        self.assertIn('vertex_event_midpoint', step_sample.metadata['sampling']['sources'])
        self.assertEqual(metadata['sampling_policy']['event_layer_count'], 4)
        self.assertGreaterEqual(metadata['max_boundary_hausdorff_mm'], .999)
        self.assertAlmostEqual(metadata['max_symmetric_difference_area_mm2'], 16)
        self.assertEqual(stack.samples[metadata['reference_sample_index']].metadata['sampling']['default_fraction'], .25)
        for sample in stack.samples:
            if sample.metadata['sampling']['default_fraction'] is not None:
                self.assertFalse(sample.metadata['comparison_to_reference']['profile_differs'])
        self.assert_stack_json(stack)

    def test_subnanometer_midpoint_is_not_replaced_by_default_at_step_boundary(self):
        # The step is only a little larger than the numerical event-cluster
        # resolution. The .25 default lies ON its endpoint, not in its interior.
        mesh = CoordinateSectionMesh(lambda z: [rectangle(0, 0, 10, 8)],
                                     events=(0, .25, .2500000015, 1))
        stack = analyze_sections(mesh)
        interior = [s for s in stack.samples if .25 < s.metadata['z_mm'] < .2500000015]
        self.assertEqual(len(interior), 1)
        self.assertIn('vertex_event_midpoint', interior[0].metadata['sampling']['sources'])
        self.assertEqual(stack.metadata['sampling_policy']['event_layer_count'], 4)
        self.assert_stack_json(stack)

    def test_z_roundoff_is_coalesced_without_fake_massive_event_count(self):
        mesh = layered_rectangular_mesh(np.linspace(0, 1, 90), [(10, 8)] * 90)
        # Model roundoff after a rigid transform and its inverse at otherwise two Z levels.
        mesh.vertices[:, 2] = np.repeat([0.0, 1.0], len(mesh.vertices) // 2)
        mesh.vertices[:, 2] += np.linspace(-2e-12, 2e-12, len(mesh.vertices))
        # Only vertices are relevant to this policy test, not the distorted faces.
        with patch('section_analysis.section_at', side_effect=lambda m, z, tolerance:
                   polygons_to_section([rectangle(0, 0, 10, 8)], z=z, tolerance=tolerance)):
            stack = analyze_sections(mesh)
        policy = stack.metadata['sampling_policy']
        self.assertGreater(policy['raw_unique_vertex_z_count'], 100)
        self.assertEqual(policy['event_layer_count'], 2)
        self.assertEqual(policy['event_interval_count'], 1)
        self.assertEqual(len(stack.samples), 7)
        self.assertFalse(policy['coverage_limited'])
        self.assertTrue(any('coalesced' in w for w in stack.metadata['warnings']))
        self.assert_stack_json(stack)

    def test_large_xy_rotation_roundoff_uses_numeric_mesh_scale_not_user_tolerance(self):
        mesh = layered_rectangular_mesh(np.linspace(0, 1, 90), [(10, 8)] * 90)
        mesh.vertices[:, :2] *= 1e6
        mesh.vertices[:, 2] = np.repeat([0.0, 1.0], len(mesh.vertices) // 2)
        mesh.vertices[:, 2] += np.linspace(-2e-8, 2e-8, len(mesh.vertices))
        with patch('section_analysis.section_at', side_effect=lambda m, z, tolerance:
                   polygons_to_section([rectangle(0, 0, 10, 8)], z=z, tolerance=tolerance)):
            stack = analyze_sections(mesh, tolerance=0)
        self.assertEqual(stack.metadata['sampling_policy']['event_layer_count'], 2)
        self.assertEqual(len(stack.samples), 7)
        self.assertGreater(stack.metadata['sampling_policy']['event_merge_tolerance_mm'], 2e-8)
        self.assert_stack_json(stack)

    def test_many_events_have_hard_cap_and_explicit_omitted_coverage(self):
        # Keep defaults strictly inside intervals: actual vertex-event planes
        # are intentionally ambiguous in the source-ID extractor.
        levels = np.linspace(0, 1, 101)
        levels[1:-1] += .001
        mesh = layered_rectangular_mesh(levels, [(10, 8)] * 101)
        stack = analyze_sections(mesh)
        policy = stack.metadata['sampling_policy']
        self.assertEqual(len(stack.samples), 40)
        self.assertEqual(policy['attempted_sample_count'], 40)
        self.assertEqual(policy['candidate_sample_count'], 107)
        self.assertEqual(policy['sampled_event_interval_count'], 33)
        self.assertEqual(policy['omitted_sample_count'], 67)
        self.assertEqual(len(policy['omitted_event_midpoints_mm']), 67)
        self.assertEqual(len(policy['omitted_event_interval_indices']), 67)
        self.assertTrue(policy['coverage_limited'])
        self.assertFalse(stack.metadata['sampling_complete'])
        self.assertFalse(stack.metadata['profile_varies'])
        self.assertEqual(stack.metadata['failed_samples'], [])
        self.assertEqual({s.metadata['sampling']['default_fraction'] for s in stack.samples
                          if s.metadata['sampling']['default_fraction'] is not None},
                         {.05, .1, .25, .5, .75, .9, .95})
        self.assertTrue(any('capped' in w for w in stack.metadata['warnings']))
        self.assert_stack_json(stack)

    def test_hole_center_movement_same_counts_same_area_still_varies(self):
        def paths(z):
            shift = 0 if z <= .5 else 1
            return [rectangle(-5, -5, 5, 5), rectangle(-1 + shift, -1, 1 + shift, 1)]
        stack = analyze_sections(CoordinateSectionMesh(paths))
        self.assertEqual({s.metadata['net_area_mm2'] for s in stack.samples}, {96})
        self.assertEqual({s.metadata['loop_count'] for s in stack.samples}, {2})
        self.assertEqual({s.metadata['void_loop_count'] for s in stack.samples}, {1})
        self.assertFalse(stack.metadata['topology_changed'])
        self.assertTrue(stack.metadata['profile_varies'])
        self.assertAlmostEqual(stack.metadata['max_boundary_hausdorff_mm'], 1)
        self.assertAlmostEqual(stack.metadata['max_symmetric_difference_area_mm2'], 4)
        self.assert_stack_json(stack)

    def test_actual_equal_area_sections_can_translate(self):
        mesh = trimesh.creation.box(extents=[10, 8, 2])
        shear = np.eye(4)
        shear[0, 2] = .5
        mesh.apply_transform(shear)
        stack = analyze_sections(mesh)
        self.assertTrue(stack.metadata['profile_varies'])
        self.assertEqual({round(s.metadata['net_area_mm2'], 8) for s in stack.samples}, {80})
        self.assertEqual(stack.metadata['failed_samples'], [])
        self.assertGreater(stack.metadata['max_boundary_hausdorff_mm'], .5)
        self.assert_stack_json(stack)

    def test_comparison_threshold_scales_with_user_tolerance(self):
        def paths(z):
            shift = 0 if z <= .5 else .02
            return [rectangle(-5 + shift, -5, 5 + shift, 5)]
        mesh = CoordinateSectionMesh(paths)
        self.assertFalse(analyze_sections(mesh, tolerance=.05).metadata['profile_varies'])
        precise = analyze_sections(mesh, tolerance=.005)
        self.assertTrue(precise.metadata['profile_varies'])
        self.assertAlmostEqual(precise.metadata['max_boundary_hausdorff_mm'], .02)
        self.assertAlmostEqual(precise.metadata['comparison_policy']['symmetric_difference_area_threshold_mm2'], .2)

    def test_new_tiny_hole_is_topology_change_even_below_metric_thresholds(self):
        def paths(z):
            loops = [rectangle(0, 0, 10, 10)]
            if z > .5:
                loops.append(rectangle(.001, 5, .011, 5.01))
            return loops
        stack = analyze_sections(CoordinateSectionMesh(paths), tolerance=.05)
        self.assertTrue(stack.metadata['profile_varies'])
        self.assertTrue(stack.metadata['topology_changed'])
        changed = stack.samples[-1].metadata['comparison_to_reference']
        self.assertFalse(changed['boundary_exceeds_tolerance'])
        self.assertFalse(changed['area_exceeds_threshold'])
        self.assertTrue(changed['topology_changed'])
        self.assertIn('void_loop_count', changed['changed_topology_fields'])
        self.assert_stack_json(stack)

    def test_component_and_nesting_changes_always_vary(self):
        def paths(z):
            if z <= .5:
                return [rectangle(0, 0, 10, 10), rectangle(2, 2, 8, 8), rectangle(4, 4, 6, 6)]
            return [rectangle(0, 0, 10, 10), rectangle(2, 2, 8, 8), rectangle(12, 0, 14, 2)]
        stack = analyze_sections(CoordinateSectionMesh(paths), tolerance=100)
        self.assertTrue(stack.metadata['profile_varies'])
        comparison = stack.samples[-1].metadata['comparison_to_reference']
        self.assertTrue(comparison['topology_changed'])
        self.assertIn('outer_region_count', comparison['changed_topology_fields'])
        self.assertIn('max_nesting_depth', comparison['changed_topology_fields'])
        self.assertEqual(stack.samples[0].metadata['material_region_count'], 2)
        self.assertEqual(stack.samples[-1].metadata['material_region_count'], 2)
        self.assert_stack_json(stack)

    def test_nesting_structure_detected_when_all_counts_and_max_depth_match(self):
        outer = [rectangle(0, 0, 12, 12), rectangle(20, 0, 32, 12)]
        def paths(z):
            # Either two holes in the first exterior, or one hole in each.
            holes = ([rectangle(2, 2, 4, 4), rectangle(7, 7, 9, 9)] if z <= .5 else
                     [rectangle(2, 2, 4, 4), rectangle(27, 7, 29, 9)])
            return outer + holes
        stack = analyze_sections(CoordinateSectionMesh(paths), tolerance=100)
        comparison = stack.samples[-1].metadata['comparison_to_reference']
        self.assertTrue(stack.metadata['profile_varies'])
        self.assertEqual(comparison['changed_topology_fields'], ['topology_signature'])
        self.assertFalse(comparison['boundary_exceeds_tolerance'])
        self.assertFalse(comparison['area_exceeds_threshold'])

    def test_loop_order_changes_do_not_imply_topology_changes(self):
        def paths(z):
            loops = [rectangle(0, 0, 10, 10), rectangle(2, 2, 8, 8), rectangle(4, 4, 6, 6)]
            return loops if z <= .5 else [p[::-1] for p in loops[::-1]]
        stack = analyze_sections(CoordinateSectionMesh(paths))
        self.assertFalse(stack.metadata['topology_changed'])
        self.assertFalse(stack.metadata['profile_varies'])
        self.assert_stack_json(stack)

    def test_nonreference_failed_sample_is_reported_not_declared_constant(self):
        def paths(z):
            if np.isclose(z, .75):
                return [rectangle(0, 0, 10, 10)[:-1]]
            return [rectangle(0, 0, 10, 10)]
        stack = analyze_sections(CoordinateSectionMesh(paths))
        metadata = stack.metadata
        self.assertEqual(len(stack.samples), 6)
        self.assertEqual(len(metadata['failed_samples']), 1)
        self.assertIsNone(metadata['profile_varies'])
        self.assertEqual(metadata['profile_status'], 'indeterminate')
        self.assertFalse(metadata['automatic_full_thickness_suggestion_allowed'])
        failure = metadata['failed_samples'][0]
        self.assertAlmostEqual(failure['z_mm'], .75)
        self.assertEqual(failure['error_code'], 'open_or_branched_section')
        self.assertTrue(failure['reason'])
        self.assertIsNotNone(failure['evidence'])
        self.assertEqual(metadata['reference_sample_index'], 2)
        self.assertTrue(any('failed' in w for w in metadata['warnings']))
        self.assert_stack_json(stack)

    def test_failed_event_midpoint_is_not_counted_as_successful_coverage(self):
        mesh = CoordinateSectionMesh(lambda z: [rectangle(0, 0, 10, 8)], events=(0, .3, 1))
        def extract(m, z, tolerance):
            if np.isclose(z, .15):
                raise RuntimeError('midpoint unavailable')
            return polygons_to_section([rectangle(0, 0, 10, 8)], z=z, tolerance=tolerance)
        with patch('section_analysis.section_at', side_effect=extract):
            stack = analyze_sections(mesh)
        policy = stack.metadata['sampling_policy']
        self.assertEqual(policy['attempted_event_interval_count'], 2)
        self.assertEqual(policy['sampled_event_interval_count'], 1)
        self.assertEqual(policy['successful_sample_count'], 8)
        self.assertEqual(len(stack.metadata['failed_samples']), 1)
        self.assertIsNone(stack.metadata['profile_varies'])
        self.assert_stack_json(stack)

    def test_failed_reference_is_not_replaced_by_different_layer(self):
        def paths(z):
            if np.isclose(z, .25):
                return [rectangle(0, 0, 10, 10), rectangle(5, 5, 15, 15)]
            return [rectangle(0, 0, 10, 10)]
        stack = analyze_sections(CoordinateSectionMesh(paths))
        self.assertIsNone(stack.metadata['reference_sample_index'])
        self.assertIsNone(stack.metadata['profile_varies'])
        self.assertIsNone(stack.metadata['max_boundary_hausdorff_mm'])
        self.assertIsNone(stack.metadata['max_symmetric_difference_area_mm2'])
        self.assertEqual(len(stack.metadata['failed_samples']), 1)
        self.assertTrue(all(s.metadata['comparison_to_reference'] is None for s in stack.samples))
        self.assertFalse(stack.metadata['automatic_full_thickness_suggestion_allowed'])
        self.assert_stack_json(stack)

    def test_observed_variation_remains_true_when_another_sample_fails(self):
        def paths(z):
            if np.isclose(z, .1):
                return [rectangle(0, 0, 10, 10)[:-1]]
            return [rectangle(0, 0, 10 if z <= .5 else 12, 10)]
        stack = analyze_sections(CoordinateSectionMesh(paths))
        self.assertTrue(stack.metadata['profile_varies'])
        self.assertEqual(len(stack.metadata['failed_samples']), 1)
        self.assertFalse(stack.metadata['automatic_full_thickness_suggestion_allowed'])
        self.assertEqual(stack.samples[stack.metadata['reference_sample_index']].metadata['z_mm'], .25)
        self.assert_stack_json(stack)

    def test_empty_vs_material_is_topology_variation_with_json_safe_distance(self):
        mesh = CoordinateSectionMesh(lambda z: None if z > .5 else [rectangle(0, 0, 10, 10)])
        stack = analyze_sections(mesh)
        self.assertTrue(stack.metadata['profile_varies'])
        self.assertTrue(stack.metadata['topology_changed'])
        self.assertEqual(stack.metadata['max_symmetric_difference_area_mm2'], 100)
        self.assertFalse(stack.metadata['boundary_hausdorff_comparison_complete'])
        self.assertEqual(stack.metadata['hausdorff_unavailable_sample_indices'], [4, 5, 6])
        self.assertIsNone(stack.samples[-1].metadata['comparison_to_reference']['boundary_hausdorff_mm'])
        self.assert_stack_json(stack)

    def test_all_empty_samples_are_not_full_thickness_reconstruction_evidence(self):
        stack = analyze_sections(CoordinateSectionMesh(lambda z: None))
        self.assertFalse(stack.metadata['profile_varies'])
        self.assertFalse(stack.metadata['sampling_complete'])
        self.assertFalse(stack.metadata['automatic_full_thickness_suggestion_allowed'])
        self.assertEqual(stack.metadata['max_boundary_hausdorff_mm'], 0)
        self.assertEqual(stack.metadata['max_symmetric_difference_area_mm2'], 0)
        self.assert_stack_json(stack)

    def test_unexpected_extraction_error_is_retained(self):
        mesh = CoordinateSectionMesh(lambda z: [rectangle(0, 0, 10, 10)])
        def extract(m, z, tolerance):
            if np.isclose(z, .9):
                raise RuntimeError('section backend unavailable')
            return polygons_to_section([rectangle(0, 0, 10, 10)], z=z, tolerance=tolerance)
        with patch('section_analysis.section_at', side_effect=extract):
            stack = analyze_sections(mesh)
        self.assertIsNone(stack.metadata['profile_varies'])
        self.assertEqual(len(stack.metadata['failed_samples']), 1)
        self.assertEqual(stack.metadata['failed_samples'][0]['error_type'], 'RuntimeError')
        self.assertIn('backend unavailable', stack.metadata['failed_samples'][0]['reason'])
        self.assert_stack_json(stack)

    def test_comparison_error_is_retained_not_silently_skipped(self):
        mesh = trimesh.creation.box(extents=[10, 8, 2])
        with patch('section_analysis._compare_sections', side_effect=RuntimeError('GEOS unavailable')):
            stack = analyze_sections(mesh)
        self.assertEqual(len(stack.samples), 7)
        self.assertEqual(len(stack.metadata['failed_samples']), 7)
        self.assertEqual({f['stage'] for f in stack.metadata['failed_samples']}, {'comparison'})
        self.assertIsNone(stack.metadata['profile_varies'])
        self.assertIsNone(stack.metadata['max_boundary_hausdorff_mm'])
        self.assertFalse(stack.metadata['automatic_full_thickness_suggestion_allowed'])
        self.assert_stack_json(stack)

    def test_invalid_mesh_z_extent_or_vertices_are_rejected(self):
        for vertices in ([], [[0, 0, 0], [1, 1, 0]], [[0, 0, float('nan')]], [[0, 0]]):
            mesh = type('Mesh', (), {'vertices': vertices})()
            with self.subTest(vertices=vertices):
                with self.assertRaises(ValueError):
                    analyze_sections(mesh)


if __name__ == '__main__':
    unittest.main()
