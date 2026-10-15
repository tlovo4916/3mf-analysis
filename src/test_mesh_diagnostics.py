"""Boundary/shell evidence must remain bounded, uncertain, and non-mutating."""
from collections import Counter
from contextlib import ExitStack, contextmanager
from itertools import combinations
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from mesh_diagnostics import mesh_advisories, shell_relationships
from mesh_evidence import MeshBody, split_bodies


@contextmanager
def no_repairs_or_certification():
    with ExitStack() as stack:
        for name in ('fill_holes', 'merge_vertices', 'fix_normals', 'invert',
                     'remove_degenerate_faces', 'remove_duplicate_faces', 'process',
                     'contains', 'split'):
            stack.enter_context(patch.object(trimesh.Trimesh, name, create=True,
                                             side_effect=AssertionError(f'Forbidden {name}')))
        yield


def inward_box(extents=(1., 1., 1.)):
    mesh = trimesh.creation.box(extents=extents)
    mesh.faces = mesh.faces[:, ::-1]
    return mesh


def disconnected_triangles(count=20):
    vertices = np.arange(count * 9, dtype=float).reshape((-1, 3)) * .0001
    faces = np.arange(len(vertices)).reshape((-1, 3))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def l_prism():
    """A single concave, positive closed shell; its missing quadrant is outside."""
    polygon = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], dtype=float)
    vertices = np.vstack([np.column_stack([polygon, np.zeros(6)]),
                          np.column_stack([polygon, np.full(6, 2.)])])
    triangulation = [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]]
    faces = [tri[::-1] for tri in triangulation]
    faces.extend([[v + 6 for v in tri] for tri in triangulation])
    for i in range(6):
        j = (i + 1) % 6
        faces.extend([[i, j, j + 6], [i, j + 6, i + 6]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class MeshAdvisoryTests(unittest.TestCase):
    def assert_all_boundary_edges_walked(self, boundary):
        edges = Counter(map(tuple, boundary['directed_edges_vertex_ids']))
        walked = Counter()
        for walk in boundary['loops'] + boundary['open_chains']:
            path = walk['vertex_ids']
            walked.update(zip(path[:-1], path[1:]))
            self.assertEqual(len(walk['directed_edge_indices']), len(path) - 1)
        self.assertEqual(walked, edges)

    def test_box_with_missing_panel_has_source_directed_hole_loop(self):
        box = trimesh.creation.box(extents=[4, 6, 2])
        mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2], process=False)
        original = (mesh.vertices.copy(), mesh.faces.copy())
        with no_repairs_or_certification():
            report = mesh_advisories(mesh)
        boundary = report['boundary']
        self.assertTrue(report['has_boundary_edges'])
        self.assertEqual(boundary['boundary_edge_count'], 4)
        self.assertEqual(boundary['boundary_vertex_count'], 4)
        self.assertEqual(len(boundary['loops']), 1)
        self.assertEqual(boundary['open_chains'], [])
        self.assertEqual(boundary['branch_vertex_ids'], [])
        loop = boundary['loops'][0]
        self.assertEqual(loop['vertex_ids'][0], loop['vertex_ids'][-1])
        self.assertTrue(loop['is_closed'])
        self.assertFalse(boundary['is_cad_trim'])
        directed = {tuple(edge) for face in mesh.faces for edge in
                    ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))}
        self.assertTrue(set(map(tuple, boundary['directed_edges_vertex_ids'])) <= directed)
        self.assert_all_boundary_edges_walked(boundary)
        np.testing.assert_array_equal(mesh.vertices, original[0])
        np.testing.assert_array_equal(mesh.faces, original[1])
        self.assertFalse(report['mesh_repair_performed'])
        self.assertFalse(report['normal_repair_performed'])
        self.assertFalse(report['self_intersection_checked'])
        self.assertFalse(report['solid_validity_certified'])

    def test_unclosed_directed_chains_and_branch_ids_are_not_reversed(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                               faces=[[0, 1, 2], [0, 3, 2]], process=False)
        boundary = mesh_advisories(mesh)['boundary']
        self.assertEqual(boundary['loops'], [])
        self.assertEqual([c['vertex_ids'] for c in boundary['open_chains']],
                         [[0, 1, 2], [0, 3, 2]])
        self.assertEqual(boundary['branch_vertex_ids'], [0, 2])
        self.assertEqual(boundary['open_vertex_ids'], [0, 2])
        self.assertEqual(boundary['topology_status'], 'branched')
        self.assert_all_boundary_edges_walked(boundary)

    def test_nondirected_branch_is_not_chosen_as_arbitrary_closed_loop(self):
        # Two triangles meet only at vertex 0: preserve both cycles, expose branch.
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                        [-1, 0, 0], [0, -1, 0]],
                               faces=[[0, 1, 2], [0, 3, 4]], process=False)
        boundary = mesh_advisories(mesh)['boundary']
        self.assertEqual(boundary['branch_vertex_ids'], [0])
        self.assertEqual(len(boundary['loops']), 2)
        self.assertTrue(all(loop['touches_branch'] for loop in boundary['loops']))
        self.assertEqual(boundary['topology_status'], 'branched')
        self.assert_all_boundary_edges_walked(boundary)

    def test_nonmanifold_open_chains_keep_all_uses_and_ids(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [.5, 1, 0],
                                        [.5, -1, 0], [.5, 0, 1]],
                               faces=[[0, 1, 2], [1, 0, 3], [0, 1, 4]], process=False)
        report = mesh_advisories(mesh)
        self.assertEqual(report['nonmanifold_edge_count'], 1)
        self.assertEqual(report['boundary']['boundary_edge_count'], 6)
        self.assertTrue(report['boundary']['open_chains'])
        self.assert_all_boundary_edges_walked(report['boundary'])

    def test_near_seam_is_only_distance_evidence_and_never_a_weld(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0],
                    [0, 0, .02], [1, 0, .02], [0, 1, .02]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [3, 5, 4]], process=False)
        original = (mesh.vertices.copy(), mesh.faces.copy())
        with no_repairs_or_certification(), patch('mesh_diagnostics.cKDTree', wraps=cKDTree) as tree:
            report = mesh_advisories(mesh, tolerance=.021)
        tree.assert_called_once()
        pairs = report['near_boundary_vertex_pairs']
        self.assertEqual([p['vertex_ids'] for p in pairs], [[0, 3], [1, 4], [2, 5]])
        self.assertTrue(report['has_near_boundary_vertex_pairs'])
        for pair in pairs:
            self.assertAlmostEqual(pair['distance_mm'], .02)
            self.assertEqual(pair['evidence'], 'distance_only')
            self.assertEqual(pair['weld_requirement'], 'not_determined')
        self.assertTrue(report['near_boundary_vertex_pair_search']['complete_for_all_boundary_vertices'])
        self.assertFalse(report['welding_performed'])
        self.assertFalse(report['normal_repair_performed'])
        np.testing.assert_array_equal(mesh.vertices, original[0])
        np.testing.assert_array_equal(mesh.faces, original[1])

    def test_excludes_all_source_edges_including_interior_diagonal(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [.01, 0, 0],
                                        [.01, .01, 0], [0, .01, 0]],
                               faces=[[0, 1, 2], [0, 2, 3]], process=False)
        report = mesh_advisories(mesh, tolerance=.05)
        self.assertEqual([p['vertex_ids'] for p in report['near_boundary_vertex_pairs']], [[1, 3]])
        self.assertNotIn([0, 2], [p['vertex_ids'] for p in report['near_boundary_vertex_pairs']])

    def test_proximity_matches_bruteforce_nonadjacent_boundary_pairs(self):
        rng = np.random.default_rng(51)
        mesh = trimesh.Trimesh(vertices=rng.uniform(0, 1, (12, 3)),
                               faces=np.arange(12).reshape((-1, 3)), process=False)
        report = mesh_advisories(mesh, tolerance=.7)
        adjacent = {tuple(sorted(edge)) for face in mesh.faces for edge in
                    ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))}
        expected = {(i, j) for i, j in combinations(range(12), 2)
                    if (i, j) not in adjacent and math.dist(mesh.vertices[i], mesh.vertices[j]) <= .7}
        actual = {tuple(pair['vertex_ids']) for pair in report['near_boundary_vertex_pairs']}
        self.assertEqual(actual, expected)
        self.assertTrue(report['near_boundary_vertex_pair_search']['complete_for_all_boundary_vertices'])

    def test_radius_is_euclidean_inclusive_and_zero_does_not_merge_ids(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [10, 0, 0], [0, 10, 0],
                                        [.03, .04, 0], [20, 0, 0], [0, 20, 0]],
                               faces=[[0, 1, 2], [3, 4, 5]], process=False)
        self.assertEqual(mesh_advisories(mesh, .05)['near_boundary_vertex_pairs'][0]['vertex_ids'], [0, 3])
        self.assertFalse(mesh_advisories(mesh, .0499)['has_near_boundary_vertex_pairs'])
        mesh.vertices[3] = mesh.vertices[0]
        result = mesh_advisories(mesh, tolerance=0)
        self.assertEqual(result['near_boundary_vertex_pairs'][0]['distance_mm'], 0)
        self.assertEqual(len(mesh.vertices), 6)

    def test_no_boundary_edges_is_false_without_tree_even_with_near_unused_points(self):
        box = trimesh.creation.box()
        box.vertices = np.vstack([box.vertices, [[0, 0, 0], [0, 0, .001]]])
        empty = SimpleNamespace(vertices=np.array([[0., 0, 0], [0, 0, .001]]),
                                faces=np.empty((0, 3), dtype=int))
        for mesh in (box, empty):
            with self.subTest(mesh=type(mesh).__name__), patch('mesh_diagnostics.cKDTree') as tree:
                report = mesh_advisories(mesh)
            self.assertFalse(report['has_boundary_edges'])
            self.assertFalse(report['has_near_boundary_vertex_pairs'])
            self.assertFalse(report['near_boundary_vertex_pair_search']['search_performed'])
            self.assertEqual(report['boundary']['loops'], [])
            tree.assert_not_called()

    def test_pair_limit_100_does_not_materialize_an_unbounded_pair_array(self):
        mesh = disconnected_triangles(60)
        with patch('mesh_diagnostics.cKDTree', wraps=cKDTree) as tree:
            report = mesh_advisories(mesh, tolerance=10)
        tree.assert_called_once()
        self.assertEqual(len(report['near_boundary_vertex_pairs']), 100)
        search = report['near_boundary_vertex_pair_search']
        self.assertEqual(search['pair_limit'], 100)
        self.assertTrue(search['pairs_truncated'])
        self.assertTrue(search['pair_limit_reached'])
        self.assertFalse(search['complete_for_sampled_vertices'])
        self.assertIsNone(search['total_eligible_near_pair_count'])
        self.assertEqual(search['nonadjacent_distance_test_count'], 101)
        self.assertLess(search['queried_vertex_count'], len(mesh.vertices))

    def test_boundary_vertex_cap_is_explicit_but_topology_is_not_sampled(self):
        mesh = disconnected_triangles(40)
        report = mesh_advisories(mesh, tolerance=10, max_boundary_vertices=9)
        search = report['near_boundary_vertex_pair_search']
        self.assertEqual(search['total_boundary_vertex_count'], 120)
        self.assertEqual(search['sampled_boundary_vertex_count'], 9)
        self.assertEqual(search['omitted_boundary_vertex_count'], 111)
        self.assertAlmostEqual(search['vertex_sample_coverage'], 9 / 120)
        self.assertTrue(search['vertex_sampling_limited'])
        self.assertFalse(search['complete_for_all_boundary_vertices'])
        self.assertTrue(search['complete_for_sampled_vertices'])
        sampled = search['sampled_boundary_vertex_ids']
        self.assertEqual(len(set(sampled)), 9)
        self.assertEqual(sampled[0], 0)
        self.assertEqual(sampled[-1], 119)
        self.assertTrue(all(set(p['vertex_ids']) <= set(sampled)
                            for p in report['near_boundary_vertex_pairs']))
        self.assertEqual(report['boundary']['boundary_edge_count'], 120)
        self.assertEqual(len(report['boundary']['loops']), 40)
        self.assertTrue(report['boundary']['topology_complete'])
        self.assert_all_boundary_edges_walked(report['boundary'])

    def test_default_boundary_vertex_budget_is_5000(self):
        mesh = disconnected_triangles(1668)
        report = mesh_advisories(mesh, tolerance=10)
        search = report['near_boundary_vertex_pair_search']
        self.assertEqual(search['max_boundary_vertices'], 5000)
        self.assertEqual(search['total_boundary_vertex_count'], 5004)
        self.assertEqual(search['sampled_boundary_vertex_count'], 5000)
        self.assertEqual(search['omitted_boundary_vertex_count'], 4)
        self.assertFalse(search['complete_for_all_boundary_vertices'])
        self.assertEqual(len(report['near_boundary_vertex_pairs']), 100)
        self.assertEqual(len(report['boundary']['loops']), 1668)

    def test_no_hits_in_small_sample_does_not_claim_global_absence(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [10, 0, 0], [0, 10, 0],
                                        [0, 0, .01], [20, 0, 0], [0, 20, 0]],
                               faces=[[0, 1, 2], [3, 4, 5]], process=False)
        self.assertTrue(mesh_advisories(mesh)['has_near_boundary_vertex_pairs'])
        for cap in (0, 1, 2):
            with self.subTest(cap=cap):
                report = mesh_advisories(mesh, max_boundary_vertices=cap)
                search = report['near_boundary_vertex_pair_search']
                self.assertFalse(report['has_near_boundary_vertex_pairs'])
                self.assertFalse(search['complete_for_all_boundary_vertices'])
                self.assertTrue(search['absence_in_unexamined_vertices_is_not_proven'])
                self.assertIsNone(search['total_eligible_near_pair_count'])
                self.assertEqual(report['boundary']['boundary_edge_count'], 6)

    def test_large_coordinates_do_not_overflow_tree_or_change_local_distances(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                        [0, 0, .02], [1, 0, .02], [0, 1, .02]],
                               faces=[[0, 1, 2], [3, 5, 4]], process=False)
        mesh.apply_translation([1e12, -1e12, 1e12])
        report = mesh_advisories(mesh)
        self.assertEqual([p['vertex_ids'] for p in report['near_boundary_vertex_pairs']],
                         [[0, 3], [1, 4], [2, 5]])
        extreme = SimpleNamespace(vertices=np.array([[-1e308, 0., 0.], [0, 1, 0], [0, 0, 1],
                                                     [1e308, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                  faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int))
        result = mesh_advisories(extreme)
        self.assertEqual([p['vertex_ids'] for p in result['near_boundary_vertex_pairs']],
                         [[1, 4], [2, 5]])
        json.dumps(result, allow_nan=False)

    def test_rejects_nonfinite_boolean_or_nonnumeric_tolerance(self):
        mesh = trimesh.creation.box()
        for bad in (True, False, np.bool_(False), np.nan, np.inf, -np.inf,
                    -.001, '0.05', None, .05 + 0j, np.array([.05]), 10 ** 1000):
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                mesh_advisories(mesh, tolerance=bad)
        for good in (0, .05, np.float64(.05), np.float32(.05), np.int64(1)):
            with self.subTest(value=good):
                mesh_advisories(mesh, tolerance=good)

    def test_rejects_noninteger_or_boolean_vertex_budget(self):
        mesh = trimesh.creation.box()
        for bad in (True, np.bool_(False), -1, 1.5, 2., np.nan, np.inf, '4', None):
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                mesh_advisories(mesh, max_boundary_vertices=bad)
        mesh_advisories(mesh, max_boundary_vertices=np.int64(3))

    def test_rejects_bad_arrays_and_unused_nonfinite_coordinates(self):
        cases = [
            SimpleNamespace(vertices=np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [np.inf, 0, 0]]),
                            faces=np.array([[0, 1, 2]], dtype=int)),
            SimpleNamespace(vertices=np.array([[np.nan, 0, 0], [1, 0, 0], [0, 1, 0]]),
                            faces=np.array([[0, 1, 2]], dtype=int)),
            SimpleNamespace(vertices=np.ones((3, 3)), faces=np.array([[0., 1., 2.]])),
            SimpleNamespace(vertices=np.ones((3, 3)), faces=np.array([[0, 1, 3]])),
            SimpleNamespace(vertices=np.ones((3, 3)), faces=np.array([[0, -1, 2]])),
            SimpleNamespace(vertices=np.ones((3, 3), dtype=bool), faces=np.array([[0, 1, 2]])),
            SimpleNamespace(vertices=np.ones((3, 3), dtype=complex), faces=np.array([[0, 1, 2]])),
            SimpleNamespace(vertices=np.ones((3, 3)), faces=np.array([[False, True, True]])),
            SimpleNamespace(vertices=np.ones((3, 2)), faces=np.array([[0, 1, 2]])),
            SimpleNamespace(vertices=np.ones((3, 3)), faces=np.array([[0, 1, 2, 0]])),
        ]
        for mesh in cases:
            with self.subTest(mesh=mesh), self.assertRaises(ValueError):
                mesh_advisories(mesh)


class ShellRelationshipTests(unittest.TestCase):
    def assert_uncertified(self, report):
        self.assertFalse(report['containment_certified'])
        self.assertFalse(report['self_intersection_checked'])
        self.assertFalse(report['actual_intersection_checked'])
        self.assertFalse(report['ray_tests_performed'])
        self.assertFalse(report['mesh_repair_performed'])
        self.assertFalse(report['normal_repair_performed'])
        for pair in report['pairs']:
            self.assertFalse(pair['containment_certified'])
            self.assertFalse(pair['actual_intersection_checked'])
        for shell in report['shells']:
            self.assertFalse(shell['containment_certified'])
            self.assertFalse(shell['self_intersection_checked'])

    def test_nested_negative_meshbody_is_uncertain_cavity_or_inversion(self):
        outer = trimesh.creation.box(extents=[10, 12, 14])
        inner = inward_box((2, 4, 6))
        partitions = split_bodies(trimesh.util.concatenate([outer, inner]))
        originals = [(part.mesh.vertices.copy(), part.mesh.faces.copy(),
                      part.instance_triangle_ids.copy(), part.instance_vertex_ids.copy())
                     for part in partitions]
        with no_repairs_or_certification():
            report = shell_relationships(partitions)
        self.assertEqual([s['index'] for s in report['shells']], [0, 1])
        first, second = report['shells']
        self.assertAlmostEqual(first['signed_volume_mm3'], 1680)
        self.assertAlmostEqual(second['signed_volume_mm3'], -48)
        self.assertEqual(first['role_candidate'], 'outer_shell_candidate')
        self.assertEqual(second['role_candidate'], 'possible_cavity_or_inverted_nested_shell')
        self.assertEqual(second['container_candidates'], [0])
        self.assertEqual(second['positive_bounded_container_candidates'], [0])
        self.assertEqual(report['pairs'][0]['bounds_relation'], 'strict_aabb_containment_candidate')
        self.assertEqual(report['pairs'][0]['container_index'], 0)
        self.assertEqual(report['pairs'][0]['contained_index'], 1)
        for part, original in zip(partitions, originals):
            np.testing.assert_array_equal(part.mesh.vertices, original[0])
            np.testing.assert_array_equal(part.mesh.faces, original[1])
            np.testing.assert_array_equal(part.instance_triangle_ids, original[2])
            np.testing.assert_array_equal(part.instance_vertex_ids, original[3])
        self.assert_uncertified(report)
        json.dumps(report, allow_nan=False)

    def test_standalone_inward_role_is_orientation_review_not_fix_normals(self):
        mesh = inward_box((2, 3, 4))
        original = mesh.faces.copy()
        with no_repairs_or_certification():
            report = shell_relationships([mesh])
        shell = report['shells'][0]
        self.assertAlmostEqual(shell['signed_volume_mm3'], -24)
        self.assertEqual(shell['role_candidate'], 'orientation_review')
        self.assertEqual(shell['container_candidates'], [])
        self.assertEqual(report['pairs'], [])
        np.testing.assert_array_equal(mesh.faces, original)
        self.assert_uncertified(report)

    def test_only_positive_closed_locally_bounded_candidate_enables_cavity_role(self):
        inner = inward_box()
        positive = trimesh.creation.box(extents=[10, 10, 10])
        open_outer = trimesh.Trimesh(vertices=positive.vertices, faces=positive.faces[:-1], process=False)
        inward_outer = inward_box((10, 10, 10))
        for outer in (open_outer, inward_outer):
            with self.subTest(outer=outer):
                report = shell_relationships([outer, inner])
                shell = report['shells'][1]
                self.assertEqual(shell['container_candidates'], [0])
                self.assertEqual(shell['positive_bounded_container_candidates'], [])
                self.assertEqual(shell['role_candidate'], 'orientation_review')
        # More than one bounding candidate remains a list, not a chosen parent.
        report = shell_relationships([positive, trimesh.creation.box(extents=[4, 4, 4]), inner])
        self.assertEqual(report['shells'][2]['container_candidates'], [0, 1])
        self.assertEqual(report['shells'][2]['positive_bounded_container_candidates'], [0, 1])

    def test_strict_containment_does_not_include_equal_or_touching_bounds(self):
        outer = trimesh.creation.box(extents=[10, 10, 10])
        inner = inward_box((2, 2, 2))
        inner.apply_translation([4, 0, 0])  # shares max-x boundary with outer
        report = shell_relationships([outer, inner, outer.copy()])
        self.assertTrue(all(p['bounds_relation'] == 'overlapping_bounds_candidate'
                            for p in report['pairs']))
        self.assertEqual(report['shells'][1]['container_candidates'], [])
        self.assertEqual(report['shells'][1]['role_candidate'], 'orientation_review')
        self.assert_uncertified(report)

    def test_disjoint_multi_shells_are_separate_not_merged(self):
        meshes = [trimesh.creation.box() for _ in range(4)]
        for index, mesh in enumerate(meshes):
            mesh.apply_translation([index * 4, 0, 0])
        report = shell_relationships(meshes)
        self.assertEqual(len(report['pairs']), 6)
        self.assertTrue(all(p['bounds_relation'] == 'disjoint_bounds' for p in report['pairs']))
        self.assertTrue(all(s['role_candidate'] == 'separate_shell' for s in report['shells']))
        self.assertTrue(all(s['container_candidates'] == [] for s in report['shells']))
        self.assertTrue(report['pair_coverage']['complete'])

    def test_overlapping_and_touching_bounds_are_not_actual_intersection(self):
        first = trimesh.creation.box(extents=[2, 2, 2])
        overlap = first.copy()
        overlap.apply_translation([1, 1, 0])
        touching = first.copy()
        touching.apply_translation([-2, 0, 0])
        report = shell_relationships([first, overlap, touching])
        self.assertEqual(report['pairs'][0]['bounds_relation'], 'overlapping_bounds_candidate')
        self.assertEqual(report['pairs'][1]['bounds_relation'], 'touching_bounds_candidate')
        self.assert_uncertified(report)
        self.assertNotIn('intersecting_shells', json.dumps(report))

    def test_concave_bounds_mislead_and_must_not_certify_containment(self):
        outer = l_prism()
        inner = inward_box((.5, .5, .5))
        inner.apply_translation([2, 2, 1])
        # All vertices of the little box are in the L's missing quadrant.
        self.assertTrue(np.all(inner.vertices[:, 0] > 1))
        self.assertTrue(np.all(inner.vertices[:, 1] > 1))
        self.assertTrue(np.all(inner.vertices[:, :2] < 3))
        self.assertEqual(len(split_bodies(outer)), 1)
        with no_repairs_or_certification():
            report = shell_relationships([outer, inner])
        self.assertAlmostEqual(report['shells'][0]['signed_volume_mm3'], 10)
        self.assertEqual(report['pairs'][0]['bounds_relation'], 'strict_aabb_containment_candidate')
        self.assertEqual(report['shells'][1]['container_candidates'], [0])
        self.assertEqual(report['shells'][1]['role_candidate'],
                         'possible_cavity_or_inverted_nested_shell')
        self.assert_uncertified(report)

    def test_nested_positive_shell_is_not_interpreted_as_cavity(self):
        outer = trimesh.creation.box(extents=[8, 8, 8])
        inner = trimesh.creation.box()
        report = shell_relationships([inner, outer])
        self.assertEqual(report['pairs'][0]['container_index'], 1)
        self.assertEqual(report['shells'][0]['container_candidates'], [1])
        self.assertEqual(report['shells'][0]['role_candidate'], 'nested_shell_review')
        self.assert_uncertified(report)

    def test_huge_sibling_bounds_and_translation_do_not_break_local_quality(self):
        outer = trimesh.creation.box(extents=[8, 12, 16])
        inner = inward_box((2, 4, 6))
        baseline = shell_relationships([outer, inner])
        for mesh in (outer, inner):
            mesh.apply_translation([1e12, -1e12, 1e12])
        huge = trimesh.creation.box(extents=[1e14, 1e14, 1e14])
        huge.apply_translation([1e16, 0, 0])
        tiny = trimesh.creation.box(extents=[1e-4, 2e-4, 3e-4])
        report = shell_relationships([outer, inner, huge, tiny])
        for index in (0, 1):
            self.assertEqual(report['shells'][index]['local_quality'], baseline['shells'][index]['local_quality'])
            self.assertEqual(report['shells'][index]['signed_volume_mm3'],
                             baseline['shells'][index]['signed_volume_mm3'])
            self.assertEqual(report['shells'][index]['role_candidate'], baseline['shells'][index]['role_candidate'])
        self.assertEqual(report['shells'][3]['local_quality']['degenerate_triangle_count'], 0)
        self.assertGreater(report['shells'][3]['signed_volume_mm3'], 0)
        self.assertEqual(report['shells'][3]['role_candidate'], 'separate_shell')
        tiny_inner = inward_box((1e-4, 2e-4, 3e-4))
        huge_outer = trimesh.creation.box(extents=[1e14, 1e14, 1e14])
        nested = shell_relationships([huge_outer, tiny_inner])
        self.assertEqual(nested['shells'][1]['role_candidate'],
                         'possible_cavity_or_inverted_nested_shell')
        self.assertAlmostEqual(nested['shells'][1]['signed_volume_mm3'] / -6e-12, 1.)
        self.assertEqual(nested['shells'][1]['local_quality']['degenerate_triangle_count'], 0)
        json.dumps(report, allow_nan=False)

    def test_unused_far_points_do_not_expand_bounds_or_distort_local_volume(self):
        first = trimesh.creation.box(extents=[2, 4, 6])
        first.vertices = np.vstack([first.vertices, [[1e150, -1e150, 1e150]]])
        second = trimesh.creation.box()
        second.apply_translation([10, 0, 0])
        report = shell_relationships([first, second])
        self.assertAlmostEqual(report['shells'][0]['signed_volume_mm3'], 48)
        self.assertEqual(report['shells'][0]['bounds_mm'], [[-1., -2., -3.], [1., 2., 3.]])
        self.assertEqual(report['pairs'][0]['bounds_relation'], 'disjoint_bounds')

    def test_pair_cap_reports_coverage_and_never_calls_unchecked_shells_separate(self):
        meshes = [trimesh.creation.box() for _ in range(6)]
        for i, mesh in enumerate(meshes):
            mesh.apply_translation([3 * i, 0, 0])
        report = shell_relationships(meshes, max_pairs=2)
        self.assertEqual([p['indices'] for p in report['pairs']], [[0, 1], [0, 2]])
        coverage = report['pair_coverage']
        self.assertEqual(coverage['total_possible_pairs'], 15)
        self.assertEqual(coverage['evaluated_pair_count'], 2)
        self.assertEqual(coverage['omitted_pair_count'], 13)
        self.assertAlmostEqual(coverage['coverage_fraction'], 2 / 15)
        self.assertFalse(coverage['complete'])
        self.assertTrue(coverage['truncated'])
        self.assertTrue(all(s['role_candidate'] == 'relationship_review' for s in report['shells']))
        self.assertTrue(all(s['unexamined_shell_count'] > 0 for s in report['shells']))
        self.assertEqual(report['shells'][-1]['checked_shell_count'], 0)

    def test_default_pair_limit_128_zero_budget_and_limited_inner_evidence(self):
        meshes = [trimesh.creation.box() for _ in range(20)]
        default = shell_relationships(meshes)
        self.assertEqual(len(default['pairs']), 128)
        self.assertEqual(default['pair_coverage']['total_possible_pairs'], 190)
        self.assertFalse(default['pair_coverage']['complete'])
        outer = trimesh.creation.box(extents=[10, 10, 10])
        inner = inward_box()
        unchecked = shell_relationships([inner, outer], max_pairs=0)
        self.assertEqual(unchecked['pairs'], [])
        self.assertEqual(unchecked['shells'][0]['container_candidates'], [])
        self.assertEqual(unchecked['shells'][0]['role_candidate'], 'orientation_review')
        self.assertFalse(unchecked['shells'][0]['relationships_complete'])
        partial = shell_relationships([outer, inner, trimesh.creation.box()], max_pairs=1)
        self.assertEqual(partial['shells'][1]['role_candidate'],
                         'possible_cavity_or_inverted_nested_shell')
        self.assertFalse(partial['shells'][1]['relationships_complete'])

    def test_empty_mixed_meshbody_and_generator_inputs_are_supported(self):
        empty = shell_relationships([])
        self.assertEqual(empty['shells'], [])
        self.assertEqual(empty['pairs'], [])
        self.assertTrue(empty['pair_coverage']['complete'])
        mesh = trimesh.creation.box()
        body = MeshBody(mesh, np.arange(12), np.arange(8))
        report = shell_relationships(item for item in [body, mesh.copy()])
        self.assertEqual(len(report['shells']), 2)
        self.assertEqual(report['pairs'][0]['bounds_relation'], 'overlapping_bounds_candidate')
        complete = shell_relationships([mesh, mesh.copy()], max_pairs=10 ** 100)
        self.assertEqual(len(complete['pairs']), 1)
        self.assertTrue(complete['pair_coverage']['complete'])

    def test_pair_budget_and_invalid_shells_are_rejected(self):
        mesh = trimesh.creation.box()
        for bad in (True, np.bool_(False), -1, 1.5, 2., np.inf, np.nan, '4', None):
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                shell_relationships([mesh], max_pairs=bad)
        shell_relationships([mesh], max_pairs=np.int64(1))
        for bad in (None, mesh, MeshBody(mesh, np.arange(12), np.arange(8)), [None],
                    [SimpleNamespace(vertices=np.ones((3, 3)), faces=np.array([[0, 1, 3]]))],
                    [SimpleNamespace(vertices=np.ones((3, 3)), faces=np.empty((0, 3), dtype=int))]):
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                shell_relationships(bad)
        mesh.vertices = np.vstack([mesh.vertices, [[0, np.inf, 0]]])
        with self.assertRaises(ValueError):
            shell_relationships([mesh])


if __name__ == '__main__':
    unittest.main()
