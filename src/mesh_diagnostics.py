"""Read-only mesh advisories; none of these candidates authorize a repair.

Lengths/volumes follow the project's millimeter convention. Vertex IDs address
input-mesh arrays, not welded coordinates. Boundary topology is exhaustive;
only the proximity search is vertex-sampled and capped at 100 reported pairs.
Shell relationships use strictly nested AABBs, never rays or triangle tests.
An inward nested shell might describe a cavity OR an inverted separate shell:
its normals must not be automatically reversed on this evidence.
"""
from collections import defaultdict
from itertools import chain, combinations, islice
import math
from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree

from mesh_evidence import MeshBody, mesh_quality


_NEAR_PAIR_LIMIT = 100


def _limit(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer, not a boolean')
    return int(value)


def _tolerance(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('tolerance must be finite and nonnegative, not a boolean')
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError('tolerance must be finite and nonnegative') from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    return result


def _arrays(mesh):
    """Validate without invoking mesh processing or silently casting indices."""
    try:
        source_vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('Expected mesh vertices and triangle faces') from exc
    if (source_vertices.ndim != 2 or source_vertices.shape[1:] != (3,)
            or faces.ndim != 2 or faces.shape[1:] != (3,)):
        raise ValueError('Expected (n, 3) vertices and (m, 3) triangle faces')
    if source_vertices.dtype.kind not in 'iuf':
        raise ValueError('Mesh coordinates must be real finite numbers')
    with np.errstate(over='ignore', invalid='ignore'):
        vertices = np.asarray(source_vertices, dtype=float)
    if not np.isfinite(vertices).all():
        raise ValueError('Mesh coordinates must be finite, including unused vertices')
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('Triangle indices must be integers')
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Triangle index outside vertex array')
    return vertices, faces


def _edge_data(faces):
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2))
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse] == 1]
    return edges, boundary, counts


def _boundary_walks(boundary):
    """Consume each directed edge once, stopping rather than choosing at branches."""
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for edge_id, (start, end) in enumerate(boundary):
        outgoing[int(start)].append(edge_id)
        incoming[int(end)].append(edge_id)
    vertex_ids = sorted(set(incoming) | set(outgoing))
    branches = [v for v in vertex_ids if len(incoming[v]) > 1 or len(outgoing[v]) > 1]
    branch_set = set(branches)
    open_vertices = [v for v in vertex_ids if len(incoming[v]) != len(outgoing[v])]
    regular = {v for v in vertex_ids if len(incoming[v]) == len(outgoing[v]) == 1}
    starts = [i for i, (start, _) in enumerate(boundary) if int(start) not in regular]
    used = np.zeros(len(boundary), dtype=bool)
    loops, open_chains = [], []
    for seed in chain(starts, range(len(boundary))):
        if used[seed]:
            continue
        path, edge_ids, current = [int(boundary[seed, 0])], [], seed
        while not used[current]:
            used[current] = True
            edge_ids.append(current)
            end = int(boundary[current, 1])
            path.append(end)
            if end == path[0] or end not in regular:
                break
            current = outgoing[end][0]
        closed = path[0] == path[-1]
        record = {'vertex_ids': path, 'directed_edge_indices': edge_ids,
                  'is_closed': closed, 'touches_branch': bool(branch_set.intersection(path))}
        (loops if closed else open_chains).append(record)
    status = ('none' if not len(boundary) else 'branched' if branches else
              'open' if open_chains else 'closed_directed_loops')
    return {'directed_edges_vertex_ids': boundary.tolist(),
            'boundary_edge_count': len(boundary), 'boundary_vertex_count': len(vertex_ids),
            'loops': loops, 'open_chains': open_chains,
            'branch_vertex_ids': branches, 'open_vertex_ids': open_vertices,
            'topology_status': status, 'topology_complete': True,
            'walk_policy': 'source_winding_stop_at_branches_no_reversal_or_gap_closure',
            'is_cad_trim': False, 'self_intersection_checked': False}


def _near_pairs(vertices, edges, boundary, tolerance, cap):
    ids = np.unique(boundary)
    count = len(ids)
    if count > cap:
        positions = np.linspace(0, count - 1, cap, dtype=np.int64)
        sample = ids[positions]
    else:
        sample = ids
    sampled = len(sample)
    records = []
    truncated = False
    queried_vertices = distance_tests = 0
    if sampled >= 2:
        # Only source-nonadjacent endpoints qualify. Even an interior edge
        # joining two boundary vertices excludes that pair from this advisory.
        selected = np.isin(edges, sample).all(axis=1)
        adjacent = set(map(tuple, edges[selected].tolist()))
        points = vertices[sample]
        # Bounded coordinates prevent cKDTree's squared-distance overflow.
        # The small roundoff margin is ONLY a broad-phase allowance; every
        # reported Euclidean distance is checked on original coordinates.
        with np.errstate(over='ignore', invalid='ignore'):
            offsets = points - points[0]
        local = offsets if np.isfinite(offsets).all() else points
        scale = max(float(np.abs(local).max()), tolerance) or 1.
        conditioned = local / scale
        radius = tolerance / scale + 16 * np.finfo(float).eps
        tree = cKDTree(conditioned)
        for i, point in enumerate(conditioned):
            queried_vertices += 1
            for j in tree.query_ball_point(point, radius, return_sorted=True):
                if j <= i:
                    continue
                a, b = int(sample[i]), int(sample[j])
                if (a, b) in adjacent:
                    continue
                distance_tests += 1
                distance = math.dist(points[i], points[j])
                if distance > tolerance:
                    continue
                if len(records) == _NEAR_PAIR_LIMIT:
                    truncated = True
                    break
                records.append({'vertex_ids': [a, b], 'distance_mm': distance,
                                'evidence': 'distance_only',
                                'weld_requirement': 'not_determined'})
            if truncated:
                break
    coverage = {
        'method': 'cKDTree_radius_prefilter_then_original_euclidean_distance',
        'eligibility': 'boundary_vertices_not_joined_by_any_source_mesh_edge',
        'tolerance_mm': tolerance, 'max_boundary_vertices': cap,
        'total_boundary_vertex_count': count, 'sampled_boundary_vertex_count': sampled,
        'omitted_boundary_vertex_count': count - sampled,
        'sampled_boundary_vertex_ids': sample.tolist(),
        'vertex_sample_coverage': sampled / count if count else 1.,
        'vertex_sampling_limited': sampled < count,
        'vertex_sampling_method': ('evenly_spaced_sorted_source_ids' if sampled < count else
                                   'all_boundary_vertices'),
        'search_performed': sampled >= 2,
        'queried_vertex_count': queried_vertices, 'nonadjacent_distance_test_count': distance_tests,
        'pair_limit': _NEAR_PAIR_LIMIT, 'reported_pair_count': len(records),
        'pair_limit_reached': len(records) == _NEAR_PAIR_LIMIT,
        'pairs_truncated': truncated,
        'complete_for_sampled_vertices': not truncated,
        'complete_for_all_boundary_vertices': sampled == count and not truncated,
        'total_eligible_near_pair_count': None if truncated or sampled < count else len(records),
        'absence_in_unexamined_vertices_is_not_proven': sampled < count or truncated,
    }
    return records, coverage


def mesh_advisories(mesh, tolerance=.05, max_boundary_vertices=5000) -> dict:
    """Return directed boundaries and distance-only near-seam candidates.

    ``boundary`` holds closed directed walks (repeated closing vertex), open
    chains, and branch/open vertex IDs. No walk jumps a gap or turns arbitrarily
    at a branch. Its topology is not sampled. ``near_boundary_vertex_pairs``
    contains at most 100 non-edge-adjacent vertex pairs at distance <= tolerance.
    ``near_boundary_vertex_pair_search`` explicitly describes sampling and early
    termination; an empty sampled result need not establish global absence.

    Zero tolerance tests coincident, distinct source IDs without welding them.
    A zero vertex budget disables proximity search, not boundary reporting.
    Empty triangle arrays are allowed; with no boundary edges both the near-pair
    flag and the search-performed flag are false. All inputs remain unchanged.
    """
    tolerance = _tolerance(tolerance)
    cap = _limit(max_boundary_vertices, 'max_boundary_vertices')
    vertices, faces = _arrays(mesh)
    edges, boundary, counts = _edge_data(faces)
    pairs, coverage = _near_pairs(vertices, edges, boundary, tolerance, cap)
    return {
        'vertex_id_scope': 'input_mesh_vertices',
        'has_boundary_edges': bool(len(boundary)),
        'boundary': _boundary_walks(boundary),
        'nonmanifold_edge_count': int(np.count_nonzero(counts > 2)),
        'near_boundary_vertex_pairs': pairs,
        'has_near_boundary_vertex_pairs': bool(pairs),
        'near_boundary_vertex_pair_search': coverage,
        'evidence_scope': 'topological_boundaries_and_distance_evidence_not_a_weld_requirement',
        'mesh_repair_performed': False, 'welding_performed': False,
        'normal_repair_performed': False, 'self_intersection_checked': False,
        'solid_validity_certified': False,
    }


def _bounds_relation(first, second):
    if np.any(first[1] < second[0]) or np.any(second[1] < first[0]):
        return 'disjoint_bounds', None
    if np.all(first[0] < second[0]) and np.all(first[1] > second[1]):
        return 'strict_aabb_containment_candidate', 0
    if np.all(second[0] < first[0]) and np.all(second[1] > first[1]):
        return 'strict_aabb_containment_candidate', 1
    if np.any(first[1] == second[0]) or np.any(second[1] == first[0]):
        return 'touching_bounds_candidate', None
    return 'overlapping_bounds_candidate', None


def shell_relationships(partitions, max_pairs=128) -> dict:
    """Compare MeshBody/mesh iterable entries using at most ``max_pairs`` AABBs.

    Shell indices are positions in the input iterable. Bounds use referenced
    vertices only and require strict inequalities in all axes for containment
    *candidates*. Equal/touching bounds do not qualify. Signed volumes and local
    quality come from each shell independently, never a global extent threshold.

    ``container_candidates`` lists tested strictly bounding shell indices, NOT
    certified containers. A negative-volume shell with a positive, closed,
    locally well-formed bounding candidate gets the deliberately uncertain role
    ``possible_cavity_or_inverted_nested_shell``. An inward shell without such
    evidence gets ``orientation_review``, not an instruction to flip normals.
    Unchecked pairs cannot establish independence. Neither rays, actual triangle
    intersections, containment certification nor repairs are performed here.
    """
    budget = _limit(max_pairs, 'max_pairs')
    if isinstance(partitions, MeshBody) or hasattr(partitions, 'faces'):
        raise ValueError('partitions must be an iterable of MeshBody objects or meshes')
    try:
        partitions = list(partitions)
    except TypeError as exc:
        raise ValueError('partitions must be an iterable of MeshBody objects or meshes') from exc
    shells, bounds, positive_bounded = [], [], []
    for index, partition in enumerate(partitions):
        mesh = partition.mesh if isinstance(partition, MeshBody) else partition
        vertices, faces = _arrays(mesh)
        if not len(faces):
            raise ValueError('Each shell must contain at least one triangle')
        points = vertices[np.unique(faces)]
        box = np.array([points.min(axis=0), points.max(axis=0)])
        quality = mesh_quality(mesh)
        volume = quality['signed_volume_mm3']
        bounded = (quality['watertight'] and quality['winding_consistent']
                   and not quality['degenerate_triangle_ids']
                   and not quality['duplicate_triangle_count']
                   and quality['surface_area_mm2'] is not None
                   and quality['surface_area_mm2'] > 0)
        positive_bounded.append(bool(bounded and volume is not None and volume > 0))
        bounds.append(box)
        shells.append({
            'index': index, 'bounds_mm': box.tolist(), 'signed_volume_mm3': volume,
            'role_candidate': None, 'container_candidates': [],
            'positive_bounded_container_candidates': [], 'contained_shell_candidates': [],
            'checked_shell_count': 0, 'bounds_overlap_candidates': [],
            'local_quality': {
                'watertight': quality['watertight'],
                'winding_consistent': quality['winding_consistent'],
                'degenerate_triangle_count': len(quality['degenerate_triangle_ids']),
                'duplicate_triangle_count': quality['duplicate_triangle_count'],
                'boundary_edge_count': quality['boundary_edge_count'],
                'nonmanifold_edge_count': quality['nonmanifold_edge_count'],
                'inconsistent_winding_edge_count': quality['inconsistent_winding_edge_count'],
                'surface_area_mm2': quality['surface_area_mm2'],
            },
            'containment_certified': False, 'self_intersection_checked': False,
        })
    pairs = []
    total = len(shells) * (len(shells) - 1) // 2
    for i, j in islice(combinations(range(len(shells)), 2), min(budget, total)):
        relation, container_side = _bounds_relation(bounds[i], bounds[j])
        pair = {'indices': [i, j], 'bounds_relation': relation,
                'container_index': None, 'contained_index': None,
                'containment_certified': False, 'actual_intersection_checked': False}
        shells[i]['checked_shell_count'] += 1
        shells[j]['checked_shell_count'] += 1
        if container_side is not None:
            outer, inner = (i, j) if container_side == 0 else (j, i)
            pair.update(container_index=outer, contained_index=inner)
            shells[inner]['container_candidates'].append(outer)
            shells[outer]['contained_shell_candidates'].append(inner)
            if positive_bounded[outer]:
                shells[inner]['positive_bounded_container_candidates'].append(outer)
        elif relation != 'disjoint_bounds':
            shells[i]['bounds_overlap_candidates'].append(j)
            shells[j]['bounds_overlap_candidates'].append(i)
        pairs.append(pair)
    for shell in shells:
        volume = shell['signed_volume_mm3']
        shell['unexamined_shell_count'] = len(shells) - 1 - shell['checked_shell_count']
        shell['relationships_complete'] = shell['unexamined_shell_count'] == 0
        if volume is not None and volume < 0:
            role = ('possible_cavity_or_inverted_nested_shell'
                    if shell['positive_bounded_container_candidates'] else 'orientation_review')
        elif not positive_bounded[shell['index']]:
            role = 'orientation_or_closure_review'
        elif shell['container_candidates']:
            role = 'nested_shell_review'
        elif shell['contained_shell_candidates']:
            role = 'outer_shell_candidate'
        elif shell['bounds_overlap_candidates']:
            role = 'bounds_overlap_review'
        elif shell['relationships_complete']:
            role = 'separate_shell'
        else:
            role = 'relationship_review'
        shell['role_candidate'] = role
    checked = len(pairs)
    return {
        'shells': shells, 'pairs': pairs,
        'pair_coverage': {
            'max_pairs': budget, 'total_possible_pairs': total, 'evaluated_pair_count': checked,
            'omitted_pair_count': total - checked, 'complete': checked == total,
            'coverage_fraction': checked / total if total else 1.,
            'pair_limit_reached': checked == budget and total > 0,
            'truncated': checked < total, 'selection': 'lexicographic_input_index_pairs',
        },
        'relationship_scope': 'referenced_vertex_aabbs_only_not_actual_shell_containment_or_intersection',
        'containment_certified': False, 'self_intersection_checked': False,
        'actual_intersection_checked': False, 'ray_tests_performed': False,
        'mesh_repair_performed': False, 'normal_repair_performed': False,
        'solid_validity_certified': False, 'requires_review': True,
    }
