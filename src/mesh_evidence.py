"""Lossless triangle-shell partitioning and explicit mesh quality evidence.

A connected shell is not necessarily a solid body: an inner cavity shell may be
disconnected from the outer shell. No repair, Boolean union or face deletion is
performed. Self-intersection and shell-containment certification remain absent.
"""
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class MeshBody:
    mesh: trimesh.Trimesh
    instance_triangle_ids: np.ndarray
    instance_vertex_ids: np.ndarray


def _arrays(mesh):
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,)
            or faces.ndim != 2 or faces.shape[1:] != (3,)
            or not len(vertices) or not len(faces)):
        raise ValueError('A nonempty triangle mesh is required')
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('Triangle indices must be integers')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Triangle index outside vertex array')
    if not np.isfinite(vertices).all():
        raise ValueError('Mesh coordinates must be finite')
    return vertices, faces


def _edges(faces):
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2))
    unique, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
    return directed, unique, inverse.reshape(-1), counts


def split_bodies(mesh):
    """Partition ALL faces by shared-edge connectivity, without Trimesh.split repair.

    Degenerate, duplicate and isolated faces survive. Mappings address original
    build-instance arrays. Nonmanifold shared edges connect all incident faces.
    """
    vertices, faces = _arrays(mesh)
    _, _, inverse, _ = _edges(faces)
    order = np.argsort(inverse, kind='stable')
    labels = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
    ends = np.r_[starts[1:], len(order)]
    rows, cols = [], []
    for start, end in zip(starts, ends):
        members = np.unique(order[start:end] // 3)
        if len(members) > 1:
            rows.extend([int(members[0])] * (len(members) - 1))
            cols.extend(members[1:].tolist())
    graph = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                       shape=(len(faces), len(faces))).tocsr()
    _, groups = connected_components(graph, directed=False)
    # Group once instead of rescanning the entire face array for every shell.
    # For disconnected triangle soups the old loop was O(faces * shells),
    # before any downstream analysis budget could be applied.
    grouped_faces = np.argsort(groups, kind='stable')
    breaks = np.r_[0, np.flatnonzero(np.diff(groups[grouped_faces])) + 1, len(faces)]
    bodies = []
    for start, end in zip(breaks[:-1], breaks[1:]):
        face_ids = grouped_faces[start:end]
        vertex_ids, local_faces = np.unique(faces[face_ids], return_inverse=True)
        body = trimesh.Trimesh(vertices=vertices[vertex_ids].copy(),
                               faces=local_faces.reshape((-1, 3)), process=False)
        body.metadata['partition'] = 'shared_edge_connected_triangle_shell'
        body.metadata['mesh_repair_performed'] = False
        bodies.append(MeshBody(body, face_ids, vertex_ids))
    return bodies


def mesh_quality(mesh):
    """Quality gates with recentered signed volume; never abs/fix winding.

    These checks are necessary, not sufficient for a valid solid. In particular
    overlapping nonadjacent triangles and nested shells are not certified.
    """
    vertices, faces = _arrays(mesh)
    used = np.unique(faces)
    points = vertices[used]
    anchor = points[0]
    with np.errstate(all='ignore'):
        relative = vertices - anchor
        tri = relative[faces]
        first, second = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        scale = np.maximum(np.max(np.abs(first), axis=1), np.max(np.abs(second), axis=1))
        cross = np.cross(np.divide(first, scale[:, None], out=np.zeros_like(first), where=scale[:, None] > 0),
                         np.divide(second, scale[:, None], out=np.zeros_like(second), where=scale[:, None] > 0))
        length = np.linalg.norm(cross, axis=1)
        area = .5 * length * scale ** 2
    degenerate = np.flatnonzero((scale <= 0) | (length <= 32 * np.finfo(float).eps)
                               | ~np.isfinite(area))
    _, occurrences = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    duplicates = int(np.sum(np.maximum(occurrences - 1, 0)))
    directed, unique_edges, inverse, counts = _edges(faces)
    boundary = np.flatnonzero(counts == 1)
    nonmanifold = np.flatnonzero(counts > 2)
    orientation = np.where(directed[:, 0] < directed[:, 1], 1,
                           np.where(directed[:, 0] > directed[:, 1], -1, 0))
    balances = np.bincount(inverse, weights=orientation, minlength=len(counts))
    inconsistent = np.flatnonzero((counts == 2) & (balances != 0))
    # Topological closure and geometric validity are separate evidence.
    closed = not len(boundary) and not len(nonmanifold)
    winding = not len(inconsistent) and not len(nonmanifold)
    signed = None
    if closed and winding:
        # A common nearby origin keeps the signed sum translation-invariant.
        center = (relative[used].min(axis=0) + relative[used].max(axis=0)) / 2
        centered = tri - center
        with np.errstate(all='ignore'):
            value = float(np.einsum('ij,ij->i', centered[:, 0],
                                    np.cross(centered[:, 1], centered[:, 2])).sum() / 6)
        if np.isfinite(value):
            signed = value
    issues = []
    if len(degenerate):
        issues.append('degenerate_triangles')
    if duplicates:
        issues.append('duplicate_triangles')
    if len(boundary):
        issues.append('open_boundary_edges')
    if len(nonmanifold):
        issues.append('nonmanifold_edges')
    if len(inconsistent):
        issues.append('inconsistent_face_winding')
    if closed and winding and (signed is None or signed <= 0):
        issues.append('nonpositive_or_nonfinite_signed_volume')
    if not np.isfinite(area).all() or not np.isfinite(np.sum(area)):
        issues.append('nonfinite_surface_area')
    return {
        'vertices': len(vertices), 'referenced_vertices': len(used),
        'unused_vertices': len(vertices) - len(used), 'triangles': len(faces),
        'degenerate_triangle_ids': degenerate.tolist(),
        'duplicate_triangle_count': duplicates,
        'boundary_edge_count': len(boundary), 'nonmanifold_edge_count': len(nonmanifold),
        'inconsistent_winding_edge_count': len(inconsistent),
        'boundary_edges_vertex_ids': unique_edges[boundary].tolist(),
        'nonmanifold_edges_vertex_ids': unique_edges[nonmanifold].tolist(),
        'watertight': bool(closed), 'winding_consistent': bool(winding),
        'signed_volume_mm3': signed,
        'surface_area_mm2': float(np.sum(area)) if np.isfinite(area).all() and np.isfinite(np.sum(area)) else None,
        'triangle_id_scope': 'input_mesh_faces', 'vertex_id_scope': 'input_mesh_vertices',
        'blocking_issues': issues, 'mesh_repair_performed': False,
        'self_intersection_status': 'not_checked',
        'shell_containment_status': 'not_checked',
        'solid_validity_certified': False,
        'quality_scope': 'edge_manifold_winding_degeneracy_and_signed_volume_only',
    }


def analysis_view(body):
    """Explicit NON-REPAIRED view for fits, retaining source faces in the evidence.

    Zero-area triangles cannot define normals. Only those triangles are omitted
    from this view; the original partition and quality blockers remain unchanged.
    """
    quality = mesh_quality(body.mesh)
    skipped = quality['degenerate_triangle_ids']
    mask = np.ones(len(body.mesh.faces), dtype=bool)
    mask[skipped] = False
    face_ids = np.flatnonzero(mask)
    if not len(face_ids):
        raise ValueError('No nondegenerate triangles available for candidate analysis')
    vertex_ids, faces = np.unique(body.mesh.faces[face_ids], return_inverse=True)
    view = trimesh.Trimesh(vertices=body.mesh.vertices[vertex_ids].copy(),
                           faces=faces.reshape((-1, 3)), process=False)
    result = MeshBody(view, body.instance_triangle_ids[face_ids], body.instance_vertex_ids[vertex_ids])
    metadata = {'purpose': 'candidate_fitting_only_not_mesh_repair',
                'source_triangle_count': len(body.mesh.faces),
                'analyzed_triangle_count': len(face_ids),
                'skipped_degenerate_instance_triangle_ids': body.instance_triangle_ids[skipped].tolist(),
                'faces_removed_from_source': 0,
                'source_quality_blockers_preserved': quality['blocking_issues']}
    return result, metadata


def map_source_indices(evidence, face_map, vertex_map, *, copy_data=False):
    """Map declared source indices only; generated node/segment IDs stay unchanged.

    A pure export uses copy_data=True. Shared boundary/candidate objects are
    visited once, preventing aliasing from applying a mapping twice.
    """
    evidence = deepcopy(evidence) if copy_data else evidence
    face_map, vertex_map = np.asarray(face_map, dtype=int), np.asarray(vertex_map, dtype=int)
    visited = set()
    face_arrays = {'source_triangle_ids', 'covered_source_triangle_ids', 'unresolved_source_triangle_ids',
                   'excluded_source_triangle_ids', 'caller_excluded_source_triangle_ids',
                   'event_triangle_ids', 'coplanar_triangle_ids', 'degenerate_triangle_ids'}
    vertex_arrays = {'source_vertex_ids', 'vertex_ids', 'on_plane_vertex_ids',
                     'boundary_edges_vertex_ids', 'nonmanifold_edges_vertex_ids',
                     'nonmanifold_edges', 'inconsistent_winding_edges',
                     'directed_edges_vertex_ids', 'branch_vertex_ids', 'open_vertex_ids',
                     'sampled_boundary_vertex_ids'}

    def visit(value):
        if isinstance(value, (dict, list)):
            if id(value) in visited:
                return
            visited.add(id(value))
        if isinstance(value, dict):
            is_edge_use = 'local_edge_index' in value and 'source_triangle_id' in value
            for key in list(value):
                if key in face_arrays:
                    value[key] = face_map[np.asarray(value[key], dtype=int)].tolist()
                elif key == 'source_triangle_id':
                    value[key] = int(face_map[value[key]])
                elif key in {'source_triangle_id_scope', 'triangle_id_scope'}:
                    value[key] = 'build_instance_faces'
                elif key in {'source_vertex_id_scope', 'vertex_id_scope'}:
                    value[key] = 'build_instance_vertices'
                elif key == 'source_key' and isinstance(value[key], (list, tuple)):
                    source_key = value[key]
                    if source_key and source_key[0] in {'edge', 'vertex'}:
                        value[key] = [source_key[0], *[int(vertex_map[i]) for i in source_key[1:]]]
                elif key in vertex_arrays:
                    value[key] = vertex_map[np.asarray(value[key], dtype=int)].tolist()
                elif is_edge_use and key in {'start', 'end', 'start_vertex_id', 'end_vertex_id'}:
                    value[key] = int(vertex_map[value[key]])
                else:
                    visit(value[key])
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(evidence)
    return evidence


def map_record_to_instance(record, body, instance_metadata, instance_index):
    """Map candidate membership and directed boundaries to retained instance arrays."""
    face_map, vertex_map = body.instance_triangle_ids, body.instance_vertex_ids
    record = map_source_indices(record, face_map, vertex_map)
    record['instance_triangle_ids'] = face_map.tolist()
    record['instance_vertex_ids'] = vertex_map.tolist()
    record['partition_kind'] = 'connected_triangle_shell_not_certified_cad_solid'
    record['source_components'] = []
    for component in instance_metadata['components']:
        ids = face_map[(face_map >= component['face_start']) & (face_map < component['face_end'])]
        if len(ids):
            record['source_components'].append({
                'component_index': component['component_index'], 'model_path': component['model_path'],
                'object_id': component['object_id'], 'component_path': component['component_path'],
                'source_face_ids': (ids - component['face_start']).tolist(),
                'source_to_world_mm': component['source_to_world_mm'],
                'original_unit': component['original_unit'],
            })
    candidates = [c for key in ('planes', 'cylinders', 'cones', 'spheres') for c in record.get(key, [])]
    for candidate in candidates:
        candidate['candidate_id'] = f'instance_{instance_index:04d}_shell_{record["body_index"]:04d}_' + candidate['candidate_id']
        candidate['identity_scope'] = 'source_snapshot_and_analysis_version_not_persistent_cad_face_id'
    return record
