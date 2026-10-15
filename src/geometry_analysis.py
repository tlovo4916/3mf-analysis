"""Area-based mesh evidence, never mesh repair or an analytic/CAD BRep.

Lengths are millimeters. Triangle IDs always address the *input body's local
face array*, before any caller's instance/source mapping. Frames are right-handed
local-to-source frames. ``plate_frame`` compacts unreferenced vertices, retaining
face order/winding and recording the vertex map in the returned mesh metadata.
"""
from collections import defaultdict

import numpy as np
import trimesh
from shapely.geometry import MultiPoint


# Adjacent panels of a 12-sided cylinder differ by 30 degrees. This is a
# connectivity threshold, NOT an accuracy certificate or a normal-fit tolerance.
_CYLINDER_BAND_DEGREES = 35.


def _nonnegative(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite and nonnegative') from exc
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _canonical(direction):
    direction = np.asarray(direction, dtype=float).copy()
    direction /= np.linalg.norm(direction)
    if direction[np.argmax(np.abs(direction))] < 0:
        direction *= -1
    return direction


def _prepare(mesh):
    """Read, validate and recenter without welding, reorienting or fixing faces."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or
            faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)):
        raise ValueError('A nonempty triangular mesh is required')
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('Triangle indices must be integers')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Triangle index outside the vertex array')
    ids, inverse = np.unique(faces, return_inverse=True)
    points = vertices[ids]
    if not np.isfinite(points).all():
        raise ValueError('Referenced mesh coordinates must be finite')
    anchor = points[0].copy()
    with np.errstate(over='ignore', invalid='ignore'):
        offsets = points - anchor
        scale = float(np.max(np.ptp(offsets, axis=0)))
    if not np.isfinite(offsets).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Mesh is degenerate or has nonfinite extent')
    compact_faces = inverse.reshape((-1, 3))
    triangles = offsets[compact_faces]
    # Normalize each face, rather than the whole body: a small cylinder next to
    # a huge plate must not become numerically degenerate.
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    edge_scale = np.maximum(np.max(np.abs(e1), axis=1), np.max(np.abs(e2), axis=1))
    if np.any(edge_scale <= 0):
        raise ValueError('Mesh contains degenerate triangles')
    cross = np.cross(e1 / edge_scale[:, None], e2 / edge_scale[:, None])
    length = np.linalg.norm(cross, axis=1)
    if np.any(length <= 32 * np.finfo(float).eps):
        raise ValueError('Mesh contains degenerate triangles')
    with np.errstate(over='ignore', invalid='ignore'):
        areas = .5 * length * edge_scale ** 2
    if not np.isfinite(areas).all() or not np.isfinite(areas.sum()) or areas.sum() <= 0:
        raise ValueError('Mesh surface area must be finite and positive')
    normals = cross / length[:, None]
    edges = defaultdict(list)
    for face_id, face in enumerate(faces):
        for edge_id in range(3):
            start, end = int(face[edge_id]), int(face[(edge_id + 1) % 3])
            edges[min(start, end), max(start, end)].append(
                (face_id, edge_id, start, end))
    adjacency = []
    for uses in edges.values():
        if len(uses) == 2:
            adjacency.append((uses[0][0], uses[1][0]))
    return {'vertices': vertices, 'faces': faces, 'vertex_ids': ids,
            'compact_faces': compact_faces, 'points': points, 'anchor': anchor,
            'offsets': offsets, 'triangles': triangles, 'normals': normals,
            'areas': areas, 'scale': scale, 'edges': edges, 'adjacency': adjacency}


def _moments(data, region=None):
    """Exact uniform-triangle surface centroid/covariance, normalized for PCA.

    For triangle vertices a,b,c, E[xx'] = (ss' + aa' + bb' + cc')/12,
    s=a+b+c. Integrating this with face-area weights is invariant to subdivision
    and duplicated vertex storage; a cloud-of-vertices covariance is not.
    """
    triangles = data['triangles'] if region is None else data['triangles'][region]
    areas = data['areas'] if region is None else data['areas'][region]
    base = triangles[0, 0]
    relative = triangles - base
    scale = float(np.max(np.abs(relative)))
    if scale <= 0:
        raise ValueError('Degenerate surface moment')
    normalized = relative / scale
    weights = areas / areas.sum()
    mean = np.einsum('i,ij->j', weights, normalized.mean(axis=1))
    centered = normalized - mean
    sums = centered.sum(axis=1)
    covariance = (np.einsum('i,ijk,ijl->kl', weights, centered, centered) +
                  np.einsum('i,ij,ik->jk', weights, sums, sums)) / 12
    return base + mean * scale, covariance, scale


def _basis(axis):
    x = np.eye(3)[np.argmin(np.abs(axis))]
    x = _canonical(x - axis * np.dot(x, axis))
    return x, np.cross(axis, x)


def _frame(origin, rotation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = origin
    return {'origin_mm': origin.tolist(), 'x_direction': rotation[:, 0].tolist(),
            'y_direction': rotation[:, 1].tolist(), 'z_direction': rotation[:, 2].tolist(),
            'local_to_source_rotation': rotation.tolist(),
            'local_to_source_transform': transform.tolist(),
            'handedness': 'right', 'coordinate_frame': 'source_mesh_mm'}


def _in_plane_rotation(data, normal, covariance):
    x, y = _basis(normal)
    basis = np.column_stack((x, y))
    xy = data['offsets'] @ basis / data['scale']
    _, vectors = np.linalg.eigh(basis.T @ covariance @ basis)
    directions = [vectors[:, -1]]
    # Recentered, normalized GEOS minimum rectangle avoids an arbitrary global
    # XY rotation. It uses the contour, not the number of interior mesh vertices.
    rectangle = MultiPoint(xy).minimum_rotated_rectangle
    if rectangle.geom_type == 'Polygon':
        corners = np.asarray(rectangle.exterior.coords)[:4]
        directions.extend(np.roll(corners, -1, axis=0) - corners)
    best = None
    for direction in directions:
        if np.linalg.norm(direction) <= np.finfo(float).eps:
            continue
        u = _canonical(basis @ direction)
        v = np.cross(normal, u)
        rotation = np.column_stack((u, v, normal))
        local = data['offsets'] @ rotation
        extent = np.ptp(local, axis=0)
        if extent[0] < extent[1]:
            u = _canonical(v)
            rotation = np.column_stack((u, np.cross(normal, u), normal))
            local = data['offsets'] @ rotation
            extent = np.ptp(local, axis=0)
        score = float((extent[0] / data['scale']) * (extent[1] / data['scale']))
        if best is None or score < best[0] - 1e-14:
            best = score, rotation, local
    return best


def _plate_frame(data):
    _, covariance, _ = _moments(data)
    _, pca = np.linalg.eigh(covariance)
    weights = data['areas'] / data['areas'].sum()
    _, normal_pca = np.linalg.eigh(
        np.einsum('i,ij,ik->jk', weights, data['normals'], data['normals']))
    normals = data['normals'].copy()
    major = np.argmax(np.abs(normals), axis=1)
    normals[normals[np.arange(len(normals)), major] < 0] *= -1
    # Aggregate area of equal normals instead of selecting frequently sampled
    # vertices/faces. PCA and normal null-space candidates are also considered.
    _, first, groups = np.unique(np.round(normals, 9), axis=0,
                                    return_index=True, return_inverse=True)
    group_area = np.bincount(groups, weights=weights)
    dominant = first[np.argsort(-group_area, kind='stable')[:24]]
    directions = [*pca.T, *normal_pca.T, *normals[dominant]]
    choices = []
    for direction in directions:
        normal = _canonical(direction)
        thickness = float(np.ptp(data['offsets'] @ normal))
        match = next((i for i, prior in enumerate(choices)
                      if np.dot(normal, prior[1]) > 1 - 1e-12), None)
        if match is not None:
            # Nearly parallel is not identical on a wide thin part. Retain
            # the thinner direction instead of letting PCA hide exact faces.
            if thickness < choices[match][0]:
                choices[match] = (thickness, normal)
            continue
        choices.append((thickness, normal))
    smallest = min(item[0] for item in choices)
    tie = 1e-11 * data['scale']
    best = None
    for thickness, normal in choices:
        if thickness > smallest + tie:
            continue
        choice = _in_plane_rotation(data, normal, covariance)
        if choice is not None and (best is None or choice[0] < best[0] - 1e-14):
            best = choice
    if best is None:
        raise ValueError('Mesh cannot define a nondegenerate frame')
    _, rotation, local = best
    midpoint = (local.min(axis=0) + local.max(axis=0)) / 2
    origin = data['anchor'] + rotation @ midpoint
    oriented = trimesh.Trimesh(vertices=local - midpoint,
                               faces=data['compact_faces'].copy(), process=False)
    oriented.metadata.update({'source_vertex_ids': data['vertex_ids'].tolist(),
                              'source_triangle_ids': list(range(len(data['faces']))),
                              'mesh_repair_performed': False})
    return rotation, origin, oriented


def plate_frame(mesh):
    """Return (right-handed local-to-source rotation, origin, local mesh).

    Surface-integral PCA / area-weighted normals select the thinnest direction;
    in-plane PCA / minimum contour rectangle reduce XY box rotation. Only
    referenced vertices participate. No normal fixing or other repair occurs.
    Empty, nonfinite referenced geometry and degenerate triangles raise ValueError.
    """
    return _plate_frame(_prepare(mesh))


def _regions(face_ids, links):
    remaining = set(int(i) for i in face_ids)
    neighbors = defaultdict(list)
    for a, b in links:
        neighbors[a].append(b)
        neighbors[b].append(a)
    result = []
    for seed in sorted(remaining):
        if seed not in remaining:
            continue
        remaining.remove(seed)
        stack, region = [seed], []
        while stack:
            face = stack.pop()
            region.append(face)
            for neighbor in neighbors[face]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        result.append(sorted(region))
    return result


def _planar_regions(data, tolerance, strict=False):
    links = []
    # Strict facets are scale-local, not an area fraction of the entire body.
    angle_cos = np.cos(np.deg2rad(1.)) if not strict else 1 - 1e-12
    for a, b in data['adjacency']:
        if abs(np.dot(data['normals'][a], data['normals'][b])) < angle_cos:
            continue
        pair = data['triangles'][[a, b]]
        span = np.max(np.abs(pair - pair[0, 0]))
        distance_limit = max(1e-10 * span, 1e-12) if strict else tolerance
        if np.max(np.abs((pair[1] - pair[0, 0]) @ data['normals'][a])) > distance_limit:
            continue
        if np.max(np.abs((pair[0] - pair[1, 0]) @ data['normals'][b])) > distance_limit:
            continue
        links.append((a, b))
    return _regions(range(len(data['faces'])), links)


def _boundary(data, region):
    """Directed edge-use walks; no welding, reversal, loop closure or CAD trims."""
    selected = set(region)
    boundary, nonmanifold, inconsistent = [], [], []
    keys = set()
    for face in data['faces'][region]:
        for i in range(3):
            start, end = int(face[i]), int(face[(i + 1) % 3])
            keys.add((min(start, end), max(start, end)))
    for key in sorted(keys):
        uses = data['edges'][key]
        chosen = [use for use in uses if use[0] in selected]
        if len(uses) > 2:
            nonmanifold.append(list(key))
        if len(chosen) == 1:
            boundary.append(chosen[0])
        elif len(chosen) == 2 and chosen[0][2:] == chosen[1][2:]:
            inconsistent.append(list(key))
    boundary.sort()
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for i, use in enumerate(boundary):
        outgoing[use[2]].append(i)
        incoming[use[3]].append(i)
    vertices = set(outgoing) | set(incoming)
    branched = any(len(outgoing[v]) > 1 or len(incoming[v]) > 1 for v in vertices)
    partial = bool(nonmanifold or inconsistent or branched)
    unused = set(range(len(boundary)))
    loops = []
    # Start at dangling/ambiguous nodes first; do not invent orientation changes.
    starts = [i for i, use in enumerate(boundary)
              if len(incoming[use[2]]) != 1 or len(outgoing[use[2]]) != 1]
    starts.extend(range(len(boundary)))
    for seed in starts:
        if seed not in unused:
            continue
        path, uses, current = [boundary[seed][2]], [], seed
        while current in unused:
            unused.remove(current)
            face, edge, start, end = boundary[current]
            uses.append({'source_triangle_id': int(face), 'local_edge_index': int(edge),
                         'start': int(start), 'end': int(end),
                         'start_vertex_id': int(start), 'end_vertex_id': int(end)})
            path.append(end)
            if end == path[0] or len(outgoing[end]) != 1 or len(incoming[end]) != 1:
                break
            current = outgoing[end][0]
        closed = path[0] == path[-1]
        loops.append({'loop_id': len(loops), 'vertex_ids': [int(v) for v in path],
                      'points_mm': data['vertices'][path].tolist(), 'edge_uses': uses,
                      'is_closed': closed,
                      'topology_status': 'partial' if partial else 'closed' if closed else 'open',
                      'is_cad_trim': False})
    status = ('partial' if partial else
              'open' if any(not loop['is_closed'] for loop in loops) else 'closed')
    return {'kind': 'directed_mesh_region_boundary_not_cad_trim', 'topology_status': status,
            'loops': loops, 'boundary_edge_count': len(boundary),
            'nonmanifold_edges': nonmanifold, 'inconsistent_winding_edges': inconsistent,
            'self_intersection_checked': False, 'is_cad_trim': False}


def _candidate_base(data, region, kind, origin, rotation):
    ids = np.unique(data['faces'][region])
    boundary = _boundary(data, region)
    return {'candidate_id': f'{kind}_{min(region):06d}', 'type': kind,
            'source_triangle_ids': [int(i) for i in region],
            'source_triangle_id_scope': 'body_local_faces',
            'source_vertex_ids': ids.tolist(), 'points_mm': data['vertices'][ids].tolist(),
            'coordinate_frame': 'source_mesh_mm', 'frame': _frame(origin, rotation),
            'ref_direction': rotation[:, 0].tolist(), 'triangle_count': len(region),
            'area_mm2': float(data['areas'][region].sum()),
            'boundary': boundary, 'boundary_loops': boundary['loops'],
            'candidate_geometry_only': True, 'cad_reconstructed': False,
            'confidence': 'geometric_fit_only'}


def _angles(dot):
    return np.rad2deg(np.arccos(np.clip(dot, -1., 1.)))


def _planar_candidates(data, tolerance, minimum_area):
    records = []
    for region in _planar_regions(data, tolerance):
        if data['areas'][region].sum() < minimum_area:
            continue
        center, covariance, _ = _moments(data, region)
        _, vectors = np.linalg.eigh(covariance)
        normal = _canonical(vectors[:, 0])
        ns = data['normals'][region]
        # Pairwise 1° adjacency is not transitive: many shallow bends can walk
        # around a toroidal band and fit a thin slab. Require every face normal
        # to agree with the FINAL plane, independently of the length tolerance.
        normal_error = float(_angles(np.abs(ns @ normal)).max())
        if normal_error > 1. + 1e-6:
            continue
        alignment = float(np.sum(data['areas'][region] * (ns @ normal)))
        if alignment < 0:
            normal = -normal
        ids = np.unique(data['compact_faces'][region])
        errors = np.abs((data['offsets'][ids] - center) @ normal)
        maximum = float(errors.max())
        if maximum > tolerance:
            continue
        centroids = data['triangles'][region].mean(axis=1)
        centroid_errors = np.abs((centroids - center) @ normal)
        u, v = _basis(normal)
        origin = data['anchor'] + center
        record = _candidate_base(data, region, 'plane', origin,
                                 np.column_stack((u, v, normal)))
        record.update({'normal': normal.tolist(), 'point_mm': origin.tolist(),
                       'parameterization': 'p(u,v) = frame.origin_mm + u*x_direction + v*y_direction',
                       'max_deviation_mm': maximum, 'max_vertex_residual_mm': maximum,
                       'max_centroid_residual_mm': float(centroid_errors.max()),
                       'max_surface_residual_mm': maximum,
                       'max_normal_residual_degrees': normal_error,
                       'normal_angular_threshold_degrees': 1.,
                       'normal_fit_within_threshold': True,
                       'max_oriented_normal_residual_degrees': float(_angles(ns @ normal).max()),
                       'normal_residual_basis': 'unoriented_source_face_normal_vs_plane_normal',
                       'surface_residual_method': 'exact_maximum_over_flat_triangles_at_vertices',
                       'surface_residual_scope': 'one_sided_mesh_to_untrimmed_analytic_support',
                       'vertex_fit_within_tolerance': True,
                       'surface_approximation_within_tolerance': True,
                       'fit_tolerance_mm': tolerance})
        records.append(record)
    return sorted(records, key=lambda record: (-record['area_mm2'], record['candidate_id']))


def planar_candidates(mesh, tolerance=.05, minimum_area=2.):
    """Connected plane evidence, including isolated single triangles of sufficient area."""
    tolerance = _nonnegative(tolerance, 'tolerance')
    minimum_area = _nonnegative(minimum_area, 'minimum_area')
    return _planar_candidates(_prepare(mesh), tolerance, minimum_area)


def _fit_circle(xy, weights):
    """Conditioned algebraic circle fit; xy already uses a recentered local frame."""
    weights = weights / weights.sum()
    mean = np.einsum('i,ij->j', weights, xy)
    centered = xy - mean
    scale = np.sqrt(np.sum(weights * np.sum(centered ** 2, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        return None
    normalized = centered / scale
    matrix = np.column_stack((2 * normalized, np.ones(len(xy))))
    weighted = matrix * np.sqrt(weights[:, None])
    condition = float(np.linalg.cond(weighted))
    if not np.isfinite(condition) or condition > 1e8:
        return None
    fit = np.linalg.lstsq(weighted, np.sum(normalized ** 2, axis=1) * np.sqrt(weights),
                          rcond=None)[0]
    squared = float(fit[2] + np.dot(fit[:2], fit[:2]))
    if not np.isfinite(squared) or squared <= 0:
        return None
    return mean + fit[:2] * scale, np.sqrt(squared) * scale, condition


def _angular_intervals(triangle_xy):
    """Union angular intervals actually spanned by projected mesh triangles."""
    intervals = []
    two_pi = 2 * np.pi
    angles = np.sort(np.mod(np.arctan2(triangle_xy[:, :, 1], triangle_xy[:, :, 0]), two_pi), axis=1)
    gaps = np.diff(np.column_stack((angles, angles[:, 0] + two_pi)), axis=1)
    for a, g in zip(angles, gaps):
        largest = int(np.argmax(g))
        length = two_pi - g[largest]
        if length >= np.pi - 1e-10:
            return None  # Triangle spans/crosses the axis, not a cylinder panel.
        start = a[(largest + 1) % 3]
        end = start + length
        if end > two_pi:
            intervals.extend(((start, two_pi), (0., end - two_pi)))
        else:
            intervals.append((start, end))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1e-10:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([float(start), float(end)])
    return merged


def _triangle_radial_minimum(xy):
    """Exact distance from the axis to each projected filled triangle."""
    following = np.roll(xy, -1, axis=1)
    edge = following - xy
    squared = np.sum(edge * edge, axis=2)
    t = np.divide(-np.sum(xy * edge, axis=2), squared,
                  out=np.zeros_like(squared), where=squared > 0)
    closest = xy + np.clip(t, 0, 1)[:, :, None] * edge
    minimum = np.sqrt(np.min(np.sum(closest * closest, axis=2), axis=1))
    crosses = xy[:, :, 0] * following[:, :, 1] - xy[:, :, 1] * following[:, :, 0]
    area2 = crosses.sum(axis=1)
    inside = ((np.all(crosses >= 0, axis=1) | np.all(crosses <= 0, axis=1)) &
              (np.abs(area2) > 1e-14 * np.max(np.sum(xy * xy, axis=2), axis=1)))
    minimum[inside] = 0
    return minimum


def _cylinder_regions(data):
    """Connected normal bands, excluding standalone strict planar facets.

    A faceted cylinder's rectangular panels are themselves strict two-triangle
    planar facets. Excluding those unconditionally would exclude *every* n=12/16
    cylinder. Retain facets incident to a changing-normal band, but exclude all
    standalone 2+ triangle strict facets (including tiny caps), without an area
    heuristic. The final axis/radius/taper tests disambiguate the retained bands.
    """
    strict = _planar_regions(data, 0., strict=True)
    labels = np.empty(len(data['faces']), dtype=int)
    for label, region in enumerate(strict):
        labels[region] = label
    links, curved_facets = [], set()
    threshold = np.cos(np.deg2rad(_CYLINDER_BAND_DEGREES))
    for a, b in data['adjacency']:
        dot = float(np.dot(data['normals'][a], data['normals'][b]))
        if dot < threshold:
            continue
        links.append((a, b))
        if dot < 1 - 1e-10:
            curved_facets.update((int(labels[a]), int(labels[b])))
    excluded = set()
    for label, region in enumerate(strict):
        if len(region) >= 2 and label not in curved_facets:
            excluded.update(region)
    links = [(a, b) for a, b in links if a not in excluded and b not in excluded]
    return _regions(set(range(len(data['faces']))) - excluded, links)


def _cylinder_candidates(data, tolerance, *, regions=None, diagnostics=None, recover=True):
    records = []
    bands = _cylinder_regions(data) if regions is None else regions
    for region in bands:
        if len(region) < 6:
            continue
        areas = data['areas'][region]
        weights = areas / areas.sum()
        ns = data['normals'][region]
        # Partial conical arcs can fool an uncentered normal null-space into a
        # tilted cylinder axis. Cone normals instead lie in an AFFINE plane
        # offset from zero. Detect that coherent offset before the null-space
        # fit; do not flatten a shallow cone just because length errors pass.
        mean_normal = np.einsum('i,ij->j', weights, ns)
        centered_normals = ns - mean_normal
        affine_values, affine_vectors = np.linalg.eigh(
            np.einsum('i,ij,ik->jk', weights, centered_normals, centered_normals))
        affine_dots = ns @ affine_vectors[:, 0]
        affine_offset = float(np.dot(weights, affine_dots))
        affine_spread = np.sqrt(np.dot(weights, (affine_dots - affine_offset) ** 2))
        if (affine_values[1] > 1e-8 and affine_values[0] < 1e-6 * affine_values[1] and
                abs(affine_offset) > max(1e-8, 8 * affine_spread)):
            continue
        values, vectors = np.linalg.eigh(np.einsum('i,ij,ik->jk', weights, ns, ns))
        if values[1] < 1e-8 or values[0] / values[1] > .01:
            continue
        axis = _canonical(vectors[:, 0])
        axis_normal_angle = np.rad2deg(np.arcsin(np.clip(np.abs(ns @ axis), 0, 1)))
        if axis_normal_angle.max() > .5:
            continue
        u, v = _basis(axis)
        basis = np.column_stack((u, v))
        center, _, _ = _moments(data, region)
        point_ids, inverse = np.unique(data['compact_faces'][region], return_inverse=True)
        inverse = inverse.reshape((-1, 3))
        points = data['offsets'][point_ids] - center
        vertex_weights = np.bincount(inverse.ravel(), weights=np.repeat(weights / 3, 3),
                                     minlength=len(point_ids))
        xy = points @ basis
        fitted = _fit_circle(xy, vertex_weights)
        if fitted is None:
            continue
        circle, radius, condition = fitted
        if not np.isfinite(radius) or radius <= 0:
            continue
        radial_xy = xy - circle
        radii = np.linalg.norm(radial_xy, axis=1)
        errors = np.abs(radii - radius)
        maximum = float(errors.max())
        if maximum > tolerance:
            continue
        axial = points @ axis
        height = float(np.ptp(axial))
        if height <= 1e-10 * max(radius, height):
            continue
        # Reject systematic taper, even when a shallow cone's vertex residuals
        # happen to be below the requested length tolerance.
        axial_centered = axial - np.dot(vertex_weights, axial)
        radius_centered = radii - np.dot(vertex_weights, radii)
        axial_variance = np.dot(vertex_weights, axial_centered ** 2)
        radial_variance = np.dot(vertex_weights, radius_centered ** 2)
        covariance = np.dot(vertex_weights, axial_centered * radius_centered)
        numerical = max(1e-8 * radius, 64 * np.finfo(float).eps * data['scale'])
        if (radial_variance > numerical ** 2 and axial_variance > 0 and
                abs(covariance) / np.sqrt(axial_variance * radial_variance) > .95 and
                abs(covariance / axial_variance) * height > numerical):
            continue
        triangle_xy = radial_xy[inverse]
        intervals = _angular_intervals(triangle_xy)
        if intervals is None:
            continue
        coverage = float(np.rad2deg(sum(end - start for start, end in intervals)))
        if coverage < 60 - 1e-3:  # angular roundoff allowance, not a mesh repair
            continue
        centroid_xy = triangle_xy.mean(axis=1)
        centroid_radii = np.linalg.norm(centroid_xy, axis=1)
        if np.any(centroid_radii <= 0) or np.any(radii <= 0):
            continue
        centroid_radial = (centroid_xy / centroid_radii[:, None]) @ basis.T
        signed_alignment = np.einsum('ij,ij->i', ns, centroid_radial)
        # A conical strip has a coherent nonzero axial slope in its normals.
        normal_slopes = np.divide(-(ns @ axis), signed_alignment,
                                  out=np.zeros(len(region)), where=np.abs(signed_alignment) > .5)
        mean_slope = np.dot(weights, normal_slopes)
        slope_spread = np.sqrt(np.dot(weights, (normal_slopes - mean_slope) ** 2))
        if abs(mean_slope) * height > numerical and slope_spread < .05 * abs(mean_slope):
            continue
        minimum_radii = _triangle_radial_minimum(triangle_xy)
        surface_error = max(maximum, float(np.max(np.abs(minimum_radii - radius))))
        vertex_radial = (triangle_xy / radii[inverse, None]) @ basis.T
        normal_dots = np.einsum('ij,ikj->ik', ns, vertex_radial)
        midpoint = float((axial.min() + axial.max()) / 2)
        axis_center = center + basis @ circle + axis * midpoint
        origin = data['anchor'] + axis_center
        rotation = np.column_stack((u, v, axis))
        record = _candidate_base(data, region, 'cylinder', origin, rotation)
        # Normalized floating-point sums can be 1 + 2e-16. Bound the producer's
        # fraction; do not relax strict validation of user-supplied evidence.
        inward_fraction = float(np.clip(np.sum(weights[signed_alignment < 0]), 0., 1.))
        inward = inward_fraction >= .95
        outward = inward_fraction <= .05
        role = 'hole' if inward else 'boss_or_outer_surface' if outward else 'unknown'
        role_status = ('possible_hole' if inward else
                       'possible_boss_or_outer_surface' if outward else 'ambiguous')
        record.update({'axis': axis.tolist(), 'axis_point_mm': origin.tolist(),
                       'radius_mm': float(radius), 'axial_length_mm': height,
                       'axial_parameter_range_mm': [-height / 2, height / 2],
                       'angular_parameter_intervals_radians': intervals,
                       'parameterization': ('p(theta,z) = frame.origin_mm + radius_mm * '
                                            '(cos(theta)*x_direction + sin(theta)*y_direction) + z*z_direction'),
                       'max_radial_error_mm': maximum, 'max_vertex_residual_mm': maximum,
                       'max_centroid_residual_mm': float(np.abs(centroid_radii - radius).max()),
                       'max_surface_residual_mm': surface_error,
                       'max_mesh_sagitta_mm': float(np.maximum(radius - minimum_radii, 0).max()),
                       'max_normal_residual_degrees': float(_angles(np.abs(normal_dots)).max()),
                       'max_centroid_normal_residual_degrees': float(_angles(np.abs(signed_alignment)).max()),
                       'max_axis_normal_residual_degrees': float(axis_normal_angle.max()),
                       'normal_residual_basis': 'unoriented_source_face_vs_analytic_radial_normal_at_face_vertices',
                       'surface_residual_method': 'exact_radial_extrema_over_projected_flat_triangles',
                       'surface_residual_scope': 'one_sided_mesh_to_untrimmed_analytic_support',
                       'vertex_fit_within_tolerance': True,
                       'surface_approximation_within_tolerance': bool(surface_error <= tolerance),
                       'fit_tolerance_mm': tolerance, 'fit_condition_number': condition,
                       'angular_coverage_degrees': min(360., coverage),
                       'wraps_full_circumference': bool(coverage >= 360 - 1e-6),
                       'normal_band_threshold_degrees': _CYLINDER_BAND_DEGREES,
                       'planar_facet_policy': 'exclude_standalone_strict_facets_retain_curved_band_panels',
                       'role_candidate': role, 'role_status': role_status,
                       'hole': {'classification': 'possible_hole' if inward else 'not_established',
                                'confirmed': False,
                                'basis': 'area_weighted_inward_source_face_normals_only',
                                'inward_normal_area_fraction': inward_fraction,
                                'cavity_topology_verified': False},
                       'role_evidence_note': ('Inward winding is not proof of a hole. Partial arcs, fillets and '
                                              'inverted shells require review; no role is confirmed.')})
        record['segmentation_method'] = 'whole_connected_normal_band'
        records.append(record)
    if recover:
        from cylinder_segmentation import recover_cylinders
        additional, search = recover_cylinders(
            data, tolerance, records, bands,
            lambda region: _cylinder_candidates(data, tolerance, regions=[region], recover=False))
        records.extend(additional)
        if diagnostics is not None:
            diagnostics.update(search)
    return sorted(records, key=lambda record: (-record['triangle_count'], record['candidate_id']))


def cylinder_candidates(mesh, tolerance=.05):
    """Connected, non-tapered cylinder fits; no coaxial-region merging.

    Vertex-fit tolerance does not imply a flat triangular surface approximates
    the cylinder to that tolerance. See max_surface_residual_mm and the explicit
    surface_approximation_within_tolerance flag. All roles are unconfirmed.
    """
    tolerance = _nonnegative(tolerance, 'tolerance')
    return _cylinder_candidates(_prepare(mesh), tolerance)


def _coverage(data, candidates):
    usage = np.zeros(len(data['faces']), dtype=int)
    for record in candidates:
        usage[record['source_triangle_ids']] += 1
    covered = np.flatnonzero(usage)
    unresolved = np.flatnonzero(usage == 0)
    # Include nonmanifold edge connectivity when grouping unresolved evidence.
    links = []
    for uses in data['edges'].values():
        links.extend((uses[0][0], use[0]) for use in uses[1:])
    regions = []
    for indices in _regions(unresolved, links):
        points = data['triangles'][indices].reshape((-1, 3))
        regions.append({'region_id': f'unresolved_{min(indices):06d}',
                        'source_triangle_ids': indices, 'triangle_count': len(indices),
                        'area_mm2': float(data['areas'][indices].sum()),
                        'bounds_mm': (data['anchor'] + np.array([points.min(axis=0), points.max(axis=0)])).tolist()})
    total_area = float(data['areas'].sum())
    return {'source_triangle_id_scope': 'body_local_faces',
            'total_triangle_count': len(usage), 'total_area_mm2': total_area,
            'covered_triangle_count': len(covered),
            'covered_area_mm2': float(data['areas'][covered].sum()),
            'covered_source_triangle_ids': covered.tolist(),
            'unresolved_triangle_count': len(unresolved),
            'unresolved_area_mm2': float(data['areas'][unresolved].sum()),
            'unresolved_source_triangle_ids': unresolved.tolist(),
            'overlap_triangle_count': int(np.sum(usage > 1)),
            'unresolved_regions': regions,
            'semantics': 'union_of_candidate_support_not_BRep_completeness_or_surface_tolerance_certificate'}


def inspect_mesh(mesh, instance, body, tolerance=.05):
    """Mesh evidence and review strategy, never CAD reconstruction/acceptance."""
    tolerance = _nonnegative(tolerance, 'tolerance')
    data = _prepare(mesh)
    rotation, origin, local = _plate_frame(data)
    ext = local.extents
    watertight = bool(mesh.is_watertight)
    winding = bool(mesh.is_winding_consistent)
    # Recenter first to avoid cancellation at large source translations. Keep
    # the SIGN: an inverted closed shell needs repair, not abs(volume).
    volume = None
    if watertight:
        center, _, _ = _moments(data)
        tri = data['triangles'] - center
        with np.errstate(over='ignore', invalid='ignore'):
            signed = float(np.einsum('ij,ij->i', tri[:, 0],
                                    np.cross(tri[:, 1], tri[:, 2])).sum() / 6)
        volume = signed if np.isfinite(signed) else None
    quality = None
    try:
        from mesh_evidence import mesh_quality
    except ModuleNotFoundError as exc:
        if exc.name != 'mesh_evidence':
            raise
    else:
        quality = mesh_quality(mesh)
    blocking = list(quality.get('blocking_issues', [])) if quality is not None else []
    reasons = []
    if not watertight:
        blocking.append('non_watertight_mesh')
        reasons.append('网格不封闭，需要修复；禁止自动实体重建。')
    if not winding:
        blocking.append('inconsistent_face_winding')
        reasons.append('面绕向不一致；分析未自动修复法向。')
    if watertight and (volume is None or volume <= 0):
        blocking.append('nonpositive_or_nonfinite_signed_volume')
        reasons.append('有向体积非正或非有限，可能存在倒置壳；未取绝对值或自动修复。')
    ratio = float(ext[2] / min(ext[:2])) if min(ext[:2]) else 1.
    # Preserve fine mechanics without treating every tessellated triangle of a
    # smooth shape as a designed plane. The public plane API still permits an
    # explicit area threshold; this adaptive threshold is part of the evidence.
    minimum_plane_area = min(2., float(data['areas'].sum()) * .001)
    planes = _planar_candidates(data, tolerance, minimum_plane_area)
    cylinder_search = {}
    cylinders = _cylinder_candidates(data, tolerance, diagnostics=cylinder_search)
    from analytic_surfaces import additional_surface_candidates
    recognized_cylinder_ids = sorted({i for c in cylinders for i in c['source_triangle_ids']})
    additional = additional_surface_candidates(mesh, tolerance, recognized_cylinder_ids)
    cones, spheres = additional['cones'], additional['spheres']
    # Analytic curved supports may include flat faceted panels. Retain their
    # faces in the support, but suppress redundant single-panel plane reports.
    curved_ids = {i for c in cylinders + cones + spheres for i in c['source_triangle_ids']}
    planes = [p for p in planes if not set(p['source_triangle_ids']).issubset(curved_ids)]
    if ratio <= .12 and not blocking:
        strategy = 'plate_profile_review'
        reasons.append('检测到薄板几何，可提出截面重建计划；仍需检查厚度方向变化。')
    elif len(planes) >= 2 or cylinders or cones or spheres:
        strategy = 'mechanical_feature_review'
        reasons.append('存在机械几何候选，但尚无完整特征组合重建器；不能输出已验证参数化零件。')
    else:
        strategy = 'freeform_or_unresolved'
        reasons.append('未找到可靠的机械结构分解，需要曲面逆向或人工确认。')
    if blocking:
        strategy = 'repair_required'
        reasons.append('存在网格阻塞证据，需要人工检查；候选拟合不解除重建阻塞。')
    coverage = _coverage(data, planes + cylinders + cones + spheres)
    reasons.append('候选覆盖仅指三角面归属，不是解析 BRep、CAD 裁剪边界或离散面自交验收。')
    if any(not c['surface_approximation_within_tolerance'] for c in cylinders + cones + spheres):
        reasons.append('部分曲面仅顶点拟合达标；整片误差或保守上界超差，不能称为精准解析曲面。')
    return {'instance_index': instance, 'body_index': body,
            'vertices': len(data['vertices']), 'referenced_vertices': len(data['vertex_ids']),
            'unused_vertices': len(data['vertices']) - len(data['vertex_ids']),
            'triangles': len(data['faces']), 'watertight': watertight, 'winding_consistent': winding,
            'bounds_mm': (data['anchor'] + np.array([data['offsets'].min(axis=0),
                                                   data['offsets'].max(axis=0)])).tolist(),
            'volume_mm3': volume, 'signed_volume_mm3': volume,
            'oriented_extents_mm': ext.tolist(), 'thinness_ratio': ratio,
            'local_to_source_rotation': rotation.tolist(), 'source_origin_mm': origin.tolist(),
            'planes': planes, 'cylinders': cylinders, 'cones': cones, 'spheres': spheres,
            'additional_surface_fit_diagnostics': additional,
            'cylinder_segmentation_diagnostics': cylinder_search,
            'plane_detection_policy': {'minimum_area_mm2': minimum_plane_area,
                                       'rule': 'min(2 mm2, total_surface_area * 0.001)',
                                       'curved_support_panels_suppressed': True},
            'candidate_coverage': coverage,
            'unresolved_regions': coverage['unresolved_regions'],
            'strategy': strategy, 'notes': reasons, 'blocking_issues': blocking,
            'mesh_quality': quality, 'mesh_repair_performed': False,
            'mesh_self_intersection_status': 'not_certified_by_geometry_candidates',
            'candidate_geometry_only': True, 'requires_review': True,
            'cad_reconstructed': False, 'brep_reconstructed': False,
            'design_intent_recovered': False}
