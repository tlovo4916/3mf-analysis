"""Read-only relationships between analytic *support candidates*, not CAD features.

All positions/directions use the record's source/world-mm frame. Candidate IDs
and collection indices refer back to the supplied record; surfaces, triangle
sets and review gates are never merged or changed. Spheres have no unique axis
and are deliberately excluded from coaxial comparisons.

``feature_relationships`` returns ``plane_relationships``, pairwise
``coaxial_surface_groups``, ``geometric_radial_gap_candidates``, an advisory
``classification_candidate`` and ``comparison_coverage``. There is a shared
512-pair budget, not a separate budget for each relationship type. Missing
optional angular, interval or winding evidence cannot establish a radial gap.
``section_directions`` returns source-frame axes and complete section frames
whose Z columns are those axes. It never estimates new normals from a mesh.
"""
from itertools import combinations
from numbers import Number, Real

import numpy as np


__all__ = ['feature_relationships', 'section_directions']

_PAIR_LIMIT = 512
_AXIS_ANGLE_DEGREES = .5  # independent of the LENGTH tolerance and mesh scale
_PLANE_ANGLE_DEGREES = .5
_FULL_ANGULAR_SHORTFALL_DEGREES = 1e-6
_DIRECTION_DOT_THRESHOLD = 1 - 1e-8
_NUMERIC_PARALLEL_SINE = 1e-12
_COLLECTIONS = {'planes': 'plane', 'cylinders': 'cylinder',
                'cones': 'cone', 'spheres': 'sphere'}


def _finite_numbers(value, path='record'):
    """Reject nonfinite input even in omitted candidates/unconsumed fit evidence."""
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_numbers(item, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_numbers(item, f'{path}[{index}]')
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in 'biuf':
            if not np.isfinite(value).all():
                raise ValueError(f'{path} must contain only finite numbers')
        elif value.dtype.kind == 'c':
            raise ValueError(f'{path} must contain real numbers')
        else:
            for index, item in enumerate(value.flat):
                _finite_numbers(item, f'{path}[{index}]')
    elif isinstance(value, (Number, np.number)) and not isinstance(value, (bool, np.bool_)):
        try:
            finite = bool(np.isfinite(float(value)))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            raise ValueError(f'{path} must contain only finite real numbers')


def _number(value, name, minimum=0., positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number')
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite real number') from exc
    if not np.isfinite(result) or result < minimum or (positive and result <= minimum):
        raise ValueError(f'{name} must be finite and {"greater than" if positive else "at least"} {minimum}')
    return result


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite array of shape {shape}') from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite array of shape {shape}')
    return result


def _unit(value, name):
    vector = _array(value, (3,), name)
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        raise ValueError(f'{name} must be nonzero')
    # Do not underflow tiny axes or overflow large, otherwise finite vectors.
    vector = vector / scale
    return vector / np.linalg.norm(vector)


def _canonical(axis):
    return -axis if axis[np.argmax(np.abs(axis))] < 0 else axis.copy()


def _frame_data(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a source-frame dictionary')
    transform = None
    if 'local_to_source_transform' in value:
        transform = _array(value['local_to_source_transform'], (4, 4), name + '.transform')
        if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-10):
            raise ValueError(f'{name} must be an affine rigid frame')
    if 'local_to_source_rotation' in value:
        rotation = _array(value['local_to_source_rotation'], (3, 3), name + '.rotation')
    elif transform is not None:
        rotation = transform[:3, :3]
    elif all(key in value for key in ('x_direction', 'y_direction', 'z_direction')):
        rotation = np.column_stack([_array(value[key], (3,), name + '.' + key)
                                    for key in ('x_direction', 'y_direction', 'z_direction')])
    else:
        raise ValueError(f'{name} is missing its source rotation')
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8) or
            not np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-8)):
        raise ValueError(f'{name} must be right-handed and orthonormal')
    origin_value = value.get('origin_mm', value.get('source_origin_mm'))
    if origin_value is None and transform is not None:
        origin_value = transform[:3, 3]
    origin = _array(origin_value, (3,), name + '.origin_mm')
    if transform is not None and (
            not np.allclose(rotation, transform[:3, :3], rtol=0, atol=1e-8) or
            not np.allclose(origin, transform[:3, 3], rtol=0, atol=1e-8)):
        raise ValueError(f'{name} has inconsistent rotation/transform/origin')
    for index, key in enumerate(('x_direction', 'y_direction', 'z_direction')):
        if key in value and not np.allclose(_array(value[key], (3,), name + '.' + key),
                                          rotation[:, index], rtol=0, atol=1e-8):
            raise ValueError(f'{name} has inconsistent direction columns')
    return origin, rotation


def _body_frame(record):
    if 'frame' in record:
        return _frame_data(record['frame'], 'record.frame')
    if any(key in record for key in ('local_to_source_rotation', 'local_to_source_transform')):
        return _frame_data(record, 'record')
    return None


def _mesh_finite(mesh):
    # A mesh is not required for support-only callers. Empty meshes are allowed;
    # invalid topology/repair blockers are NOT silently repaired or cleared.
    if mesh is None:
        return
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('mesh must expose finite vertices and triangular faces') from exc
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or
            not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1:] != (3,) or
            not np.issubdtype(faces.dtype, np.integer)):
        raise ValueError('mesh must expose finite vertices and integer triangular faces')
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('mesh triangle index outside vertex array')


def _angular_coverage(candidate, name):
    claimed = candidate.get('angular_coverage_degrees')
    if claimed is not None:
        claimed = _number(claimed, name + '.angular_coverage_degrees')
        if claimed > 360.:
            raise ValueError(f'{name}.angular_coverage_degrees must not exceed 360')
    intervals = candidate.get('angular_parameter_intervals_radians')
    measured = None
    if intervals is not None:
        try:
            intervals = np.asarray(intervals, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} angular intervals must be finite pairs') from exc
        if intervals.size == 0:
            intervals = np.empty((0, 2))
        if (intervals.ndim != 2 or intervals.shape[1:] != (2,) or
                not np.isfinite(intervals).all() or np.any(intervals[:, 0] < 0) or
                np.any(intervals[:, 1] > 2 * np.pi) or
                np.any(intervals[:, 1] < intervals[:, 0])):
            raise ValueError(f'{name} angular intervals must lie in [0, 2*pi]')
        merged = []
        for start, end in sorted(intervals.tolist()):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        measured = min(360., float(np.rad2deg(sum(end - start for start, end in merged))))
    coverage = min(claimed, measured) if claimed is not None and measured is not None else (
        claimed if claimed is not None else measured if measured is not None else 0.)
    wraps = candidate.get('wraps_full_circumference', False)
    if not isinstance(wraps, (bool, np.bool_)):
        raise ValueError(f'{name}.wraps_full_circumference must be boolean')
    full = bool(wraps and coverage >= 360. - _FULL_ANGULAR_SHORTFALL_DEGREES)
    return coverage, full


def _orientation(candidate, name):
    orientations = []
    if 'normal_orientation' in candidate:
        label = candidate['normal_orientation']
        if label not in ('inward', 'outward', 'mixed', 'unknown'):
            raise ValueError(f'{name}.normal_orientation is invalid')
        orientations.append(label)
    fraction = candidate.get('inward_normal_area_fraction')
    hole = candidate.get('hole', {})
    if not isinstance(hole, dict):
        raise ValueError(f'{name}.hole must be an evidence dictionary')
    for value in (fraction, hole.get('inward_normal_area_fraction')):
        if value is not None:
            value = _number(value, name + '.inward_normal_area_fraction')
            if value > 1:
                raise ValueError(f'{name}.inward_normal_area_fraction must not exceed 1')
            orientations.append('inward' if value >= .95 else 'outward' if value <= .05 else 'mixed')
    # Role labels alone are not winding evidence, and conflicting evidence is
    # deliberately not resolved in favor of a convenient shell hypothesis.
    return orientations[0] if orientations and len(set(orientations)) == 1 else 'unknown'


def _records(record):
    if not isinstance(record, dict):
        raise ValueError('record must be a dictionary')
    _finite_numbers(record)
    body_frame = _body_frame(record)
    if 'thinness_ratio' in record:
        _number(record['thinness_ratio'], 'record.thinness_ratio')
    if 'oriented_extents_mm' in record:
        extents = _array(record['oriented_extents_mm'], (3,), 'record.oriented_extents_mm')
        if np.any(extents < 0):
            raise ValueError('record.oriented_extents_mm must be nonnegative')
    candidates = []
    counts = {}
    for collection, kind in _COLLECTIONS.items():
        values = record.get(collection, [])
        if not isinstance(values, (list, tuple)):
            raise ValueError(f'record.{collection} must be a candidate list')
        counts[collection] = len(values)
        for index, candidate in enumerate(values):
            name = f'record.{collection}[{index}]'
            if not isinstance(candidate, dict):
                raise ValueError(f'{name} must be a candidate dictionary')
            candidate_id = candidate.get('candidate_id', f'{kind}_{index:06d}')
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError(f'{name}.candidate_id must be a nonempty string')
            frame = _frame_data(candidate['frame'], name + '.frame') if 'frame' in candidate else None
            item = {'kind': kind, 'ref': {'candidate_id': candidate_id,
                    'collection': collection, 'index': index},
                    'area': _number(candidate.get('area_mm2', 0.), name + '.area_mm2')}
            if kind == 'plane' or kind in ('cylinder', 'cone'):
                vector_key, point_key = ('normal', 'point_mm') if kind == 'plane' else ('axis', 'axis_point_mm')
                direction = _unit(candidate.get(vector_key, frame[1][:, 2] if frame else None), name + '.' + vector_key)
                origin = _array(candidate.get(point_key, frame[0] if frame else None), (3,), name + '.' + point_key)
                if frame is not None and abs(np.dot(direction, frame[1][:, 2])) < _DIRECTION_DOT_THRESHOLD:
                    raise ValueError(f'{name} axis/normal disagrees with its source frame')
                item.update({'axis': _canonical(direction), 'origin': origin,
                             'parameter_axis': frame[1][:, 2] if frame else direction,
                             'parameter_origin': frame[0] if frame else origin})
            if kind in ('cylinder', 'cone'):
                coverage, full = _angular_coverage(candidate, name)
                item.update({'coverage': coverage, 'full': full,
                             'orientation': _orientation(candidate, name)})
                bounds = candidate.get('axial_parameter_range_mm')
                if bounds is not None:
                    bounds = _array(bounds, (2,), name + '.axial_parameter_range_mm')
                    if bounds[1] < bounds[0]:
                        raise ValueError(f'{name} axial parameter bounds must be ordered')
                item['bounds'] = bounds
            if kind == 'cylinder':
                radius = candidate.get('radius_mm')
                item['radius'] = _number(radius, name + '.radius_mm', positive=True) if radius is not None else None
            if kind == 'sphere' and 'radius_mm' in candidate:
                _number(candidate['radius_mm'], name + '.radius_mm', positive=True)
            candidates.append(item)
    return candidates, counts, body_frame


def _angle(first, second):
    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(first, second)),
                                      abs(np.dot(first, second)))))


def _line_distance(first, second):
    with np.errstate(over='ignore', invalid='ignore'):
        delta = second['origin'] - first['origin']
        cross = np.cross(first['axis'], second['axis'])
        sine = float(np.linalg.norm(cross))
        if sine <= _NUMERIC_PARALLEL_SINE:
            distance = float(np.hypot.reduce(np.cross(delta, first['axis'])))
        else:
            distance = float(abs(np.dot(delta, cross / sine)))
    if not np.isfinite(distance):
        raise ValueError('Axis line distance is nonfinite at the supplied coordinate scale')
    return distance


def _pair_refs(first, second):
    return {'candidate_ids': [first['ref']['candidate_id'], second['ref']['candidate_id']],
            'candidate_refs': [dict(first['ref']), dict(second['ref'])]}


def _plane_relationship(first, second):
    angle = _angle(first['axis'], second['axis'])
    if angle <= _PLANE_ANGLE_DEGREES:
        with np.errstate(over='ignore', invalid='ignore'):
            delta = second['origin'] - first['origin']
            projections = [float(abs(np.dot(delta, item['axis']))) for item in (first, second)]
        if not np.isfinite(projections).all():
            raise ValueError('Plane separation is nonfinite at the supplied coordinate scale')
        parallel = np.linalg.norm(np.cross(first['axis'], second['axis'])) <= _NUMERIC_PARALLEL_SINE
        return {**_pair_refs(first, second), 'type': 'parallel_plane_candidate',
                'normal_angle_degrees': angle,
                'distance_mm': projections[0] if parallel else None,
                'anchor_projected_distances_mm': projections,
                'distance_is_constant_on_supports': bool(parallel),
                'distance_basis': 'untrimmed_supports_not_material_thickness; tilted_supports_have_only_anchor_projections'}
    if 90. - angle <= _PLANE_ANGLE_DEGREES:
        return {**_pair_refs(first, second), 'type': 'orthogonal_plane_candidate',
                'normal_angle_degrees': angle,
                'trimmed_surface_intersection_verified': False}
    return None


def _radial_gap(first, second, group):
    if not (first['kind'] == second['kind'] == 'cylinder' and first['full'] and second['full']):
        return None
    if {first['orientation'], second['orientation']} != {'inward', 'outward'}:
        return None
    inner, outer = (first, second) if first['orientation'] == 'inward' else (second, first)
    if (inner['radius'] is None or outer['radius'] is None or
            outer['radius'] <= inner['radius'] or first['bounds'] is None or second['bounds'] is None):
        return None
    # Parameter intervals belong to each candidate's OWN signed frame Z axis.
    # Project relative to one origin, not onto an absolute world coordinate.
    origin, axis = first['parameter_origin'], first['axis']
    projected = []
    with np.errstate(over='ignore', invalid='ignore'):
        for item in (first, second):
            offset = np.dot(item['parameter_origin'] - origin, axis)
            bounds = offset + item['bounds'] * np.dot(item['parameter_axis'], axis)
            projected.append([float(bounds.min()), float(bounds.max())])
    if not np.isfinite(projected).all():
        raise ValueError('Projected axial intervals are nonfinite')
    overlap = [max(interval[0] for interval in projected), min(interval[1] for interval in projected)]
    if overlap[1] <= overlap[0]:  # endpoint contact is not an overlapping wall region
        return None
    overlap_length = _number(overlap[1] - overlap[0], 'projected axial overlap length', positive=True)
    return {**_pair_refs(inner, outer), 'type': 'geometric_radial_gap_candidate',
            'inner_candidate_id': inner['ref']['candidate_id'],
            'outer_candidate_id': outer['ref']['candidate_id'],
            'inner_radius_mm': inner['radius'], 'outer_radius_mm': outer['radius'],
            'radial_gap_mm': outer['radius'] - inner['radius'],
            'axis_angle_degrees': group['axis_angle_degrees'], 'line_distance_mm': group['line_distance_mm'],
            'angular_coverage_degrees': [inner['coverage'], outer['coverage']],
            'full_circumference_shortfall_threshold_degrees': _FULL_ANGULAR_SHORTFALL_DEGREES,
            'reference_axis_point_mm': origin.tolist(), 'reference_axis': axis.tolist(),
            'axial_overlap_parameter_range_mm': overlap,
            'axial_overlap_length_mm': overlap_length,
            'candidate_geometry_only': True, 'wall_thickness_certified': False,
            'note': ('Difference of fitted support radii only. Source winding, near-coaxial axes and projected '
                     'interval overlap do not certify material, uniform wall thickness, trim overlap or CAD topology.')}


def _classification(counts, planes, groups, gaps, axis_coverage):
    axes = counts['cylinders'] + counts['cones']
    if gaps:
        return 'housing_or_shell_review'
    if axes and not counts['spheres']:
        total = axes * (axes - 1) // 2
        if axis_coverage['omitted_pair_count'] == 0 and len(groups) == total:
            return 'revolved_part_review'
        return 'mixed_mechanical_review'
    if axes:
        return 'mixed_mechanical_review'
    if planes and not counts['spheres']:
        return 'prismatic_sections_review'
    return 'freeform_review'


def feature_relationships(mesh, record, tolerance=.05) -> dict:
    """Compare supports in source/world mm with a finite, strictly positive tolerance.

    Coaxial groups are *pairs*, never connected components: A~B and B~C do not
    establish A~C. The distance test is the shortest distance between infinite
    axis lines; the separate unoriented axis-angle limit is always 0.5 degrees.
    Near-parallel planes get anchor projections, not a fictional constant
    support distance. Classification is only a review hint, not a CAD/gate state.
    ``mesh=None`` permits record-only analysis; a supplied mesh is read-only.
    """
    tolerance = _number(tolerance, 'tolerance', positive=True)
    _mesh_finite(mesh)
    candidates, counts, _ = _records(record)
    planes = sorted((item for item in candidates if item['kind'] == 'plane'),
                    key=lambda item: (-item['area'], item['ref']['candidate_id'], item['ref']['index']))
    axes = sorted((item for item in candidates if item['kind'] in ('cylinder', 'cone')),
                  key=lambda item: (-(item['area'] * (item['coverage'] / 360.)),
                                    item['ref']['candidate_id'], item['kind'], item['ref']['index']))
    # Normalize angular coverage before multiplying: even finite, huge areas
    # cannot overflow this relevance score.
    streams = {'plane_pairs': iter(combinations(planes, 2)), 'axial_surface_pairs': iter(combinations(axes, 2))}
    coverage = {}
    for key, values in (('plane_pairs', planes), ('axial_surface_pairs', axes)):
        coverage[key] = {'eligible_pair_count': len(values) * (len(values) - 1) // 2,
                         'compared_pair_count': 0}
    plane_relations, groups, gaps, seen = [], [], [], set()
    compared = 0
    while streams and compared < _PAIR_LIMIT:
        for key in list(streams):
            if compared >= _PAIR_LIMIT:
                break
            pair = next(streams[key], None)
            if pair is None:
                del streams[key]
                continue
            first, second = pair
            compared += 1
            coverage[key]['compared_pair_count'] += 1
            seen.update((item['ref']['collection'], item['ref']['index']) for item in pair)
            if key == 'plane_pairs':
                relation = _plane_relationship(first, second)
                if relation is not None:
                    plane_relations.append(relation)
                continue
            angle = _angle(first['axis'], second['axis'])
            if angle > _AXIS_ANGLE_DEGREES:
                continue
            distance = _line_distance(first, second)
            if distance > tolerance:
                continue
            group = {**_pair_refs(first, second), 'type': 'coaxial_support_pair_candidate',
                     'axis_angle_degrees': angle, 'line_distance_mm': distance,
                     'axis_angular_threshold_degrees': _AXIS_ANGLE_DEGREES,
                     'line_distance_threshold_mm': tolerance,
                     'all_members_pairwise_compatible': True,
                     'grouping_method': 'compatible_pairs_only_no_transitive_union',
                     'candidate_geometry_only': True, 'surfaces_merged': False}
            groups.append(group)
            gap = _radial_gap(first, second, group)
            if gap is not None:
                gaps.append(gap)
    for item in coverage.values():
        item['omitted_pair_count'] = item['eligible_pair_count'] - item['compared_pair_count']
    total = sum(item['eligible_pair_count'] for item in coverage.values())
    omitted = total - compared
    classification = _classification(counts, plane_relations, groups, gaps, coverage['axial_surface_pairs'])
    return {'plane_relationships': plane_relations, 'coaxial_surface_groups': groups,
            'geometric_radial_gap_candidates': gaps, 'classification_candidate': classification,
            'classification_basis': 'observed_support_relationships_only_not_design_intent_or_complete_part_symmetry',
            'thresholds': {'plane_angular_threshold_degrees': _PLANE_ANGLE_DEGREES,
                           'axis_angular_threshold_degrees': _AXIS_ANGLE_DEGREES,
                           'axis_line_distance_tolerance_mm': tolerance,
                           'full_circumference_shortfall_threshold_degrees': _FULL_ANGULAR_SHORTFALL_DEGREES},
            'comparison_coverage': {'pair_limit': _PAIR_LIMIT, 'eligible_pair_count': total,
                                    'compared_pair_count': compared, 'omitted_pair_count': omitted,
                                    'pair_coverage_fraction': compared / total if total else 1.,
                                    'complete': omitted == 0, 'by_kind': coverage,
                                    'candidate_counts': counts, 'compared_candidate_count': len(seen),
                                    'uncompared_eligible_candidate_count': len(planes) + len(axes) - len(seen),
                                    'axisless_sphere_candidate_count': counts['spheres'],
                                    'order': 'round_robin_plane_axial; area_or_area_times_angular_coverage_then_id',
                                    'scope': 'eligible_support_pairs_not_surface_area_or_BRep_completeness'},
            'candidate_geometry_only': True, 'requires_review': True, 'affects_review_gate': False,
            'cad_reconstructed': False, 'design_intent_recovered': False,
            'note': 'Design candidates are not recovered CAD features; source review blockers remain unchanged.'}


def _is_plate(record):
    if 'thinness_ratio' in record:
        return record['thinness_ratio'] <= .12
    extents = record.get('oriented_extents_mm')
    return bool(extents is not None and min(extents[:2]) > 0 and extents[2] / min(extents[:2]) <= .12)


def _section_frame(origin, axis, preferred_x):
    x = preferred_x - axis * np.dot(preferred_x, axis)
    if np.linalg.norm(x) <= np.sqrt(2e-8):
        seed = np.eye(3)[np.argmin(np.abs(axis))]
        x = seed - axis * np.dot(seed, axis)
    # Repeat the projection to avoid cancellation in nearly parallel seeds.
    x -= axis * np.dot(x, axis)
    x = _canonical(_unit(x, 'section x direction'))
    rotation = np.column_stack((x, np.cross(axis, x), axis))
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, origin
    return {'origin_mm': origin.tolist(), 'x_direction': x.tolist(),
            'y_direction': rotation[:, 1].tolist(), 'z_direction': axis.tolist(),
            'local_to_source_rotation': rotation.tolist(),
            'local_to_source_transform': transform.tolist(),
            'handedness': 'right', 'coordinate_frame': 'source_mesh_mm'}


def section_directions(record, max_directions=3) -> list[dict]:
    """Return at most three deterministic, independent source-frame directions.

    Plate: local column 2 first, then columns 0/1. Otherwise, rank cylinder/cone
    axes by area * angular coverage, then fall back to local columns 0/1/2
    (column 0 is the recorded longest axis). Opposite/duplicate directions use
    abs(dot) >= 1-1e-8; rank-deficient triples are also skipped so that an
    orthogonal local fallback can supply the missing dimension. No input edits.
    """
    if (isinstance(max_directions, (bool, np.bool_)) or
            not isinstance(max_directions, (int, np.integer)) or max_directions < 0):
        raise ValueError('max_directions must be a nonnegative integer')
    limit = min(int(max_directions), 3)
    candidates, _, body_frame = _records(record)
    plate = _is_plate(record)
    choices = []
    if not plate:
        curved = sorted((item for item in candidates if item['kind'] in ('cylinder', 'cone')),
                        key=lambda item: (-(item['area'] * (item['coverage'] / 360.)),
                                          item['ref']['candidate_id'], item['kind'], item['ref']['index']))
        for item in curved:
            choices.append((item['axis'], f'{item["kind"]}_axis:{item["ref"]["candidate_id"]}',
                            'Dominant observed curved-support axis, ranked by area times angular coverage; not design intent.',
                            False, item['parameter_origin'], item['ref']))
    if body_frame is not None:
        origin, rotation = body_frame
        for index in ((2, 0, 1) if plate else (0, 1, 2)):
            reason = ('Recorded thin-plate direction (local column 2).' if plate and index == 2 else
                      'Recorded longest local axis (column 0).' if index == 0 else
                      'Independent recorded local-axis fallback.')
            choices.append((_canonical(_unit(rotation[:, index], 'body axis')),
                            f'local_frame_column_{index}', reason, plate and index == 2, origin, None))
    result, selected, basis = [], [], []
    for axis, source, reason, plate_direction, candidate_origin, reference in choices:
        if len(result) >= limit:
            break
        if any(abs(np.dot(axis, prior)) >= _DIRECTION_DOT_THRESHOLD for prior in selected):
            continue
        residual = axis.copy()
        for direction in basis:
            residual -= direction * np.dot(direction, residual)
        norm = float(np.linalg.norm(residual))
        if norm <= np.sqrt(2e-8):
            continue
        basis.append(residual / norm)
        selected.append(axis)
        origin = body_frame[0] if body_frame is not None else candidate_origin
        preferred_x = body_frame[1][:, 0] if body_frame is not None else np.eye(3)[np.argmin(np.abs(axis))]
        result.append({'id': f'section_direction_{len(result):02d}', 'axis': axis.tolist(),
                       'axis_source': source, 'reason': reason, 'is_plate_direction': bool(plate_direction),
                       'source_candidate_ref': dict(reference) if reference is not None else None,
                       'frame': _section_frame(origin, axis, preferred_x),
                       'coordinate_frame': 'source_mesh_mm', 'candidate_geometry_only': True})
    return result
