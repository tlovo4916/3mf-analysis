"""Connected sphere/cone evidence, not CAD surfaces or recovered design intent.

Only input-body triangle IDs are returned. No welding, normal repair, random
hypotheses, disconnected-patch merging, or whole-body fallback fit is performed.
Sphere mesh-to-support error is an exact radial-extrema calculation. Cone point
errors are sampled; a separate conservative bound, never the samples alone,
controls the surface-approximation flag. Neither result certifies CAD trims.
"""
import numpy as np
import trimesh
from scipy.optimize import least_squares

from geometry_analysis import (
    _CYLINDER_BAND_DEGREES, _angles, _angular_intervals, _basis, _candidate_base,
    _canonical, _cylinder_regions, _moments, _nonnegative, _prepare, _regions,
    _triangle_radial_minimum,
)


_MIN_TRIANGLES = 12
_MIN_ANGLE_DEGREES = 60.


def _failure(reason, **values):
    return None, {'reason': reason, **values}


def _samples(data, region):
    """Condition locally; lump each face's area equally onto its three vertices."""
    areas = data['areas'][region]
    weights = areas / areas.sum()
    ids, inverse = np.unique(data['compact_faces'][region], return_inverse=True)
    inverse = inverse.reshape((-1, 3))
    vertex_weights = np.bincount(inverse.ravel(), weights=np.repeat(weights / 3, 3),
                                 minlength=len(ids))
    center, _, _ = _moments(data, region)
    relative = data['offsets'][ids] - center
    scale = float(np.sqrt(np.dot(vertex_weights, np.sum(relative ** 2, axis=1))))
    points = relative / scale
    # This is source-coordinate quantization, not permission to repair a mesh.
    source_ids = np.unique(data['faces'][region])
    uncertainty = max(16 * np.finfo(float).eps * scale,
                      float(np.max(np.abs(np.spacing(data['vertices'][source_ids])))))
    return {'points': points, 'triangles': points[inverse], 'inverse': inverse,
            'weights': weights, 'vertex_weights': vertex_weights, 'center': center,
            'scale': scale, 'uncertainty_mm': uncertainty,
            'normals': data['normals'][region]}


def _normal_evidence(samples, expected, corner_expected):
    """Unoriented fit, with winding retained and a resolution-aware acceptance gate.

    Analytic normals at flat-face centroids need not equal flat-face normals.
    Half the corner-normal spread plus 0.25 degrees allows ordinary faceting,
    but is bounded at 12 degrees and does not replace the position-fit test.
    """
    ns, weights = samples['normals'], samples['weights']
    signed = np.einsum('ij,ij->i', ns, expected)
    residual = _angles(np.abs(signed))
    spread = _angles(np.einsum('ij,ikj->ik', expected, corner_expected)).max(axis=1)
    allowed = np.minimum(12., .25 + .5 * spread)
    # Large translations leave quantized source vertices; do not use world
    # coordinates in the fit, but account for their angular uncertainty.
    triangles = samples['triangles'] * samples['scale']
    edges = np.linalg.norm(np.roll(triangles, -1, axis=1) - triangles, axis=2)
    numerical = np.rad2deg(8 * samples['uncertainty_mm'] / edges.min(axis=1))
    allowed = np.maximum(allowed, np.minimum(2., numerical))
    inward_fraction = float(np.clip(weights[signed < 0].sum(), 0., 1.))
    uniform_winding = inward_fraction <= .05 or inward_fraction >= .95
    evidence = {
        'normal_rms_degrees': float(np.sqrt(np.dot(weights, residual ** 2))),
        'normal_max_degrees': float(residual.max()),
        'max_normal_residual_degrees': float(residual.max()),
        'normal_residual_basis': 'unoriented_source_face_vs_support_normal_at_face_centroids',
        'normal_orientation': ('inward' if inward_fraction >= .95 else
                               'outward' if inward_fraction <= .05 else 'mixed'),
        'inward_normal_area_fraction': inward_fraction,
        'normal_resolution_allowance_max_degrees': float(allowed.max()),
        'normal_consistency_within_resolution': bool(np.all(residual <= allowed) and uniform_winding),
    }
    return evidence


def _sphere(data, region, samples, tolerance):
    p, vw = samples['points'], samples['vertex_weights']
    # Cylinder/frustum rings can all lie on a sphere. Their face normals occupy
    # an affine plane, unlike a genuinely two-dimensional spherical normal field.
    ns, weights = samples['normals'], samples['weights']
    mean_normal = np.einsum('i,ij->j', weights, ns)
    normal_delta = ns - mean_normal
    normal_values = np.linalg.eigvalsh(np.einsum('i,ij,ik->jk', weights, normal_delta, normal_delta))
    if normal_values[0] < .005 * normal_values[-1] or normal_values[-1] <= 1e-8:
        return _failure('sphere_normals_lack_two_dimensional_curvature')
    mean = np.einsum('i,ij->j', vw, p)
    q = p - mean
    matrix = np.column_stack((2 * q, np.ones(len(q))))
    weighted = matrix * np.sqrt(vw[:, None])
    singular = np.linalg.svd(weighted, compute_uv=False)
    if singular[-1] <= 0 or singular[-1] < 1e-6 * singular[0]:
        return _failure('sphere_fit_rank_or_condition_deficient')
    condition = float(singular[0] / singular[-1])
    fit = np.linalg.lstsq(weighted, np.sum(q * q, axis=1) * np.sqrt(vw), rcond=None)[0]
    squared = float(fit[3] + np.dot(fit[:3], fit[:3]))
    if not np.isfinite(squared) or squared <= 0:
        return _failure('sphere_nonpositive_radius')
    center = mean + fit[:3]
    radius = np.sqrt(squared)
    radial = p - center
    radii = np.linalg.norm(radial, axis=1)
    if np.any(radii <= 1e-12 * radius):
        return _failure('sphere_points_at_center')
    errors = np.abs(radii - radius) * samples['scale']
    maximum = float(errors.max())
    if maximum > tolerance:
        return _failure('sphere_vertex_residual_exceeds_tolerance', max_vertex_residual_mm=maximum)

    directions = radial / radii[:, None]
    mean_direction = np.einsum('i,ij->j', vw, directions)
    centered = directions - mean_direction
    values, vectors = np.linalg.eigh(np.einsum('i,ij,ik->jk', vw, centered, centered))
    # An annular fillet remainder or a tiny spherical patch does not identify a
    # well-observed sphere, even if a permissive length tolerance lets it fit.
    span = np.ptp(directions @ vectors, axis=0)
    spread = np.rad2deg(2 * np.arcsin(np.clip(span / 2, 0, 1)))
    if (values[0] < .01 * values[-1] or
            min(spread[-2:]) < _MIN_ANGLE_DEGREES - 1e-3):
        return _failure('sphere_insufficient_two_dimensional_arc_extent',
                        radial_direction_spread_degrees=spread.tolist())

    triangles = radial[samples['inverse']]
    centroids = triangles.mean(axis=1)
    centroid_radii = np.linalg.norm(centroids, axis=1)
    if np.any(centroid_radii <= 1e-12 * radius):
        return _failure('sphere_triangle_spans_center')
    evidence = _normal_evidence(samples, centroids / centroid_radii[:, None],
                                directions[samples['inverse']])
    if not evidence['normal_consistency_within_resolution']:
        return _failure('sphere_normals_inconsistent_with_radial_support', **evidence)

    # Filled triangles, not just vertices or centroids. This vectorized routine
    # requires neither an rtree nor any nearest-surface query.
    closest = trimesh.triangles.closest_point(triangles, np.zeros((len(region), 3)))
    minimum = np.linalg.norm(closest, axis=1)
    maximum_radii = radii[samples['inverse']].max(axis=1)
    surface_error = float(np.max(np.maximum(np.abs(minimum - radius),
                                            np.abs(maximum_radii - radius))) * samples['scale'])
    origin = data['anchor'] + (samples['center'] + center * samples['scale'])
    record = _candidate_base(data, region, 'sphere', origin, np.eye(3))
    record.update(evidence)
    record.update({
        'center_mm': origin.tolist(), 'radius_mm': float(radius * samples['scale']),
        'parameterization': ('p(theta,phi) = center_mm + radius_mm * '
                             '(cos(phi)*cos(theta)*x_direction + '
                             'cos(phi)*sin(theta)*y_direction + sin(phi)*z_direction)'),
        'frame_orientation_basis': 'source_axes_not_inferred_design_orientation',
        'fit_method': 'conditioned_area_weighted_algebraic_sphere',
        'fit_weighting': 'triangle_area_lumped_equally_to_incident_vertices',
        'fit_condition_number': condition,
        'radial_direction_spread_degrees': spread.tolist(),
        'radial_direction_covariance_eigenvalues': values.tolist(),
        'max_radial_error_mm': maximum, 'max_vertex_residual_mm': maximum,
        'fit_rms_residual_mm': float(np.sqrt(np.dot(vw, errors ** 2))),
        'max_centroid_residual_mm': float(np.max(np.abs(centroid_radii - radius)) * samples['scale']),
        'max_surface_residual_mm': surface_error,
        'max_mesh_sagitta_mm': float(np.maximum(radius - minimum, 0).max() * samples['scale']),
        'surface_residual_method': 'exact_radial_extrema_over_filled_triangles_using_origin_closest_point',
        'surface_residual_is_exact': True,
        'surface_residual_scope': 'one_sided_mesh_to_untrimmed_analytic_support',
        'vertex_fit_within_tolerance': True,
        'surface_approximation_within_tolerance': bool(surface_error <= tolerance),
        'fit_tolerance_mm': tolerance,
    })
    return record, None


def _cone_coordinates(p, center, axis):
    q = p - center
    z = q @ axis
    radial = q - z[..., None] * axis
    return z, radial, np.linalg.norm(radial, axis=-1)


def _cone_distance(radii, z, radius, slope):
    """Exact POINT distance to the nonnegative-radius branch, including apex."""
    target = radius + slope * z
    line_distance = np.abs(radii - target) / np.sqrt(1 + slope * slope)
    projected_radius = (target + slope * slope * radii) / (1 + slope * slope)
    return np.where(projected_radius >= 0, line_distance,
                    np.hypot(radii, z + radius / slope))


def _cone_intervals(triangle_xy, epsilon):
    """An apex has no azimuth; give it the mid-direction of its other vertices."""
    radii = np.linalg.norm(triangle_xy, axis=2)
    directions = np.divide(triangle_xy, radii[:, :, None],
                           out=np.zeros_like(triangle_xy), where=radii[:, :, None] > epsilon)
    summed = directions.sum(axis=1)
    length = np.linalg.norm(summed, axis=1)
    if np.any(length <= 1e-12):
        return None, None
    directions = np.where((radii <= epsilon)[:, :, None],
                          (summed / length[:, None])[:, None, :], directions)
    angles = np.sort(np.mod(np.arctan2(directions[:, :, 1], directions[:, :, 0]), 2 * np.pi), axis=1)
    gaps = np.diff(np.column_stack((angles, angles[:, 0] + 2 * np.pi)), axis=1)
    widths = 2 * np.pi - gaps.max(axis=1)
    return _angular_intervals(directions), widths


def _cone(data, region, samples, tolerance):
    ns, weights = samples['normals'], samples['weights']
    mean_normal = np.einsum('i,ij->j', weights, ns)
    centered = ns - mean_normal
    values, vectors = np.linalg.eigh(np.einsum('i,ij,ik->jk', weights, centered, centered))
    if values[1] <= 1e-8 or values[0] > .002 * values[1]:
        return _failure('cone_normals_do_not_define_affine_axis')
    initial_axis = _canonical(vectors[:, 0])
    dots = ns @ initial_axis
    offset = float(np.dot(weights, dots))
    spread = float(np.sqrt(np.dot(weights, (dots - offset) ** 2)))
    triangles_mm = samples['triangles'] * samples['scale']
    minimum_edge = float(np.linalg.norm(np.roll(triangles_mm, -1, axis=1) - triangles_mm, axis=2).min())
    angular_numerical = 8 * samples['uncertainty_mm'] / minimum_edge
    if abs(offset) <= max(1e-10, 16 * spread):
        return _failure('cone_nonzero_taper_not_resolved_cylinder_or_plane')
    if (spread > .06 * abs(offset) + angular_numerical or
            np.max(np.abs(dots - offset)) > .2 * abs(offset) + 4 * angular_numerical):
        return _failure('cone_mixed_or_nonconical_normal_offsets',
                        affine_normal_offset=offset, affine_normal_rms=spread)

    u, v = _basis(initial_axis)
    basis = np.column_stack((u, v))
    p, vw = samples['points'], samples['vertex_weights']
    # Intersect area-weighted projected normal lines for a deterministic axis
    # center seed. For complete planar panels the tangential centroid biases of
    # their two triangles cancel; unlike a single circle fit this handles taper.
    radial_normals = ns @ basis
    radial_lengths = np.linalg.norm(radial_normals, axis=1)
    if np.any(radial_lengths <= 1e-8):
        return _failure('cone_plane_like_normals')
    tangents = np.column_stack((-radial_normals[:, 1], radial_normals[:, 0])) / radial_lengths[:, None]
    tangent_matrix = tangents * np.sqrt(weights[:, None])
    if np.linalg.cond(tangent_matrix) > 1e6:
        return _failure('cone_axis_center_rank_or_condition_deficient')
    centroid_xy = samples['triangles'].mean(axis=1) @ basis
    circle = np.linalg.lstsq(tangent_matrix,
                            np.einsum('ij,ij->i', tangents, centroid_xy) * np.sqrt(weights),
                            rcond=None)[0]
    z, _, radii = _cone_coordinates(p, basis @ circle, initial_axis)
    linear = np.column_stack((np.ones(len(z)), z)) * np.sqrt(vw[:, None])
    if np.linalg.cond(linear) > 1e6:
        return _failure('cone_axial_fit_rank_or_condition_deficient')
    r0, slope = np.linalg.lstsq(linear, radii * np.sqrt(vw), rcond=None)[0]
    if not np.isfinite([*circle, r0, slope]).all() or r0 <= 0:
        return _failure('cone_invalid_initial_radius')

    def unpack(parameters):
        axis = initial_axis + parameters[4] * u + parameters[5] * v
        axis /= np.linalg.norm(axis)
        return basis @ parameters[:2], axis, parameters[2], parameters[3]

    apex_seed = ((radii <= max(1e-9 * r0, 8 * samples['uncertainty_mm'] / samples['scale'])) &
                 (np.abs(r0 + slope * z) <= max(1e-9 * r0, 8 * samples['uncertainty_mm'] / samples['scale'])))

    def residual(parameters):
        center, axis, radius, taper = unpack(parameters)
        axial, radial_vectors, radial_lengths = _cone_coordinates(p, center, axis)
        ordinary = (radial_lengths - radius - taper * axial)[~apex_seed] * np.sqrt(vw[~apex_seed])
        # rho is nondifferentiable at an apex. Three smooth constraints identify
        # the axis and zero radius there instead of trusting its unstable Jacobian.
        apex = np.column_stack((radial_vectors[apex_seed] @ basis,
                                radius + taper * axial[apex_seed]))
        return np.concatenate((ordinary, (apex * np.sqrt(vw[apex_seed, None])).ravel()))

    initial = np.array([*circle, r0, slope, 0., 0.])
    try:
        fit = least_squares(residual, initial, max_nfev=150,
                            bounds=([-np.inf, -np.inf, 1e-12, -np.inf, -.5, -.5],
                                    [np.inf, np.inf, np.inf, np.inf, .5, .5]),
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
    except (ValueError, FloatingPointError):
        return _failure('cone_nonlinear_fit_failed')
    if not fit.success or not np.isfinite(fit.x).all():
        return _failure('cone_nonlinear_fit_failed')
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    if singular[-1] <= 0 or singular[-1] < 1e-7 * singular[0]:
        return _failure('cone_fit_rank_or_condition_deficient')
    condition = float(singular[0] / singular[-1])
    center, axis, radius, slope = unpack(fit.x)
    canonical_axis = _canonical(axis)
    slope *= float(np.dot(axis, canonical_axis))
    axis = canonical_axis
    z, radial, radii = _cone_coordinates(p, center, axis)
    height = float(np.ptp(z))
    numerical = max(1e-9 * radius * samples['scale'],
                    8 * samples['uncertainty_mm'])
    if height <= 1e-10 or abs(slope) * height * samples['scale'] <= numerical:
        return _failure('cone_nonzero_taper_not_resolved_cylinder_or_plane')
    target = radius + slope * z
    if target.min() < -max(8 * samples['uncertainty_mm'] / samples['scale'], 1e-9 * radius):
        return _failure('cone_region_crosses_wrong_apex_branch')
    errors = _cone_distance(radii, z, radius, slope) * samples['scale']
    maximum = float(errors.max())
    if maximum > tolerance:
        return _failure('cone_vertex_residual_exceeds_tolerance', max_vertex_residual_mm=maximum)

    u, v = _basis(axis)
    basis = np.column_stack((u, v))
    xy = radial @ basis
    epsilon = max(1e-9 * max(radius, radii.max()),
                  8 * samples['uncertainty_mm'] / samples['scale'])
    intervals, widths = _cone_intervals(xy[samples['inverse']], epsilon)
    if intervals is None:
        return _failure('cone_triangle_crosses_axis_or_angular_span_ambiguous')
    coverage = float(np.rad2deg(sum(end - start for start, end in intervals)))
    if coverage < _MIN_ANGLE_DEGREES - 1e-3:
        return _failure('cone_insufficient_angular_extent', angular_coverage_degrees=coverage)

    # A systematic second angular harmonic is ellipticity, not circular-cone
    # evidence made acceptable by a large user tolerance. Ignore apex points;
    # first harmonics and a constant absorb center/radius-fit bias.
    observed = target > 16 * epsilon
    angles = np.arctan2(xy[observed, 1], xy[observed, 0])
    harmonic = np.column_stack((np.ones(len(angles)), np.cos(angles), np.sin(angles),
                                np.cos(2 * angles), np.sin(2 * angles)))
    hw = vw[observed] / vw[observed].sum()
    weighted = harmonic * np.sqrt(hw[:, None])
    harmonic_condition = float(np.linalg.cond(weighted))
    if np.isfinite(harmonic_condition) and harmonic_condition < 1e6:
        relative_errors = (radii[observed] - target[observed]) / target[observed]
        coefficients = np.linalg.lstsq(weighted, relative_errors * np.sqrt(hw), rcond=None)[0]
        variation = float(np.dot(hw, (relative_errors - np.dot(hw, relative_errors)) ** 2))
        remainder = float(np.dot(hw, (relative_errors - harmonic @ coefficients) ** 2))
        amplitude = float(np.linalg.norm(coefficients[-2:]))
        harmonic_numerical = (8 * samples['uncertainty_mm'] * harmonic_condition /
                              (samples['scale'] * np.dot(hw, target[observed])))
        if (amplitude > max(1e-6, harmonic_numerical) and variation > 0 and
                remainder < .005 * variation):
            return _failure('cone_systematic_elliptic_angular_deformation',
                            relative_second_harmonic_amplitude=amplitude)

    centroids = samples['triangles'].mean(axis=1)
    cz, cradial, cradii = _cone_coordinates(centroids, center, axis)
    if np.any(cradii <= epsilon):
        return _failure('cone_triangle_spans_axis')
    expected = (cradial / cradii[:, None] - slope * axis) / np.sqrt(1 + slope * slope)
    vertex_directions = np.divide(radial, radii[:, None], out=np.zeros_like(radial),
                                   where=radii[:, None] > epsilon)
    corner_expected = (vertex_directions[samples['inverse']] - slope * axis) / np.sqrt(1 + slope * slope)
    # The apex's normal is undefined; it must not inflate the resolution gate.
    corner_expected = np.where((radii[samples['inverse']] <= epsilon)[:, :, None],
                               expected[:, None, :], corner_expected)
    evidence = _normal_evidence(samples, expected, corner_expected)
    if not evidence['normal_consistency_within_resolution']:
        return _failure('cone_normals_inconsistent_with_support', **evidence)
    # Recheck AFFINE consistency against the final, geometrically refined axis.
    final_dots = ns @ axis
    final_offset = float(np.dot(weights, final_dots))
    final_spread = float(np.sqrt(np.dot(weights, (final_dots - final_offset) ** 2)))
    if (final_spread > .06 * abs(final_offset) + angular_numerical or
            np.max(np.abs(final_dots - final_offset)) > .2 * abs(final_offset) + 4 * angular_numerical):
        return _failure('cone_refined_axis_normal_offsets_inconsistent')

    # On a triangle, F=rho-r(z) is convex, hence max F is at a vertex.
    # A projected angular sector of half-width d bounds the convexity gap of
    # rho by max_vertex_rho*(1-cos(d)). Thus min F >= min_vertex_F - gap.
    # Near-axis vertices have undefined/unstable azimuth. Their possible vector
    # cancellation is covered separately by twice their maximum radial length.
    # If projection onto the infinite generatrix is behind the apex, add an
    # upper bound on its distance to the apex. This is conservative, not an
    # exact extremum search or a sample-based tolerance certificate.
    inverse = samples['inverse']
    f = (radii - target)[inverse]
    sagitta_bound = radii[inverse].max(axis=1) * (1 - np.cos(widths / 2))
    sagitta_bound += 2 * np.where(radii[inverse] <= epsilon, radii[inverse], 0).max(axis=1)
    line_bound = np.maximum(np.abs(f.max(axis=1)), np.abs(f.min(axis=1) - sagitta_bound))
    minimum_rho = _triangle_radial_minimum(xy[inverse])
    branch_numerator = target[inverse].min(axis=1) + slope * slope * minimum_rho
    branch_bound = np.maximum(-branch_numerator, 0) / abs(slope)
    upper = (line_bound + branch_bound) / np.sqrt(1 + slope * slope) * samples['scale']
    centroid_errors = _cone_distance(cradii, cz, radius, slope) * samples['scale']
    midpoints = (samples['triangles'] + np.roll(samples['triangles'], -1, axis=1)) / 2
    mz, _, mradii = _cone_coordinates(midpoints, center, axis)
    midpoint_errors = _cone_distance(mradii, mz, radius, slope) * samples['scale']
    sampled_max = max(maximum, float(centroid_errors.max()), float(midpoint_errors.max()))
    upper_max = max(float(upper.max()), sampled_max)

    midpoint = float((z.min() + z.max()) / 2)
    axis_point = center + midpoint * axis
    origin = data['anchor'] + (samples['center'] + axis_point * samples['scale'])
    radius_origin = float((radius + slope * midpoint) * samples['scale'])
    height_mm = height * samples['scale']
    axial_range = [-height_mm / 2, height_mm / 2]
    end_radii = radius_origin + slope * np.asarray(axial_range)
    base_index = int(np.argmax(end_radii))
    apex_z = -radius_origin / slope
    apex = origin + apex_z * axis
    if not np.isfinite(apex).all():
        return _failure('cone_nonfinite_apex')
    record = _candidate_base(data, region, 'cone', origin, np.column_stack((u, v, axis)))
    record.update(evidence)
    record.update({
        'axis': axis.tolist(), 'axis_point_mm': origin.tolist(),
        'radius_at_origin_mm': radius_origin,
        'radius_slope_mm_per_mm': float(slope),
        'semi_angle_degrees': float(np.rad2deg(np.arctan(abs(slope)))),
        'base_radius_mm': float(max(0., end_radii[base_index])),
        'base_axial_parameter_mm': float(axial_range[base_index]),
        'radius_range_mm': [float(max(0., end_radii.min())), float(max(0., end_radii.max()))],
        'radius_range_basis': 'nonnegative_branch_of_fitted_axial_interval',
        'axial_length_mm': float(height_mm), 'axial_parameter_range_mm': axial_range,
        'apex_mm': apex.tolist(), 'apex_axial_parameter_mm': float(apex_z),
        'angular_parameter_intervals_radians': intervals,
        'angular_coverage_degrees': min(360., coverage),
        'wraps_full_circumference': bool(coverage >= 360 - 1e-6),
        'parameterization': ('p(theta,z) = frame.origin_mm + '
                             '(radius_at_origin_mm + radius_slope_mm_per_mm*z) * '
                             '(cos(theta)*x_direction + sin(theta)*y_direction) + z*z_direction; '
                             'radius_at_origin_mm + radius_slope_mm_per_mm*z >= 0'),
        'fit_method': 'conditioned_area_weighted_nonlinear_cone_affine_normal_axis_seed',
        'fit_weighting': 'triangle_area_lumped_equally_to_incident_vertices',
        'fit_condition_number': condition,
        'affine_normal_offset': final_offset, 'affine_normal_rms': final_spread,
        'max_radial_error_mm': float(np.abs(radii - target).max() * samples['scale']),
        'max_vertex_residual_mm': maximum,
        'fit_rms_residual_mm': float(np.sqrt(np.dot(vw, errors ** 2))),
        'max_centroid_residual_mm': float(centroid_errors.max()),
        'max_sampled_surface_residual_mm': sampled_max,
        'max_surface_residual_upper_bound_mm': upper_max,
        'max_surface_residual_mm': upper_max,
        'max_mesh_sagitta_upper_bound_mm': float(sagitta_bound.max() * samples['scale']),
        'surface_residual_method': 'sampled_vertices_centroids_edge_midpoints_with_conservative_triangle_upper_bound_not_exact_maximum',
        'surface_residual_is_exact': False,
        'surface_residual_scope': 'one_sided_mesh_to_untrimmed_analytic_support',
        'sampled_surface_within_tolerance': bool(sampled_max <= tolerance),
        'vertex_fit_within_tolerance': True,
        'surface_approximation_within_tolerance': bool(upper_max <= tolerance),
        'surface_tolerance_assessment': ('conservative_upper_bound_within_tolerance' if upper_max <= tolerance else
                                         'sampled_error_exceeds_tolerance' if sampled_max > tolerance else
                                         'not_established_by_conservative_upper_bound'),
        'fit_tolerance_mm': tolerance,
    })
    return record, None


def additional_surface_candidates(mesh, tolerance=.05, exclude_triangle_ids=()):
    """Return ``cones``, ``spheres``, ``rejected_regions`` and exclusion evidence.

    ``exclude_triangle_ids`` contains input-body local face IDs, typically caller
    plane/cylinder evidence. Exclusion splits connectivity; a fit cannot bridge
    an excluded face. The default normal-band extractor drops standalone strict
    coplanar caps/plates while retaining faceted curved panels. At least twelve
    connected triangles and 60 degrees of observed angular extent are required
    (for spheres in two independent directions, also with 3-D radial rank).

    Rejected regions contain independent sphere/cone reason codes. Torus and
    freeform recognition/reconstruction are deliberately unsupported. A passing
    vertex fit may still have ``surface_approximation_within_tolerance=False``.
    Cone ``max_surface_residual_mm`` is an upper bound, NOT an exact maximum;
    ``max_sampled_surface_residual_mm`` is explicitly only sampled evidence.
    """
    tolerance = _nonnegative(tolerance, 'tolerance')
    data = _prepare(mesh)
    try:
        excluded_input = list(exclude_triangle_ids)
    except TypeError as exc:
        raise ValueError('exclude_triangle_ids must contain body-local integer face IDs') from exc
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or
           i < 0 or i >= len(data['faces']) for i in excluded_input):
        raise ValueError('exclude_triangle_ids must contain body-local integer face IDs')
    excluded_input = set(int(i) for i in excluded_input)
    bands = _cylinder_regions(data)
    eligible = set(i for band in bands for i in band) - excluded_input
    threshold = np.cos(np.deg2rad(_CYLINDER_BAND_DEGREES))
    links = [(a, b) for a, b in data['adjacency']
             if a in eligible and b in eligible and np.dot(data['normals'][a], data['normals'][b]) >= threshold]
    regions = _regions(eligible, links)
    result = {'cones': [], 'spheres': [], 'rejected_regions': [],
              'excluded_source_triangle_ids': sorted(set(range(len(data['faces']))) - eligible),
              'caller_excluded_source_triangle_ids': sorted(excluded_input),
              'source_triangle_id_scope': 'body_local_faces',
              'supported_surface_types': ['cone', 'sphere'],
              'normal_band_threshold_degrees': _CYLINDER_BAND_DEGREES,
              'planar_facet_policy': 'exclude_standalone_strict_facets_retain_curved_band_panels',
              'candidate_geometry_only': True, 'cad_reconstructed': False,
              'requires_review': True}
    for region in regions:
        if len(region) < _MIN_TRIANGLES:
            sphere_error = cone_error = {'reason': 'insufficient_connected_triangles',
                                        'minimum_triangle_count': _MIN_TRIANGLES}
        else:
            samples = _samples(data, region)
            sphere, sphere_error = _sphere(data, region, samples, tolerance)
            cone = None
            if sphere is None:
                cone, cone_error = _cone(data, region, samples, tolerance)
            candidate = sphere if sphere is not None else cone
            if candidate is not None:
                candidate.update({'source_coordinate_resolution_mm': samples['uncertainty_mm'],
                                  'requires_review': True})
                result['spheres' if sphere is not None else 'cones'].append(candidate)
                continue
        result['rejected_regions'].append({
            'region_id': f'additional_rejected_{min(region):06d}',
            'source_triangle_ids': [int(i) for i in region],
            'source_triangle_id_scope': 'body_local_faces',
            'triangle_count': len(region), 'area_mm2': float(data['areas'][region].sum()),
            'fit_failures': {'sphere': sphere_error, 'cone': cone_error},
            'candidate_geometry_only': True,
        })
    for kind in ('cones', 'spheres'):
        result[kind].sort(key=lambda c: (-c['area_mm2'], c['candidate_id']))
        for candidate in result[kind]:
            candidate.update({'normal_band_threshold_degrees': _CYLINDER_BAND_DEGREES,
                              'planar_facet_policy': result['planar_facet_policy']})
    return result
