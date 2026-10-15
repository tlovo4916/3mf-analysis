"""Validate plate plans offline, or build reviewed native features on Windows.

Requires installed, licensed SolidWorks 2024 and pywin32 for building only. No
existing document is edited/closed and no existing part or receipt is overwritten.
The COM feature calls still require testing in real SolidWorks; offline validation
(and mocked unit tests) cannot verify those calls or source-geometry equivalence.
"""
import argparse
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from numbers import Real
from pathlib import Path
import sys

from shapely.errors import GEOSException
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.validation import explain_validity

from slicer_metadata import source_role_blockers


_STRATEGY = 'review_required_plate_profile'
_CIRCLE_QUAD_SEGS = 256  # 1024 chords; never use their area as the true circle area.
_VOLUME_RELATIVE_TOLERANCE = .002
_VOLUME_ROUNDOFF_ULPS = 32
_LEGACY_WARNING = (
    'Legacy schema v1: approval does not include the mandatory v2 approximation '
    'acceptance/blocker fields. Review all analysis limitations manually; this '
    'receipt does not establish source-geometry equivalence.'
)


@dataclass(frozen=True)
class _CutShape:
    id: str
    polygon: Polygon
    area_mm2: float
    chord_error_mm: float = 0.0


def _number(value, label, positive=False):
    # bool is a subclass of int, but is never a geometric quantity.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{label} must be a finite number (not bool)')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{label} must be finite') from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f'{label} must be finite' + (' and positive' if positive else ''))
    return result


def _xy(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{label} must contain two numeric coordinates')
    return tuple(_number(v, label) for v in value)


def _valid_polygon(polygon, label):
    if polygon.is_empty or not math.isfinite(polygon.area) or polygon.area <= 0:
        raise ValueError(f'{label} is empty or degenerate')
    if not polygon.is_valid or not polygon.exterior.is_simple:
        raise ValueError(f'{label} is invalid/self-intersecting: {explain_validity(polygon)}')
    return polygon


def _polygon(points, label):
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        raise ValueError(f'{label} needs at least three points')
    coordinates = [_xy(p, label) for p in points]
    # Plans use open point lists. A repeated closing point would cause a zero-
    # length CreateLine, as would consecutive/repeated vertices elsewhere.
    if len(set(coordinates)) != len(coordinates):
        raise ValueError(f'{label} has repeated vertices; use an open, nondegenerate ring')
    return _valid_polygon(Polygon(coordinates), label)


def _schema(plan):
    if not isinstance(plan, dict):
        raise ValueError('Plan must be an object')
    version = plan.get('schema_version')
    if type(version) is not int or version not in (1, 2) or plan.get('strategy') != _STRATEGY:
        raise ValueError('Unsupported reconstruction plan schema/strategy')
    return version


def _boolean(plan, name, required):
    if name not in plan and not required:
        return None
    value = plan.get(name)
    if type(value) is not bool:
        raise ValueError(f'{name} must be a boolean')
    return value


def source_build_blockers(provenance) -> list[str]:
    """Return source vetoes without mutating or interpreting geometry/approval.

    Missing metadata is allowed for legacy plans and synthetic callers. When a
    field is present its type is strict; malformed provenance is itself a veto.
    Removing a copied top-level blocker list cannot override source evidence.
    New slicer provenance is a strict bundle: raw role declarations and each
    leaf's occurrence/face-range evidence independently recompute source vetoes.
    normal_part removes no existing veto and is never approval to build.
    """
    if not isinstance(provenance, dict):
        return ['source_provenance must be an object']
    blockers = []

    def add(reason):
        if reason not in blockers:
            blockers.append(reason)

    for name, reason in (
        ('reconstruction_eligible', 'reconstruction_eligible=false'),
        ('printable', 'non_printable_build_item'),
        ('document_coverage_complete', 'document_coverage_complete=false'),
    ):
        if name in provenance:
            value = provenance[name]
            if type(value) is not bool:
                add(f'source_provenance.{name} must be a boolean')
            elif value is False:
                add(reason)

    for name in ('reconstruction_blockers', 'core_spec_violations', 'effective_object_types'):
        if name not in provenance:
            continue
        values = provenance[name]
        if not isinstance(values, list):
            add(f'source_provenance.{name} must be a list of nonempty strings')
            continue
        for value in values:
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                add(f'source_provenance.{name} must be a list of nonempty strings')
            elif name == 'effective_object_types':
                if value != 'model':
                    add('non_model_object_type:' + value)
            else:
                add(value)

    if 'effective_object_type' in provenance:
        value = provenance['effective_object_type']
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            add('source_provenance.effective_object_type must be a nonempty string')
        elif value != 'model':
            add('non_model_object_type:' + value)
    for reason in source_role_blockers(provenance):
        add(reason)
    return blockers


def _review_gate(plan, version):
    blockers = plan.get('build_blockers', [] if version == 1 else None)
    if not isinstance(blockers, list):
        raise ValueError('build_blockers must be a list (required for schema v2)')
    if blockers:
        raise ValueError(f'Unresolved build_blockers; approval cannot override them: {blockers!r}')
    if 'source_provenance' in plan:
        source_blockers = source_build_blockers(plan['source_provenance'])
        if source_blockers:
            raise ValueError('Unresolved source_provenance build blockers; approval cannot override them: '
                             + repr(source_blockers))
    placement = _boolean(plan, 'source_placement_applied', required=version == 2)
    if placement is True:
        raise ValueError('Source placement is unsupported: output uses local sketch coordinates')
    needed = _boolean(plan, 'requires_approximation_acceptance', required=version == 2)
    accepted = _boolean(plan, 'accepted_approximation', required=version == 2)
    if version == 2 and needed and not accepted:
        raise ValueError('Approximation requires accepted_approximation=true after review')
    if _boolean(plan, 'approved_for_build', required=True) is not True:
        raise ValueError('先阅读分析报告，确认差异后将 approved_for_build 设为 true。')


def _geometry_volume_mm3(plan):
    parameters = plan.get('parameters')
    if not isinstance(parameters, dict):
        raise ValueError('parameters must be an object')
    thickness = _number(parameters.get('thickness_mm'), 'thickness_mm', positive=True)
    outer = _polygon(plan.get('outer_points_mm'), 'Outer polygon')
    cuts = plan.get('cuts')
    if not isinstance(cuts, list):
        raise ValueError('cuts must be a list')

    # Numerical safety margin in mm, separate from the circle chord error.
    scale = max(1.0, *(abs(v) for v in outer.bounds))
    clearance = max(1e-9, 32 * math.ulp(scale))
    ids = set()
    shapes = []
    for cut in cuts:
        if not isinstance(cut, dict):
            raise ValueError('Each cut must be an object')
        cut_id = cut.get('id')
        if not isinstance(cut_id, str) or not cut_id.strip() or cut_id != cut_id.strip():
            raise ValueError('Cut ID must be a nonempty string without surrounding whitespace')
        if cut_id in ids:
            raise ValueError('Duplicate cut ID: ' + cut_id)
        ids.add(cut_id)
        if cut.get('type') == 'circle':
            center = _xy(cut.get('center_mm'), cut_id + '.center_mm')
            diameter = _number(cut.get('diameter_mm'), cut_id + '.diameter_mm', positive=True)
            radius = _number(diameter / 2, cut_id + '.radius_mm', positive=True)
            area = _number(math.pi * radius * radius, cut_id + '.area_mm2', positive=True)
            polygon = _valid_polygon(
                Point(center).buffer(radius, quad_segs=_CIRCLE_QUAD_SEGS), cut_id
            )
            # An inscribed buffer alone can miss tangency/overlap between vertices.
            # Its Hausdorff error to the true disk is this maximum chord sagitta.
            chord_error = radius * (1 - math.cos(math.pi / (4 * _CIRCLE_QUAD_SEGS)))
            shape = _CutShape(cut_id, polygon, area, chord_error)
        elif cut.get('type') == 'polygon':
            polygon = _polygon(cut.get('points_mm'), cut_id + ' polygon')
            shape = _CutShape(cut_id, polygon, polygon.area)
        else:
            raise ValueError(f'Unsupported cut type: {cut.get("type")!r}')
        gap = outer.boundary.distance(shape.polygon)
        if (not outer.contains(shape.polygon) or not math.isfinite(gap)
                or gap <= shape.chord_error_mm + clearance):
            raise ValueError(f'Cut {cut_id} must be strictly inside outer polygon, with safe clearance')
        shapes.append(shape)

    # Each pair is checked once. No circle fitting or geometry is recomputed.
    for a, b in combinations(shapes, 2):
        gap = a.polygon.distance(b.polygon)
        if (a.polygon.intersects(b.polygon) or not math.isfinite(gap)
                or gap <= a.chord_error_mm + b.chord_error_mm + clearance):
            raise ValueError(f'Cuts {a.id} and {b.id} intersect, contain, overlap or touch')

    remainder = outer.difference(unary_union([s.polygon for s in shapes])) if shapes else outer
    if remainder.geom_type != 'Polygon':
        raise ValueError('Cuts must leave exactly one connected plate, not multiple solids')
    _valid_polygon(remainder, 'Remaining plate')
    # Circle area is analytic, not the inscribed validation polygon's area.
    area = outer.area - math.fsum(s.area_mm2 for s in shapes)
    return _number(area * thickness, 'Current plan volume_mm3', positive=True)


def _validated_plan_volume(plan, require_review=True):
    version = _schema(plan)
    if require_review:
        _review_gate(plan, version)
    try:
        return _geometry_volume_mm3(plan)
    except GEOSException as exc:
        raise ValueError('Geometry cannot be validated safely') from exc


def validate(plan):
    """Validate geometry and build-review gates; return None for v1 compatibility."""
    _validated_plan_volume(plan)


def plan_volume_mm3(plan):
    """Volume of current geometric parameters only, even before human approval.

    Schema and geometry must be valid; review/acceptance/blocker gates are NOT
    evaluated. This estimate is not authorization to build, a source-volume
    comparison, or evidence of source-geometry equivalence.
    """
    return _validated_plan_volume(plan, require_review=False)


def _built_volume_check(actual, expected, label, *, numerical_scale=None, measurement_bound=0.0):
    actual = _number(actual, label + ' actual_volume_mm3')
    expected = _number(expected, label + ' expected_volume_mm3', positive=True)
    scale = max(abs(actual), expected)
    if numerical_scale is not None:
        scale = max(scale, _number(numerical_scale, label + ' numerical_scale', positive=True))
    # No absolute tolerance floor: it could hide a complete cut in a small part.
    # This is a floating-point roundoff budget, not a CAD-kernel accuracy claim.
    numerical_bound = max(_VOLUME_ROUNDOFF_ULPS * math.ulp(scale), measurement_bound)
    allowed_error = expected * _VOLUME_RELATIVE_TOLERANCE
    if allowed_error <= 0 or not math.isfinite(numerical_bound) or numerical_bound > allowed_error:
        raise RuntimeError(f'Volume validation unable to resolve {label}: numerical bound '
                           f'{numerical_bound:.17g} exceeds 0.2% budget {allowed_error:.17g} mm3')
    error = abs(actual - expected)
    if actual <= 0 or error > allowed_error:
        raise RuntimeError(f'Volume validation failed for {label}: actual {actual:.17g}, '
                           f'current plan {expected:.17g} mm3 (0.2% relative tolerance). '
                           'Check extrusion, cut direction and openings.')
    return {
        'actual_volume_mm3': actual, 'expected_volume_mm3': expected,
        'absolute_error_mm3': error, 'allowed_error_mm3': allowed_error,
        'relative_tolerance': _VOLUME_RELATIVE_TOLERANCE,
        'numerical_bound_mm3': numerical_bound,
        'scope': 'current_plan_parameters_only', 'status': 'passed',
    }


def validate_built_volume(actual, expected, label='Solid body volume'):
    """Check mm3 against the current plan with 0.2% relative tolerance only.

    Return receipt details on success; reject nonfinite/nonpositive quantities,
    mismatches and scales whose ULP budget cannot resolve the tolerance. This
    checks neither source equivalence nor whether real COM calls have executed.
    """
    return _built_volume_check(actual, expected, label)


def _feature_volumes_mm3(plan):
    """Analytic base/cut volumes of already-validated current parameters."""
    thickness = plan['parameters']['thickness_mm']
    base = _number(Polygon(plan['outer_points_mm']).area * thickness,
                   'Base extrusion volume_mm3', positive=True)
    cuts = []
    for cut in plan['cuts']:
        if cut['type'] == 'circle':
            radius = cut['diameter_mm'] / 2
            area = math.pi * radius * radius
        else:
            area = Polygon(cut['points_mm']).area
        cuts.append(_number(area * thickness, cut['id'] + ' removed_volume_mm3', positive=True))
    return base, cuts


def _receipt_base(plan, volume):
    try:
        canonical = json.dumps(plan, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ValueError('Receipt requires a finite, JSON-serializable plan') from exc
    return {
        'schema_version': plan['schema_version'],
        'plan_sha256': hashlib.sha256(canonical).hexdigest(),
        'plan_sha256_basis': 'canonical JSON UTF-8; sorted keys; compact separators; ensure_ascii=False',
        'target_solidworks_year': 2024,
        'required_solidworks_major_revision': 32,
        'output_coordinates': 'local_sketch_plane_mm',
        'source_placement_applied': False,
        'placement_note': 'First reference plane of new part template; source rotation/origin not applied.',
        'plan_volume_mm3': volume,
        'plan_volume_scope': 'current_plan_parameters_only',
        'volume_note': 'Analytic net profile area times current thickness; not source-geometry equivalence.',
        'source_geometry_equivalence_verified': False,
        'warnings': [_LEGACY_WARNING] if plan['schema_version'] == 1 else [],
        'circle_validation_quad_segs': _CIRCLE_QUAD_SEGS,
        'circle_clearance_policy': 'conservative chord-error plus numerical margin',
    }


def offline_validation_receipt(plan):
    """No COM, output part, or software verification is implied by this receipt."""
    receipt = _receipt_base(plan, _validated_plan_volume(plan))
    receipt.update(receipt_type='offline_validation', solidworks_runtime_verified=False,
                   actual_solidworks_revision=None)
    return receipt


def _check_output_paths(output):
    if output.suffix.lower() != '.sldprt':
        raise ValueError('Output must end in .sldprt')
    receipt = output.with_suffix('.build.json')
    for path in (output, receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Refusing to overwrite existing part or receipt: {path}')
    return receipt


def build(plan, output):
    expected = _validated_plan_volume(plan)
    receipt = _receipt_base(plan, expected)
    expected_base, expected_cuts = _feature_volumes_mm3(plan)
    # Do not resolve a final-component symlink away before checking it.
    output = Path(output).absolute()
    receipt_path = _check_output_paths(output)
    if sys.platform != 'win32':
        raise RuntimeError('SolidWorks build requires Windows; use --validate-only for offline validation')
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    try:
        app = win32com.client.Dispatch('SldWorks.Application')
        revision = app.RevisionNumber()
        parts = revision.split('.') if isinstance(revision, str) else []
        if not parts or not all(p.isdigit() for p in parts) or int(parts[0]) != 32:
            raise RuntimeError(f'SolidWorks 2024 (major revision 32) required; actual revision: {revision!r}')
        app.Visible = True
        # swDefaultTemplatePart = 1; use user's configured part template.
        template = app.GetUserPreferenceStringValue(1)
        if not template:
            raise RuntimeError('请先在 SolidWorks 配置默认零件模板。')
        doc = app.NewDocument(template, 0, 0, 0)
        if doc is None:
            raise RuntimeError('Cannot create new SolidWorks part')
        # Only this new document is accessed. Never close/activate an old document.
        feature = doc.FirstFeature()
        plane = None
        while feature is not None:
            if feature.GetTypeName2() == 'RefPlane':
                plane = feature
                break
            feature = feature.GetNextFeature()
        if plane is None:
            raise RuntimeError('No reference plane in part template')
        sm = doc.SketchManager
        fm = doc.FeatureManager

        def sketch(name, points=None, circle=None):
            doc.ClearSelection2(True)
            if not plane.Select2(False, 0):
                raise RuntimeError('Cannot select reference plane')
            sm.InsertSketch(True)
            active = sm.ActiveSketch
            if active is None:
                raise RuntimeError('Cannot start sketch')
            if points is not None:
                for a, b in zip(points, points[1:] + points[:1]):
                    line = sm.CreateLine(a[0]/1000, a[1]/1000, 0, b[0]/1000, b[1]/1000, 0)
                    if line is None:
                        raise RuntimeError('Failed to create sketch line')
            else:
                x, y = circle['center_mm']
                r = circle['diameter_mm']/2000
                if sm.CreateCircleByRadius(x/1000, y/1000, 0, r) is None:
                    raise RuntimeError('Failed to create hole circle')
            sf = active.GetFeature()
            sf.Name = name
            sm.InsertSketch(True)
            doc.ClearSelection2(True)
            if not sf.Select2(False, 0):
                raise RuntimeError('Cannot select completed sketch')

        def rebuilt_body_volume(label):
            if not doc.ForceRebuild3(False):
                raise RuntimeError('SolidWorks rebuild reported failure: ' + label)
            bodies = doc.GetBodies2(0, False)  # swSolidBody; exactly one at EVERY phase.
            if bodies is None or len(bodies) != 1:
                raise RuntimeError('Expected exactly one solid body: ' + label)
            mass_properties = bodies[0].GetMassProperties(1.0)
            if mass_properties is None:
                raise RuntimeError('Cannot read solid mass properties: ' + label)
            try:
                native_volume = mass_properties[3]
            except (IndexError, KeyError, TypeError) as exc:
                raise RuntimeError('Cannot read solid mass properties: ' + label) from exc
            native_volume = _number(native_volume, label + ' volume_m3', positive=True)
            volume = _number(native_volume * 1e9, label + ' volume_mm3', positive=True)
            # Preserve native-unit precision too: subnormal m3 values can lose
            # resolution even when their converted mm3 values are not subnormal.
            bound = _VOLUME_ROUNDOFF_ULPS * math.ulp(native_volume) * 1e9
            return volume, bodies, bound

        sketch('Outline_Profile', points=plan['outer_points_mm'])
        thickness = plan['parameters']['thickness_mm']/1000
        base = fm.FeatureExtrusion2(True, False, False, 0, 0, thickness, 0, False, False, False, False,
                                    0, 0, False, False, False, False, True, True, True, 0, 0, False)
        if base is None:
            raise RuntimeError('Base extrusion failed')
        base.Name = 'Plate_Thickness'
        volume, bodies, measurement_bound = rebuilt_body_volume(base.Name)
        feature_volume_checks = [{
            'feature': base.Name, 'type': 'base_extrusion', 'solid_bodies': len(bodies),
            **_built_volume_check(volume, expected_base, base.Name, measurement_bound=measurement_bound),
        }]
        for cut_plan, expected_cut in zip(plan['cuts'], expected_cuts):
            previous_volume, previous_bound = volume, measurement_bound
            if cut_plan['type'] == 'polygon':
                sketch(cut_plan['id'] + '_Profile', points=cut_plan['points_mm'])
            else:
                sketch(cut_plan['id'] + '_Profile', circle=cut_plan)
            # Keep the existing FeatureCut3 signature/direction unchanged. Blind
            # cut reversed from the original plane, longer than plate thickness.
            # Real SolidWorks testing is still needed to confirm COM behavior.
            cut = fm.FeatureCut3(True, False, True, 0, 0, thickness+.001, thickness+.001,
                                 False, False, False, False, 0, 0, False, False, False, False,
                                 False, True, True, False, False, False, 0, 0, False)
            if cut is None:
                raise RuntimeError('Cut failed: ' + cut_plan['id'])
            cut.Name = cut_plan['id']
            volume, bodies, measurement_bound = rebuilt_body_volume(cut_plan['id'])
            removed = previous_volume - volume
            check = _built_volume_check(
                removed, expected_cut, cut_plan['id'] + ' removed volume',
                numerical_scale=previous_volume,
                measurement_bound=previous_bound + measurement_bound,
            )
            feature_volume_checks.append({
                'feature': cut_plan['id'], 'type': 'cut', 'solid_bodies': len(bodies),
                'before_volume_mm3': previous_volume, 'after_volume_mm3': volume,
                'actual_removed_volume_mm3': removed, 'expected_removed_volume_mm3': expected_cut,
                **check,
            })
        # Per-feature checks prevent a missed small cut or compensating errors
        # from passing this net-volume check. Neither check compares source mesh.
        final_volume_check = _built_volume_check(volume, expected, 'Final solid body volume',
                                                measurement_bound=measurement_bound)
        doc.ViewZoomtofit2()
        errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        warnings = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        output.parent.mkdir(parents=True, exist_ok=True)
        _check_output_paths(output)  # Recheck immediately before SaveAs.
        ok = doc.Extension.SaveAs(str(output), 0, 1, None, errors, warnings)
        if not ok or errors.value or not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f'Save failed: errors={errors.value}, warnings={warnings.value}')
        receipt.update(
            receipt_type='solidworks_build', solidworks_runtime_verified=True,
            actual_solidworks_revision=revision, output=str(output), solid_bodies=len(bodies),
            volume_mm3=volume, expected_volume_mm3=expected, save_errors=errors.value,
            save_warnings=warnings.value, native_features=True, fully_constrained_sketches=False,
            feature_volume_checks=feature_volume_checks,
            feature_volume_checks_scope='current_plan_parameters_only',
            feature_volume_checks_note='Analytic base and each cut removal; not source-geometry equivalence.',
            final_volume_check=final_volume_check,
        )
        # Exclusive create prevents overwriting a receipt even if it appeared
        # during SaveAs. Do not remove the saved part or any user's document.
        with receipt_path.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    finally:
        pythoncom.CoUninitialize()
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description='离线校验审核计划，或在本机 SolidWorks 2024 创建板件')
    parser.add_argument('plan')
    parser.add_argument('--output', help='New .sldprt path; neither part nor .build.json may already exist')
    parser.add_argument('--validate-only', action='store_true', help='Print an offline receipt; no COM or files written')
    args = parser.parse_args(argv)
    if args.validate_only and args.output:
        parser.error('--output is for building only; --validate-only prints its receipt to stdout')
    if not args.validate_only and not args.output:
        parser.error('--output is required for building')
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    if args.validate_only:
        print(json.dumps(offline_validation_receipt(plan), ensure_ascii=False, indent=2))
    else:
        build(plan, args.output)


if __name__ == '__main__':
    main()
