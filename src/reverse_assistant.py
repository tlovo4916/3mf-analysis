"""Reviewable plate plans backed by lossless source and multi-section evidence.

A sampled section is not a design sketch, and a plan is not a reconstructed CAD
part. Unsupported topology is refused, never flattened by dropping inner islands.
"""
import argparse
import csv
import hashlib
import gzip
import html
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon

from threemf_io import UNITS, load_3mf, load_document, transform
from mesh_evidence import map_source_indices, mesh_quality, split_bodies
from geometry_analysis import plate_frame
from section_analysis import analyze_sections, section_at, _bounded_boundary_hausdorff
from build_solidworks import plan_volume_mm3, source_build_blockers


def validate_tolerance(tolerance):
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float, np.number)):
        raise ValueError('tolerance must be a finite number in (0, .5] mm')
    if not np.isfinite(tolerance) or not 0 < tolerance <= .5:
        raise ValueError('tolerance must be a finite number in (0, .5] mm')
    return float(tolerance)


def source_hash(source):
    digest = hashlib.sha256()
    with Path(source).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value, *, compact=False):
    options = {'ensure_ascii': False, 'allow_nan': False}
    options.update({'separators': (',', ':')} if compact else {'indent': 2})
    Path(path).write_text(json.dumps(value, **options), encoding='utf-8')


def prepare_output(out):
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('Refusing to mix a new analysis with existing output: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    return out


def fit_circle(points, tolerance):
    """Conditioned full-loop fit, with exact radial extrema on all polyline edges.

    Vertex residual alone cannot certify a triangulated circular boundary. Coarse
    loops are retained as polygons when their chord sagitta exceeds tolerance.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Circle fit tolerance must be finite and nonnegative')
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or len(p) < 6 or not np.isfinite(p).all():
        return None
    p = p[:, :2]
    if np.array_equal(p[0], p[-1]):
        p = p[:-1]
    polygon = Polygon(p)
    if not polygon.is_valid or polygon.area <= 0:
        return None
    reference = p[0]
    offsets = p - reference
    mean = offsets.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum((offsets - mean) ** 2, axis=1))))
    if scale <= 0 or not np.isfinite(scale):
        return None
    q = (offsets - mean) / scale
    matrix = np.column_stack((2 * q, np.ones(len(q))))
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1e8:
        return None
    fit = np.linalg.lstsq(matrix, np.sum(q * q, axis=1), rcond=None)[0]
    squared = float(fit[2] + fit[:2] @ fit[:2])
    if squared <= 0:
        return None
    center_offset = mean + fit[:2] * scale
    radius = float(np.sqrt(squared) * scale)
    radial = offsets - center_offset
    vertex_error = float(np.max(np.abs(np.linalg.norm(radial, axis=1) - radius)))
    next_radial = np.roll(radial, -1, axis=0)
    edges = next_radial - radial
    squared_length = np.sum(edges * edges, axis=1)
    fractions = np.divide(-np.sum(radial * edges, axis=1), squared_length,
                          out=np.zeros(len(radial)), where=squared_length > 0)
    nearest = radial + np.clip(fractions, 0, 1)[:, None] * edges
    boundary_error = max(vertex_error, float(np.max(np.abs(np.linalg.norm(nearest, axis=1) - radius))))
    if (radius <= 0 or boundary_error > tolerance
            or abs(polygon.area / (np.pi * radius * radius) - 1) >= .025):
        return None
    return {'center_mm': (reference + center_offset).tolist(), 'diameter_mm': 2 * radius,
            'radial_error_mm': vertex_error, 'max_boundary_radial_error_mm': boundary_error,
            'fit_condition_number': condition, 'fit_scope': 'closed_section_polyline_not_design_hole'}


def section_polygons(mesh, z):
    """Legacy loop API, now with strict closure and containment validation."""
    evidence = section_at(mesh, z)
    if evidence.geometry.is_empty:
        raise ValueError('No closed section found')
    return sorted((Polygon(loop['points_mm']) for loop in evidence.metadata['loops']),
                  key=lambda polygon: polygon.area, reverse=True)


def local_mesh(mesh, rotation, origin):
    # Preserve the complete source face array, including explicitly flagged degeneracy.
    return trimesh.Trimesh(vertices=(mesh.vertices - origin) @ rotation,
                           faces=mesh.faces.copy(), process=False)


def section_stack_json(stack):
    return {**stack.metadata, 'samples': [sample.metadata for sample in stack.samples]}


def write_section_evidence(evidence, out):
    """Small navigable summary plus lossless compressed full evidence.

    Hundreds of thousands of sampled coordinates/lineage entries should not be
    expanded to a 150 MB pretty-printed JSON. The compressed file contains the
    complete original JSON object, not rounded or decimated geometry.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    full_path = out / 'sections.full.json.gz'
    with gzip.open(full_path, 'wt', encoding='utf-8', compresslevel=6) as stream:
        json.dump(evidence, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    summary = {key: value for key, value in evidence.items()
               if key not in ('samples', 'failed_samples', 'event_probes')}
    # Probe coordinates/lineage stay lossless in gzip; the navigable summary
    # must not expand another large full evidence tree for every event side.
    summary['event_probes'] = []
    for event_index, event in enumerate(evidence.get('event_probes', [])):
        entry = {key: value for key, value in event.items() if key != 'sides'}
        entry['sides'] = []
        for side_index, side in enumerate(event['sides']):
            side_summary = {key: value for key, value in side.items() if key != 'attempts'}
            side_summary['attempts'] = [
                {key: value for key, value in attempt.items() if key != 'evidence'}
                | {'full_evidence_json_pointer': f'/event_probes/{event_index}/sides/{side_index}/attempts/{attempt_index}/evidence'}
                for attempt_index, attempt in enumerate(side['attempts'])]
            entry['sides'].append(side_summary)
        summary['event_probes'].append(entry)
    summary['failed_samples'] = [
        {key: value for key, value in failure.items() if key != 'evidence'}
        for failure in evidence['failed_samples']]
    summary['full_evidence'] = {'file': full_path.name, 'encoding': 'gzip_utf8_json',
                                'geometry_and_lineage_lossless': True,
                                'sha256': source_hash(full_path)}
    summary['samples'] = []
    keys = ('z_mm', 'topology_status', 'topology_valid', 'loop_count', 'outer_region_count',
            'material_region_count', 'void_loop_count', 'max_nesting_depth', 'net_area_mm2',
            'topology_signature', 'sampling', 'comparison_to_reference', 'extraction_method',
            'numeric_on_plane_epsilon', 'segment_count', 'traversed_segment_count',
            'all_segments_traversed_once', 'ignored_degenerate_segment_count')
    for i, sample in enumerate(evidence['samples']):
        entry = {key: sample[key] for key in keys if key in sample}
        entry['full_evidence_json_pointer'] = f'/samples/{i}'
        entry['loops'] = [{key: loop[key] for key in ('id', 'parent_id', 'depth', 'role', 'area_mm2')}
                          | {'point_count': len(loop['points_mm']),
                             'source_triangle_count': len(loop.get('source_triangle_ids', []))}
                          for loop in sample['loops']]
        summary['samples'].append(entry)
    save_json(out / 'sections.json', summary)
    return summary


def _open_points(ring):
    # Remove exact duplicate consecutive endpoints, not geometric simplification.
    points = np.asarray(ring.coords)[:-1]
    mask = np.r_[True, np.any(points[1:] != points[:-1], axis=1)]
    return points[mask].tolist()


def make_plate_plan(mesh, local, rotation, origin, stack, source_name, sha256,
                    instance=0, instance_count=1, tolerance=.05, provenance=None):
    """Propose a local, single-shell plan; unsupported topology fails closed."""
    tolerance = validate_tolerance(tolerance)
    provenance = {} if provenance is None else provenance
    source_blockers = source_build_blockers(provenance)
    if source_blockers:
        raise ValueError('Source provenance blocks reconstruction: ' + ', '.join(source_blockers))
    quality = mesh_quality(mesh)
    if quality['blocking_issues']:
        raise ValueError('Source mesh quality blocks reconstruction: ' + ', '.join(quality['blocking_issues']))
    if len(split_bodies(mesh)) != 1:
        raise ValueError('Disconnected shells are not certified independent solids; reconstruction refused')
    extents = local.extents
    if min(extents[:2]) <= 0 or extents[2] / min(extents[:2]) > .12:
        raise ValueError('Not a thin plate; refusing to flatten it')
    reference_index = stack.metadata['reference_sample_index']
    if reference_index is None or stack.metadata['failed_samples']:
        raise ValueError('Section evidence is incomplete/invalid; no reconstruction plan')
    reference = stack.samples[reference_index]
    if reference.geometry.geom_type != 'Polygon' or reference.geometry.is_empty:
        raise ValueError('Multiple material regions/nested islands cannot be expressed by this plate builder')
    if any(sample.metadata['max_nesting_depth'] > 1 for sample in stack.samples):
        raise ValueError('Nested material islands are unsupported; all loops retained, none dropped')
    if stack.metadata['topology_changed']:
        raise ValueError('Section topology changes with height (pockets/bridges/steps); full-thickness cuts refused')
    simplified = reference.geometry.simplify(tolerance, preserve_topology=True)
    if simplified.geom_type != 'Polygon' or not simplified.is_valid:
        raise ValueError('Profile simplification does not retain a valid single material region')
    cuts = []
    # Fit original boundaries, not already-simplified hole polygons.
    for i, ring in enumerate(reference.geometry.interiors, 1):
        circle = fit_circle(np.asarray(ring.coords)[:-1], min(tolerance, .025))
        if circle:
            cuts.append({'id': f'hole_{i:02d}', 'type': 'circle', **circle})
        else:
            polygon = Polygon(ring).simplify(tolerance, preserve_topology=True)
            cuts.append({'id': f'opening_{i:02d}', 'type': 'polygon', 'points_mm': _open_points(polygon.exterior)})
    warnings = [
        '仅单连通壳薄板截面方案，不是通用 CAD 特征树恢复。',
        '离散截面一致不证明连续等厚；未完成网格自交、双向三维表面偏差或装配精度验收。',
        '草图未完全约束；孔径、孔中心与厚度独立修改，不通过整体缩放移动孔位。',
        '输出采用新零件模板首个基准面的局部坐标；源模型旋转和平移仅留作追溯，不会自动应用。',
        'SolidWorks 2024 的真实运行与特征构建仍须在 Windows 验证。',
    ]
    if stack.metadata['profile_varies'] is True:
        warnings.append('多层截面形状变化：本计划只是参考层的等厚近似，倒角/孔径变化等不会完整还原。')
    if stack.metadata['profile_varies'] is None:
        warnings.append('截面差异界限跨越容差，均匀性未确定。')
    if stack.metadata['sampling_policy']['coverage_limited']:
        warnings.append('顶点高度事件超过抽样上限；未抽取的高度间隔已记录，不宣称全部细节已检查。')
    plan = {
        'schema_version': 2, 'strategy': 'review_required_plate_profile',
        'source_sha256': sha256, 'source_name': source_name,
        'instance': instance, 'instance_count': instance_count,
        'source_bounds_mm': mesh.bounds.tolist(), 'source_triangles': len(mesh.faces),
        'source_volume_mm3': quality['signed_volume_mm3'],
        'local_to_source_rotation': rotation.tolist(), 'source_origin_mm': origin.tolist(),
        'source_placement_applied': False, 'parameters': {'thickness_mm': float(extents[2])},
        'section_z_mm': reference.metadata['z_mm'], 'simplification_tolerance_mm': tolerance,
        'outer_points_mm': _open_points(simplified.exterior), 'cuts': cuts,
        'approved_for_build': False, 'requires_approximation_acceptance': True,
        'accepted_approximation': False, 'build_blockers': [], 'warnings': warnings,
        'cad_reconstructed': False, 'design_intent_recovered': False,
        'section_checks': [sample.metadata for sample in stack.samples],
        'section_evidence_summary': stack.metadata, 'source_provenance': provenance,
    }
    volume = plan_volume_mm3(plan)  # Validate containment/overlap BEFORE emitting a plan.
    plan['estimated_reconstruction_volume_mm3'] = volume
    plan['volume_difference_percent'] = float((volume / quality['signed_volume_mm3'] - 1) * 100)
    proposed = Polygon(plan['outer_points_mm'])
    for cut in cuts:
        shape = (Point(cut['center_mm']).buffer(cut['diameter_mm'] / 2, quad_segs=256)
                 if cut['type'] == 'circle' else Polygon(cut['points_mm']))
        proposed = proposed.difference(shape)
    lower, upper, error = _bounded_boundary_hausdorff(reference.geometry, proposed, tolerance)
    circle_error = max((cut['diameter_mm'] / 2 * (1 - np.cos(np.pi / 1024))
                        for cut in cuts if cut['type'] == 'circle'), default=0.)
    plan['profile_approximation_checks'] = {
        'symmetric_difference_area_mm2': float(reference.geometry.symmetric_difference(proposed).area),
        'boundary_hausdorff_lower_bound_mm': max(0., lower - float(circle_error)),
        'boundary_hausdorff_upper_bound_mm': upper + float(circle_error),
        'distance_sampling_error_bound_mm': error, 'circle_discretization_error_bound_mm': float(circle_error),
        'scope': '2D_reference_profile_only_not_3D_surface_or_design_intent',
    }
    if abs(plan['volume_difference_percent']) > 2:
        warnings.append(f'等厚方案体积差 {plan["volume_difference_percent"]:+.2f}%，超过 2% 提醒阈值；不等价于源几何。')
    return plan


def write_plan(plan, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'plan.json', plan)
    with (out / 'holes.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'x_mm', 'y_mm', 'diameter_mm', 'vertex_fit_error_mm', 'boundary_fit_error_mm'])
        for cut in plan['cuts']:
            if cut['type'] == 'circle':
                writer.writerow([cut['id'], *cut['center_mm'], cut['diameter_mm'],
                                 cut['radial_error_mm'], cut['max_boundary_radial_error_mm']])
    write_report(plan, out)


def analyze(source, out, tolerance=.05, instance=0):
    tolerance = validate_tolerance(tolerance)
    source = Path(source)
    # The hash, parsed geometry and delivered bytes must all describe ONE
    # immutable snapshot, even if the external source changes during analysis.
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary) / 'source.3mf'
        shutil.copyfile(source, snapshot)
        sha256 = source_hash(snapshot)
        document = load_document(snapshot)
        source_bytes = snapshot.read_bytes()
    if type(instance) is not int or not 0 <= instance < len(document.instances):
        raise ValueError(f'instance must be 0..{len(document.instances) - 1}')
    selected = document.instances[instance]
    if mesh_quality(selected.mesh)['blocking_issues']:
        raise ValueError('Source mesh needs explicit repair; use assistant.py for full evidence')
    rotation, origin, local = plate_frame(selected.mesh)
    stack = analyze_sections(local, tolerance)
    provenance = {**selected.metadata,
                  'document_coverage_complete': document.metadata['document_coverage_complete']}
    plan = make_plate_plan(selected.mesh, local, rotation, origin, stack, source.name,
                           sha256, instance, len(document.instances), tolerance, provenance)
    evidence = map_source_indices(
        section_stack_json(stack), np.arange(len(selected.mesh.faces)),
        np.asarray(local.metadata['source_vertex_ids']), copy_data=True)
    frame = {'origin_mm': origin.tolist(), 'local_to_source_rotation': rotation.tolist(),
             'coordinate_frame': 'source_mesh_mm', 'handedness': 'right'}
    evidence.update(source_triangle_id_scope='build_instance_faces',
                    source_vertex_id_scope='build_instance_vertices', section_frame=frame)
    plan.update(section_checks=evidence['samples'],
                section_evidence_summary={k: v for k, v in evidence.items() if k != 'samples'},
                section_source_triangle_id_scope='build_instance_faces',
                section_source_vertex_id_scope='build_instance_vertices', section_frame=frame,
                source_snapshot='source.3mf')
    out = prepare_output(out)
    (out / 'source.3mf').write_bytes(source_bytes)
    plan['source_provenance']['source_path'] = str((out / 'source.3mf').resolve())
    write_plan(plan, out)
    write_section_evidence(evidence, out)
    return plan


def profile_svg(geometry, title='参考层实际材料截面'):
    if geometry.is_empty:
        return '<svg xmlns="http://www.w3.org/2000/svg"><title>空截面</title></svg>'
    polygons = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
    def path(ring):
        return 'M ' + ' L '.join(f'{x:.8g},{-y:.8g}' for x, y in ring.coords) + ' Z'
    paths = ' '.join(path(ring) for polygon in polygons for ring in [polygon.exterior, *polygon.interiors])
    x0, y0, x1, y1 = geometry.bounds
    padding = max(x1 - x0, y1 - y0) * .03 + .1
    viewbox = f'{x0-padding} {-y1-padding} {x1-x0+padding*2} {y1-y0+padding*2}'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}">'
            f'<title>{html.escape(title)}</title><path d="{paths}" fill="#b8c9d6" '
            'stroke="#334155" stroke-width=".2" fill-rule="evenodd"/></svg>')


def write_report(plan, out):
    geometry = Polygon(plan['outer_points_mm'])
    rows = []
    for cut in plan['cuts']:
        if cut['type'] == 'circle':
            x, y = cut['center_mm']
            geometry = geometry.difference(Point(x, y).buffer(cut['diameter_mm'] / 2, quad_segs=256))
            rows.append(f'<tr><td>{html.escape(cut["id"])}</td><td>{x:.4f}</td><td>{y:.4f}</td>'
                        f'<td>{cut["diameter_mm"]:.4f}</td></tr>')
        else:
            geometry = geometry.difference(Polygon(cut['points_mm']))
    svg = profile_svg(geometry, '审核前的近似方案，不是源模型完整重建')
    (Path(out) / 'profile.svg').write_text(svg, encoding='utf-8')
    warnings = ''.join('<li>' + html.escape(w) + '</li>' for w in plan['warnings'])
    report = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>板件候选审核</title>
<style>body{{font:16px system-ui;margin:32px auto;max-width:1000px;padding:0 24px;color:#243244}}svg{{width:100%;max-height:650px;background:#f3f6f9}}td,th{{padding:8px;border-bottom:1px solid #ddd}}li{{margin:8px 0}}</style>
<h1>{html.escape(plan['source_name'])} · 板件近似候选</h1>
<p>未批准、未生成 CAD。厚度 {plan['parameters']['thickness_mm']:.6f} mm；体积差 {plan['volume_difference_percent']:+.3f}%。体积接近不能证明孔位和表面相同。</p>
{svg}<h2>独立孔位参数（局部 mm）</h2><table><tr><th>孔</th><th>X</th><th>Y</th><th>直径</th></tr>{''.join(rows)}</table>
<h2>准确性与坐标边界</h2><ul>{warnings}</ul>
<p>检查 <a href="plan.json">plan.json</a> 与 <a href="sections.json">截面摘要</a>、<a href="sections.full.json.gz">完整无损截面证据</a>。明确接受近似后设置 accepted_approximation 与 approved_for_build 为 true，再用 build_solidworks.py --validate-only 离线校验；校验通过也不表示真实 SolidWorks 已执行。</p></html>'''
    (Path(out) / 'report.html').write_text(report, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='3MF 单壳薄板候选：严格截面证据与审核计划')
    parser.add_argument('source')
    parser.add_argument('--out', default='reverse-output')
    parser.add_argument('--tolerance', type=float, default=.05)
    parser.add_argument('--instance', type=int, default=0)
    args = parser.parse_args()
    try:
        plan = analyze(args.source, args.out, args.tolerance, args.instance)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'分析失败：{exc}\n')
    print(json.dumps({'report': str(Path(args.out) / 'report.html'),
                      'circle_count': sum(c['type'] == 'circle' for c in plan['cuts']),
                      'approved_for_build': False, 'warnings': plan['warnings']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
