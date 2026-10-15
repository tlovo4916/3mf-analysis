"""Bounded 3MF evidence pipeline: input/geometry/topology/plan/CAD are separate."""
import argparse
import html
import json
from pathlib import Path
import shutil
import time

import numpy as np

from threemf_io import load_document
from geometry_analysis import inspect_mesh
from feature_relations import feature_relationships, section_directions
from mesh_evidence import (analysis_view, map_record_to_instance, map_source_indices,
                           mesh_quality, split_bodies)
from mesh_diagnostics import mesh_advisories, shell_relationships
from reverse_assistant import (local_mesh, make_plate_plan, prepare_output, profile_svg,
                               save_json, section_stack_json, source_hash, validate_tolerance,
                               write_plan, write_section_evidence)
from section_analysis import analyze_sections

VERSION = '0.5.0'
DEFAULT_MAX_ANALYZED_FACES = 250_000
DEFAULT_MAX_BODIES = 128


def _reference_geometry(stack):
    index = stack.metadata['reference_sample_index']
    return stack.samples[index].geometry if index is not None else None


def _stack_with_source_ids(stack, triangle_map, vertex_map=None):
    """Pure export; local node/segment IDs are NOT source face/vertex IDs."""
    if vertex_map is None:
        raise ValueError('Both triangle and vertex source mappings are required')
    evidence = map_source_indices(section_stack_json(stack), triangle_map, vertex_map, copy_data=True)
    evidence['source_triangle_id_scope'] = 'build_instance_faces'
    evidence['source_vertex_id_scope'] = 'build_instance_vertices'
    return evidence


def _summary(stack):
    summary = {key: stack.metadata[key] for key in (
        'sample_count', 'profile_varies', 'profile_status', 'topology_changed',
        'max_boundary_hausdorff_mm', 'max_boundary_hausdorff_upper_bound_mm',
        'max_symmetric_difference_area_mm2', 'sampling_complete', 'boundary_comparison_inconclusive')}
    summary['failed_sample_count'] = len(stack.metadata['failed_samples'])
    summary['coverage_limited'] = stack.metadata['sampling_policy']['coverage_limited']
    policy = stack.metadata.get('event_probe_policy', {})
    summary['event_probe_summary'] = {key: policy.get(key, 0) for key in (
        'event_failure_count', 'attempted_section_count', 'successful_side_count',
        'failed_side_count', 'unavailable_side_count')}
    summary['event_probe_summary']['used_to_replace_exact_failures'] = False
    return summary


def _base_record(partition, i, j, quality, role):
    return {
        'instance_index': i, 'body_index': j, 'triangles': len(partition.mesh.faces),
        'vertices': len(partition.mesh.vertices), 'source_triangle_count': len(partition.mesh.faces),
        'instance_triangle_ids': partition.instance_triangle_ids.tolist(),
        'instance_vertex_ids': partition.instance_vertex_ids.tolist(),
        'mesh_quality': map_source_indices(quality, partition.instance_triangle_ids,
                                           partition.instance_vertex_ids, copy_data=True),
        'shell_role_candidate': role, 'watertight': quality['watertight'],
        'winding_consistent': quality['winding_consistent'],
        'signed_volume_mm3': quality['signed_volume_mm3'], 'volume_mm3': quality['signed_volume_mm3'],
        'blocking_issues': list(quality['blocking_issues']),
        'planes': [], 'cylinders': [], 'cones': [], 'spheres': [], 'section_analyses': [],
        'plate_plan_status': 'strategy_not_supported', 'notes': [],
        'strategy': 'freeform_or_unresolved', 'candidate_geometry_only': True,
        'requires_review': True, 'mesh_repair_performed': False,
        'cad_reconstructed': False, 'brep_reconstructed': False, 'design_intent_recovered': False,
    }


def _analyze_body(partition, instance, package, i, j, role, out, tolerance,
                  max_directions, budget_available):
    quality = mesh_quality(partition.mesh)
    base = _base_record(partition, i, j, quality, role)
    base.update(source_reconstruction_blockers=instance.metadata['reconstruction_blockers'],
                source_component_roles=[{
                    'component_index': c['component_index'], 'slicer_role': c.get('slicer_role'),
                    'mapping_status': c.get('slicer_role_mapping_status', 'not_applicable'),
                    'source_config_locator': c.get('slicer_config_locator'),
                    'config_name': c.get('slicer_config_name'),
                } for c in instance.metadata['components'] if np.any(
                    (partition.instance_triangle_ids >= c['face_start']) &
                    (partition.instance_triangle_ids < c['face_end']))],
                source_geometry_semantics='retained_core_leaf_geometry_not_evaluated_printed_material')
    if not budget_available:
        base.update(strategy='analysis_budget_exceeded', analysis_status='not_analyzed_budget',
                    plate_plan_status='blocked', plate_plan_blockers=['Analysis budget exceeded'])
        base['notes'].append('此连通壳超出候选分析预算；完整源网格与面索引保留，未默默忽略。')
        return base
    folder = out / f'instance-{i:03d}-body-{j:03d}'
    folder.mkdir()
    try:
        view, view_metadata = analysis_view(partition)
        record = inspect_mesh(view.mesh, i, j, tolerance)
        record = map_record_to_instance(record, view, instance.metadata, i)
    except (ValueError, ArithmeticError) as exc:
        base.update(strategy='repair_required', analysis_status='candidate_fit_failed',
                    analysis_error=str(exc), plate_plan_status='blocked', plate_plan_blockers=[str(exc)])
        base['notes'].append('无法拟合此连通壳；源面保留，未修复或删除：' + str(exc))
        return base
    candidate_quality = record.pop('mesh_quality')
    notes = record['notes']
    for key, value in base.items():
        if key not in {'strategy', 'notes', 'planes', 'cylinders', 'cones', 'spheres'}:
            record[key] = value
    record['candidate_view_quality'] = candidate_quality
    record['candidate_analysis_view'] = view_metadata
    record['analyzed_triangle_count'] = len(view.mesh.faces)
    record['analysis_status'] = 'candidate_analysis_complete'
    record['geometry_candidate_strategy'] = record['strategy']
    record['source_reconstruction_blockers'] = instance.metadata['reconstruction_blockers']
    if quality['blocking_issues']:
        record['strategy'] = 'repair_required'
        notes.append('源网格重建阻塞：' + ', '.join(quality['blocking_issues']))
    if role == 'possible_cavity_or_inverted_nested_shell':
        record['strategy'] = 'shell_relationship_review'
        notes.append('负体积嵌套壳可能为内腔边界或倒置独立壳；包围盒包含不是包含认证，不建议自动翻面。')
    if instance.metadata['reconstruction_blockers']:
        record['strategy'] = 'source_purpose_review'
        notes.append('源对象用途/可打印性仅允许分析，不允许重建：' + ', '.join(instance.metadata['reconstruction_blockers']))
    if view_metadata['skipped_degenerate_instance_triangle_ids']:
        notes.append('仅拟合视图跳过零面积面；源面和索引全部保留，未修复、未解除阻塞。')
        record['candidate_coverage']['excluded_degenerate_instance_triangle_ids'] = view_metadata['skipped_degenerate_instance_triangle_ids']
        record['candidate_coverage']['source_triangle_count_including_excluded'] = len(partition.mesh.faces)
    advisories = mesh_advisories(partition.mesh, tolerance)
    record['mesh_advisories'] = map_source_indices(advisories, partition.instance_triangle_ids,
                                                  partition.instance_vertex_ids, copy_data=True)
    # Relationship hints and section evidence are independent capabilities.
    # A failed advisory must not prevent otherwise well-defined frame slices.
    try:
        record['feature_relationships'] = feature_relationships(partition.mesh, record, tolerance)
    except (ValueError, ArithmeticError) as exc:
        record['relationship_analysis_error'] = str(exc)
        notes.append('曲面关系分析失败，仍尝试独立截面证据：' + str(exc))
    try:
        directions = section_directions(record, max_directions=max_directions)
    except (ValueError, ArithmeticError) as exc:
        record['section_direction_error'] = str(exc)
        notes.append('曲面引导切片失败，改用已验证局部坐标系：' + str(exc))
        fallback = {key: record[key] for key in (
            'thinness_ratio', 'oriented_extents_mm', 'local_to_source_rotation', 'source_origin_mm')}
        try:
            directions = section_directions(fallback, max_directions=max_directions)
        except (ValueError, ArithmeticError) as fallback_error:
            directions = []
            record['section_direction_error'] += '; fallback: ' + str(fallback_error)
    provenance = {**instance.metadata, 'document_coverage_complete': package['document_coverage_complete']}
    for direction_index, direction in enumerate(directions):
        section_folder = folder if direction_index == 0 else folder / direction['id']
        frame = direction['frame']
        rotation = np.asarray(frame['local_to_source_rotation'])
        origin = np.asarray(frame['origin_mm'])
        local = local_mesh(partition.mesh, rotation, origin)
        entry = {**direction, 'sampling_max_layers': 40 if direction_index == 0 else 7}
        try:
            stack = analyze_sections(local, tolerance, max_samples=entry['sampling_max_layers'])
            evidence = _stack_with_source_ids(stack, partition.instance_triangle_ids, partition.instance_vertex_ids)
            evidence['section_frame'] = frame
            evidence['direction'] = direction
            write_section_evidence(evidence, section_folder)
            entry.update(summary=_summary(stack), status='sampled_with_failures' if stack.metadata['failed_samples'] else 'sampled',
                         section_evidence=str((section_folder / 'sections.json').relative_to(out)),
                         full_section_evidence=str((section_folder / 'sections.full.json.gz').relative_to(out)))
            reference = _reference_geometry(stack)
            if reference is not None:
                (section_folder / 'reference-section.svg').write_text(profile_svg(reference), encoding='utf-8')
                entry['section_preview'] = str((section_folder / 'reference-section.svg').relative_to(out))
            if direction_index == 0:
                for key in ('section_evidence', 'full_section_evidence', 'section_preview'):
                    if key in entry:
                        record[key] = entry[key]
                record['section_summary'] = entry['summary']
            if direction['is_plate_direction']:
                try:
                    if len(instance.metadata['components']) != 1 or role not in {'separate_shell', 'outer_shell_candidate'}:
                        raise ValueError('Multiple components/nested or ambiguous shells: no union or cavity grouping inferred')
                    if instance.metadata.get('connected_shell_count_for_analysis', 1) != 1:
                        raise ValueError('Multiple shells: full solid grouping remains unverified')
                    plan = make_plate_plan(partition.mesh, local, rotation, origin, stack,
                                           out.name, source_hash(out / 'source.3mf'), i,
                                           package['instance_count'], tolerance, provenance)
                except ValueError as exc:
                    record['plate_plan_status'] = 'blocked'
                    record['plate_plan_blockers'] = [str(exc)]
                    notes.append('不生成板件计划：' + str(exc))
                else:
                    # Plan evidence must share the SAME declared instance ID
                    # scope as sections.full.json.gz, without mutating the stack.
                    plan['section_checks'] = evidence['samples']
                    plan['section_evidence_summary'] = {
                        key: value for key, value in evidence.items() if key != 'samples'}
                    plan['section_source_triangle_id_scope'] = 'build_instance_faces'
                    plan['section_source_vertex_id_scope'] = 'build_instance_vertices'
                    plan['section_frame'] = frame
                    write_plan(plan, section_folder)
                    record.update(plate_plan_status='requires_human_review',
                                  plate_plan=str((section_folder / 'plan.json').relative_to(out)),
                                  plate_report=str((section_folder / 'report.html').relative_to(out)),
                                  approximation_volume_difference_percent=plan['volume_difference_percent'])
        except (ValueError, ArithmeticError) as exc:
            entry.update(status='failed', error=str(exc))
            notes.append('此切片方向失败，不影响其他候选：' + str(exc))
            if direction['is_plate_direction']:
                record.update(plate_plan_status='blocked', plate_plan_blockers=[str(exc)])
        record['section_analyses'].append(entry)
    if not directions:
        record['analysis_status'] = 'geometry_complete_sections_unavailable'
    if instance.metadata.get('connected_shell_count_for_analysis', 1) > 1:
        notes.append('实例含多连通壳；壳间关系仅包围盒候选，不认证独立实体、不合并重建。')
    return record


def process(source, out, tolerance=.05, *, strict=False, include_nonprintable=False,
            max_section_directions=3, max_analyzed_faces=DEFAULT_MAX_ANALYZED_FACES,
            max_bodies=DEFAULT_MAX_BODIES):
    """Analyze geometry-only by default; strict mode restores all-or-error input.

    Candidate/slice work is capped per document. Every unexamined shell and any
    failed build item remain explicitly listed, while source arrays are retained.
    """
    tolerance = validate_tolerance(tolerance)
    for label, value, lower, upper in (
        ('max_section_directions', max_section_directions, 1, 3),
        ('max_analyzed_faces', max_analyzed_faces, 1, 4_000_000),
        ('max_bodies', max_bodies, 1, 10_000),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f'{label} must be an integer in [{lower}, {upper}]')
    if type(strict) is not bool or type(include_nonprintable) is not bool:
        raise ValueError('Reader mode options must be booleans')
    source = Path(source)
    out = prepare_output(out)
    snapshot = out / 'source.3mf'
    shutil.copyfile(source, snapshot)
    sha256 = source_hash(snapshot)
    start = time.perf_counter()
    document = load_document(snapshot, geometry_only=not strict, isolate_build_errors=not strict,
                             include_nonprintable=include_nonprintable)
    records, instance_records = [], []
    analyzed_faces = analyzed_bodies = 0
    for i, instance in enumerate(document.instances):
        mesh = instance.mesh
        mesh_path = out / f'instance-{i:03d}-source-mesh.npz'
        np.savez_compressed(mesh_path, vertices_mm=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces))
        partitions = split_bodies(mesh)
        relationships = shell_relationships(partitions)
        instance.metadata['connected_shell_count_for_analysis'] = len(partitions)
        instance_records.append({
            'instance_index': i, **instance.metadata, 'mesh_arrays': mesh_path.name,
            'mesh_arrays_semantics': 'world_mm_exact_weld_within_leaf_occurrence_no_face_removal_no_normal_repair',
            'connected_shell_count': len(partitions),
            'connected_shells_are_certified_independent_solids': False,
            'boolean_union_of_components_performed': False, 'shell_relationships': relationships,
            'mesh_quality': mesh_quality(mesh),
        })
        for j, partition in enumerate(partitions):
            available = analyzed_bodies < max_bodies and analyzed_faces + len(partition.mesh.faces) <= max_analyzed_faces
            if available:
                analyzed_bodies += 1
                analyzed_faces += len(partition.mesh.faces)
            record = _analyze_body(partition, instance, document.metadata, i, j,
                                   relationships['shells'][j]['role_candidate'], out,
                                   tolerance, max_section_directions, available)
            record['source_name'] = source.name
            if 'plate_plan' in record:
                plan_path = out / record['plate_plan']
                plan = json.loads(plan_path.read_text(encoding='utf-8'))
                plan['source_name'] = source.name
                write_plan(plan, plan_path.parent)
            records.append(record)
    budget_skips = sum(record.get('analysis_status') == 'not_analyzed_budget' for record in records)
    geometry_errors = any(any(key in record for key in (
        'analysis_error', 'relationship_analysis_error', 'section_direction_error')) for record in records)
    section_errors = any(any(
        entry['status'] != 'sampled' or entry['summary']['failed_sample_count'] > 0
        for entry in record['section_analyses']) for record in records)
    candidate_complete = not budget_skips and not geometry_errors and document.metadata['document_coverage_complete']
    section_attempts_complete = candidate_complete and all(
        record['section_analyses'] and all(
            entry['status'] == 'sampled' and entry['summary']['failed_sample_count'] == 0
            for entry in record['section_analyses'])
        for record in records)
    coverage_complete = candidate_complete and section_attempts_complete
    candidate_search_limits = any(
        any(record.get('cylinder_segmentation_diagnostics', {}).get(key, 0) for key in (
            'axis_hypotheses_omitted', 'axis_representatives_omitted', 'region_fits_omitted',
            'body_axis_budget_reached')) for record in records)
    event_probe_totals = {key: sum(
        entry.get('summary', {}).get('event_probe_summary', {}).get(key, 0)
        for record in records for entry in record['section_analyses']) for key in (
            'event_failure_count', 'attempted_section_count', 'successful_side_count',
            'failed_side_count', 'unavailable_side_count')}
    # A possible cavity is an explanation, NOT containment/orientation proof.
    # Preserve every source veto even when it may ultimately be a valid cavity.
    mesh_blockers = any(record['blocking_issues'] for record in records)
    shell_review = any(record['shell_role_candidate'] in {
        'possible_cavity_or_inverted_nested_shell', 'nested_shell_review',
        'outer_shell_candidate', 'bounds_overlap_review', 'relationship_review'} for record in records)
    report = {
        'schema_version': 5, 'analyzer_version': VERSION, 'source_name': source.name,
        'source_sha256': sha256, 'source_snapshot': snapshot.name,
        'units': 'millimeter', 'coordinate_convention': 'column_vector_source_world_mm',
        'tolerance_mm': tolerance, 'package': document.metadata,
        'instance_count': len(document.instances), 'body_count': len(records),
        'body_count_semantics': 'shared_edge_connected_shell_count_not_certified_solid_count',
        'instances': instance_records, 'bodies': records,
        'status': 'analysis_complete_not_cad_complete' if coverage_complete else 'analysis_partial_not_cad_complete',
        'geometry_analysis_coverage_complete': coverage_complete,
        'candidate_analysis_coverage_complete': bool(candidate_complete),
        'candidate_analysis_coverage_scope': 'scheduled_source_faces_and_shells_not_candidate_recall',
        'candidate_search_exhaustive': False,
        'candidate_recall_certified': False,
        'has_candidate_search_limits': bool(candidate_search_limits),
        'event_probe_summary': {**event_probe_totals, 'exact_failures_replaced': False,
                                'finite_offset_only_not_limit_certification': True},
        'section_analysis_attempts_complete': bool(section_attempts_complete),
        'coverage_scope': 'included_source_geometry_and_attempted_samples_not_continuous_or_BRep_certification',
        'requires_shell_relationship_review': bool(shell_review),
        'has_mesh_blockers': bool(mesh_blockers),
        'has_reconstruction_blockers': (not document.metadata['document_coverage_complete'] or
                                       document.metadata.get('reconstruction_veto', False) or
                                       not section_attempts_complete or shell_review or
                                       any(record['blocking_issues'] or record.get('source_reconstruction_blockers')
                                           or record['plate_plan_status'] == 'blocked' for record in records)),
        'has_candidate_analysis_errors': geometry_errors,
        'has_section_analysis_errors': section_errors,
        'has_body_analysis_errors': geometry_errors or section_errors,
        'analysis_budget': {'max_analyzed_faces': max_analyzed_faces, 'max_bodies': max_bodies,
                            'analyzed_faces': analyzed_faces, 'analyzed_bodies': analyzed_bodies,
                            'unexamined_shell_count': budget_skips, 'max_section_directions': max_section_directions,
                            'primary_max_layers': 40, 'secondary_max_layers': 7},
        'source_geometry_equivalence_verified': False,
        'solidworks_runtime_verified': False, 'brep_reconstructed': False, 'design_intent_recovered': False,
        'elapsed_seconds': round(time.perf_counter() - start, 3),
        'limitations': [
            'No automatic mesh repair, Boolean union, shell containment or self-intersection certification.',
            'Plane/cylinder/cone/sphere evidence is not a trimmed analytic BRep or feature history.',
            'Section comparison is 2D and bounded/discrete, not a bidirectional 3D surface certificate.',
            'Optional/appearance extensions are not recovered; unsupported required geometry extensions fail closed.',
            'No USD-BRep runtime adapter is installed or required by the core analyzer.',
            'Local cylinder search is bounded; short/noisy runs can remain unresolved. Candidate recall is not certified.',
            'Slicer roles are preserved, but Boolean material and assembly pose are NOT recovered.',
            'Finite one-sided event probes supplement, never replace, exact failed sections or CAD vetoes.',
        ],
    }
    save_json(out / 'analysis.json', report, compact=True)
    write_report(report, out)
    return report


def write_report(report, out):
    cards = []
    for body in report['bodies']:
        notes = ''.join('<li>' + html.escape(note) + '</li>' for note in body['notes'])
        links = []
        for entry in body.get('section_analyses', []):
            if 'section_evidence' in entry:
                links.append(f'<a href="{html.escape(entry["section_evidence"], quote=True)}">{entry["id"]} 截面摘要</a>')
                links.append(f'<a href="{html.escape(entry["full_section_evidence"], quote=True)}">{entry["id"]} 无损完整证据</a>')
        if 'plate_report' in body:
            links.append(f'<a href="{html.escape(body["plate_report"], quote=True)}">板件近似方案审核</a>')
        size = ' × '.join(f'{x:.4f}' for x in body.get('oriented_extents_mm', [])) or '未确定'
        counts = '，'.join(f'{label} {len(body[key])}' for key, label in
                           [('planes', '平面'), ('cylinders', '圆柱'), ('cones', '圆锥'), ('spheres', '球面')])
        surface_rows = ''.join(
            f'<tr><td>{html.escape(c["type"])}</td><td>{c["max_vertex_residual_mm"]:.6g}</td>'
            f'<td>{c["max_surface_residual_mm"]:.6g}</td><td>{c["surface_approximation_within_tolerance"]}</td></tr>'
            for key in ('cylinders', 'cones', 'spheres') for c in body[key])
        preview = (f'<img src="{html.escape(body["section_preview"], quote=True)}" '
                   'alt="参考层实际材料截面，保留嵌岛" style="width:100%;max-height:450px">'
                   if 'section_preview' in body else '')
        summaries = ''.join(
            f'<li>{entry["id"]}：{html.escape(entry["status"])}；' +
            (f'{entry["summary"]["sample_count"]} 层，{html.escape(entry["summary"]["profile_status"])}，'
             f'失败 {entry["summary"]["failed_sample_count"]} 层，抽样受限 {entry["summary"]["coverage_limited"]}。'
             if 'summary' in entry else html.escape(entry.get('error', ''))) + '</li>'
            for entry in body.get('section_analyses', []))
        quality = body['mesh_quality']
        blockers = ', '.join(body.get('plate_plan_blockers', []))
        roles = '；'.join(
            f"组件 {c['component_index']}：{c.get('slicer_role') or '未声明'} / {c['mapping_status']}"
            for c in body.get('source_component_roles', []))
        search = body.get('cylinder_segmentation_diagnostics', {})
        local_count = search.get('accepted_local_candidate_count', 0)
        probe_count = sum(e.get('summary', {}).get('event_probe_summary', {}).get('successful_side_count', 0)
                          for e in body.get('section_analyses', []))
        cards.append(f'''<section><h2>实例 {body['instance_index']} / 连通壳 {body['body_index']}</h2>
<p>策略：{html.escape(body['strategy'])}；壳角色候选：{html.escape(body['shell_role_candidate'])}；尺寸 {size} mm。</p>
<p>源三角面 {body['triangles']}；拓扑封闭 {quality['watertight']}；退化面 {len(quality['degenerate_triangle_ids'])}；重复面 {quality['duplicate_triangle_count']}；非流形边 {quality['nonmanifold_edge_count']}。自交未验收。</p>
<p>候选：{counts}，其中局部分割补充圆柱 {local_count}。候选召回未认证；曲面、边界和关系分别留存，不是已恢复 BRep。</p>
<p>来源角色：{html.escape(roles or '无切片器角色证据')}。这里分析保留网格，未将负体积求差为最终材料。事件两侧有效有限偏移截面 {probe_count}；不替换原失败、不授权重建。</p>{preview}
<ul>{summaries}</ul><p>板件计划：{html.escape(body['plate_plan_status'])} {html.escape(blockers)}</p>
<ul>{notes}</ul><p>{' · '.join(links)}</p><details><summary>曲面误差：顶点拟合不等于整片精度</summary>
<table><tr><th>类型</th><th>顶点误差 mm</th><th>整面误差/保守上界 mm</th><th>整面界限达标</th></tr>{surface_rows}</table></details></section>''')
    warnings = ''.join('<li>' + html.escape(warning) + '</li>' for warning in report['package']['warnings'])
    slicer = report['package'].get('slicer_metadata', {})
    roles_summary = html.escape(str(slicer.get('resolved_role_occurrence_counts', {})))
    veto = report['has_reconstruction_blockers']
    material_status = report['package'].get('material_semantics_complete', False)
    content = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>通用 3MF 分析</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 24px;color:#243244;overflow-wrap:anywhere}}section{{border:1px solid #ddd;padding:20px;margin:20px 0}}td,th{{padding:6px;text-align:left}}a{{color:#1765aa}}details{{overflow:auto}}li{{margin:6px 0}}</style>
<h1>{html.escape(report['source_name'])} · 通用几何分析 v{VERSION}</h1>
<p><strong>分析完成 ≠ 参数化 CAD 重建完成。</strong>所有候选需要审核；没有生成或验收 SLDPRT。重建阻塞：{veto}。</p>
<p>本轮局部搜索触发预算截断：{report['has_candidate_search_limits']}；流程跑完不代表候选找全。切片器角色映射：{html.escape(report['package'].get('role_mapping_status', 'not_applicable'))}；角色级材料语义完整：{material_status}（不等于打印实体已恢复）；用途计数 {roles_summary}。</p>
<p>{report['instance_count']} 个有效 build 实例，{report['body_count']} 个连通壳；连通壳不等于独立 CAD 实体。
失败 build 项 {len(report['package']['failed_build_items'])}；未分析壳 {report['analysis_budget']['unexamined_shell_count']}；状态 {report['status']}。</p>
<p><a href="analysis.json">完整参数、质量与来源追踪</a> · <a href="source.3mf">原始 3MF 快照</a></p>
<details><summary>包内扩展、失败项与解析提示</summary><ul>{warnings}</ul></details>{''.join(cards)}
<p>默认仅分析几何，不恢复颜色/纹理；只在单个组件内精确同坐标合并顶点；不补洞、不翻转源法向、不删除源面。USD-BRep 仅用于表示分层参考，未集成内核。</p></html>'''
    (Path(out) / 'index.html').write_text(content, encoding='utf-8')


def _next_folder(base):
    if not base.exists():
        return base
    for counter in range(1, 100_000):
        candidate = base.with_name(base.name + f'-run-{counter:03d}')
        if not candidate.exists():
            return candidate
    raise FileExistsError('Too many prior runs with the same output name')


def main():
    parser = argparse.ArgumentParser(description='通用 3MF：解析兼容性、质量闸、解析曲面与多方向截面证据')
    parser.add_argument('sources', nargs='+')
    parser.add_argument('--out', default='results')
    parser.add_argument('--tolerance', type=float, default=.05)
    parser.add_argument('--strict', action='store_true', help='严格解析；不放行外观扩展，不隔离单项错误')
    parser.add_argument('--include-nonprintable', action='store_true', help='额外分析不可打印项，但不允许重建')
    parser.add_argument('--section-directions', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--max-faces', type=int, default=DEFAULT_MAX_ANALYZED_FACES)
    parser.add_argument('--max-bodies', type=int, default=DEFAULT_MAX_BODIES)
    args = parser.parse_args()
    try:
        validate_tolerance(args.tolerance)
    except ValueError as exc:
        parser.error(str(exc))
    results = []
    for i, source in enumerate(args.sources):
        folder = _next_folder(Path(args.out) / f'{i:03d}-{Path(source).stem}')
        try:
            report = process(source, folder, args.tolerance, strict=args.strict,
                             include_nonprintable=args.include_nonprintable,
                             max_section_directions=args.section_directions,
                             max_analyzed_faces=args.max_faces, max_bodies=args.max_bodies)
            results.append({'source': source, 'report': str(folder / 'index.html'),
                            'bodies': report['body_count'], 'status': report['status'],
                            'has_mesh_blockers': report['has_mesh_blockers'],
                            'body_analysis_errors': report['has_body_analysis_errors'], 'ok': True,
                            'cad_reconstructed': False})
        except Exception as exc:
            folder.mkdir(parents=True, exist_ok=True)
            error = {'source': source, 'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)}
            save_json(folder / 'error.json', error)
            results.append(error)
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
    if any(not result['ok'] for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
