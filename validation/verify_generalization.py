"""Recheck delivered evidence independently of the corpus runner; no new CAD claims.

Baseline paths can be overridden. Defaults support the extracted delivery and
this isolated development tree. Writes verification.json only, not original runs.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from threemf_io import load_document


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    packed=ROOT/'baseline/v041'
    parser.add_argument('--baseline-src',type=Path,default=packed/'src' if packed.exists() else ROOT.parent/'deliverables/package/src')
    parser.add_argument('--baseline-real-root',type=Path,default=packed/'makerworld' if packed.exists() else ROOT.parent/'makerworld_validation')
    args=parser.parse_args()
    baseline_src=args.baseline_src;baseline_real=args.baseline_real_root
    base=read(ROOT/'validation/v041-base-hashes.json')
    assert all(sha(baseline_src/name)==digest for name,digest in base.items()),'v041 baseline changed'
    production={p.name:sha(p) for p in (ROOT/'src').glob('*.py') if not p.name.startswith('test_')}
    results=[];gzip_count=0;arrays=0;primary_identical=0;primary_changed=[];exact_failures_equal=0;probe_sides=0
    restored=read(baseline_real/'diagnosis/confirmed_cylinder_supports.json')['cases']
    support_checks=[];legacy_jobs=[];all_local=[]
    for folder in sorted((ROOT/'validation/runs').iterdir()):
        if not folder.is_dir():continue
        state=read(folder/'result.json');case_id=state['case_id']
        assert state['analyzer_version']=='0.5.0'
        assert all(state['source_hashes'].get(name)==digest for name,digest in production.items()),case_id
        snapshot=folder/'analysis/source.3mf'
        assert sha(snapshot)==state['source_sha256']
        if case_id=='official-pyramid':
            assert not state['pipeline']['produced_report'] and 'beamlattice' in state['pipeline']['error']
            results.append({'id':case_id,'expected_unsupported_geometry_rejected':True});continue
        report=read(folder/'analysis/analysis.json')
        assert report['source_sha256']==state['source_sha256']
        assert not report['candidate_search_exhaustive'] and not report['candidate_recall_certified']
        assert not report['solidworks_runtime_verified'] and not report['brep_reconstructed']
        assert not list((folder/'analysis').rglob('*.sldprt'))
        doc=load_document(snapshot,geometry_only=True,isolate_build_errors=True)
        for index,instance in enumerate(doc.instances):
            saved=np.load(folder/'analysis'/report['instances'][index]['mesh_arrays'],allow_pickle=False)
            assert np.array_equal(saved['vertices_mm'],instance.mesh.vertices)
            assert np.array_equal(saved['faces'],instance.mesh.faces);arrays+=1
            ids=[i for b in report['bodies'] if b['instance_index']==index for i in b['instance_triangle_ids']]
            assert sorted(ids)==list(range(len(instance.mesh.faces)))
        old_path=baseline_real/'runs'/case_id/'analysis'
        if case_id in ('01-air-sensor','02-oled-small','03-oled-retro'):
            assert old_path.exists() and report['status']=='analysis_partial_not_cad_complete'
            assert all('plate_plan' not in b for b in report['bodies'])
        for body in report['bodies']:
            shell_ids=set(body['instance_triangle_ids'])
            for collection in ('planes','cylinders','cones','spheres'):
                for c in body[collection]:
                    assert c['source_triangle_id_scope']=='build_instance_faces'
                    assert set(c['source_triangle_ids']).issubset(shell_ids)
                    if c.get('segmentation_method')=='local_normal_axis_consensus':
                        evidence=c['local_segmentation_evidence']
                        assert set(c['source_triangle_ids']).issubset(evidence['parent_normal_band']['source_triangle_ids'])
                        assert set(evidence['parent_normal_band']['source_triangle_ids']).issubset(shell_ids)
                        assert evidence['strict_fitter_unchanged'] and not evidence['geometry_or_normal_repair_performed']
                        assert not c['hole']['confirmed']
                        all_local.append({'case_id':case_id,'instance_index':body['instance_index'],'body_index':body['body_index'],
                                          'candidate_id':c['candidate_id'],'face_count':c['triangle_count'],'radius_mm':c['radius_mm'],
                                          'axial_length_mm':c['axial_length_mm'],'angular_coverage_degrees':c['angular_coverage_degrees'],
                                          'max_surface_residual_mm':c['max_surface_residual_mm'],'surface_approximation_within_tolerance':c['surface_approximation_within_tolerance'],
                                          'independent_feature_ground_truth_available':False})
                        if case_id=='03-oled-retro':
                            legacy_jobs.append({'instance':body['instance_index'],'faces':c['source_triangle_ids'],'radius':c['radius_mm'],'height':c['axial_length_mm']})
            for e in body['section_analyses']:
                if 'full_section_evidence' not in e:continue
                path=folder/'analysis'/e['full_section_evidence'];summary=read(folder/'analysis'/e['section_evidence'])
                assert sha(path)==summary['full_evidence']['sha256']
                with gzip.open(path,'rt',encoding='utf-8') as stream:full=json.load(stream)
                assert full['source_triangle_id_scope']=='build_instance_faces'
                gzip_count+=1
                for event in full.get('event_probes',[]):
                    parent=full['failed_samples'][event['parent_failed_sample_index']]
                    assert parent['error_code']=='ambiguous_section_event'
                    assert event['original_z_mm']==parent['z_mm']
                    for side in event['sides']:
                        probe_sides+=1
                        assert side['status'] in ('success','failed','unavailable') and side['stop_reason']
                        for attempt in side['attempts']:
                            assert not attempt['included_in_primary_sampling']
                            assert attempt['delta_mm']==attempt['actual_z_mm']-attempt['original_z_mm']
                            assert abs(attempt['delta_mm'])<=side['max_abs_delta_mm']
                            lo,hi=side['open_interval_z_mm'];assert lo<attempt['actual_z_mm']<hi
                for event in summary.get('event_probes',[]):
                    for side in event['sides']:
                        for attempt in side['attempts']:
                            assert 'evidence' not in attempt
                            value=full
                            for token in attempt['full_evidence_json_pointer'].split('/')[1:]:
                                value=value[int(token)] if isinstance(value,list) else value[token]
                            assert isinstance(value,dict) or value is None
                if old_path.exists() and (old_path/e['full_section_evidence']).exists():
                    with gzip.open(old_path/e['full_section_evidence'],'rt',encoding='utf-8') as stream:old=json.load(stream)
                    assert full['failed_samples']==old['failed_samples'],'Exact failure evidence changed'
                    exact_failures_equal+=1
                    keys=['samples','sampling_schedule','section_frame','direction','profile_varies','profile_status']
                    changed=[key for key in keys if full.get(key)!=old.get(key)]
                    if changed:primary_changed.append({'case_id':case_id,'path':e['full_section_evidence'],'changed_keys':changed})
                    else:primary_identical+=1
        if case_id=='03-oled-retro':
            old=read(old_path/'analysis.json')
            for known in restored:
                i,j=known['instance_index'],known['body_index'];expected=set(known['candidate']['source_triangle_ids'])
                b=next(b for b in report['bodies'] if b['instance_index']==i and b['body_index']==j)
                ob=next(b for b in old['bodies'] if b['instance_index']==i and b['body_index']==j)
                assert expected.issubset(ob['candidate_coverage']['unresolved_source_triangle_ids'])
                assert not any(expected&set(p['source_triangle_ids']) for p in ob['planes']) and not ob['cylinders']
                matched=[c for c in b['cylinders'] if set(c['source_triangle_ids'])==expected];assert len(matched)==1
                c=matched[0]
                assert c['triangle_count']==369 and c['wraps_full_circumference'] and c['max_surface_residual_mm']<.001
                support_checks.append({'instance':i,'body':j,'source_face_range_inclusive':[min(expected),max(expected)],
                                       'candidate_id':c['candidate_id'],'radius_mm':c['radius_mm'],
                                       'axial_length_mm':c['axial_length_mm'],'max_surface_residual_mm':c['max_surface_residual_mm'],
                                       'exact_original_source_patch_recovered':True})
        results.append({'id':case_id,'source_hash_and_full_arrays_verified':True,'status':report['status'],
                        'local_candidate_count':sum(c['case_id']==case_id for c in all_local)})
    # Isolated-original-patch refits with unchanged v041 code are a second
    # equation/threshold check, NOT an independent design-intent oracle.
    with tempfile.TemporaryDirectory() as temp:
        job=Path(temp)/'jobs.json';out=Path(temp)/'fits.json';write(job,legacy_jobs)
        code='''import json,sys,numpy as np,trimesh
from geometry_analysis import cylinder_candidates
jobs=json.load(open(sys.argv[1]));root=sys.argv[3];fits=[]
for job in jobs:
 a=np.load(root+f"/instance-{job['instance']:03d}-source-mesh.npz",allow_pickle=False)
 m=trimesh.Trimesh(a['vertices_mm'],a['faces'][job['faces']],process=False)
 cs=cylinder_candidates(m,.05)
 assert len(cs)==1
 c=cs[0]
 assert abs(c['radius_mm']-job['radius'])<1e-7 and abs(c['axial_length_mm']-job['height'])<1e-7
 fits.append({'instance':job['instance'],'source_face_min':min(job['faces']),'radius_mm':c['radius_mm'],'max_surface_residual_mm':c['max_surface_residual_mm']})
json.dump(fits,open(sys.argv[2],'w'))'''
        subprocess.run([sys.executable,'-B','-c',code,str(job),str(out),str(ROOT/'validation/runs/03-oled-retro/analysis')],cwd=baseline_src,check=True)
        refits=read(out)
    assert len(support_checks)==4 and len(refits)==24
    sweep=read(ROOT/'validation/parameter-sweep.json')
    assert all(production[name]==digest for name,digest in sweep['production_source_hashes'].items())
    assert sweep['expected_support_matches']==24 and sweep['negative_cases_with_any_cylinder']==0
    tests=(ROOT/'validation/all-tests-final.log').read_text()
    assert re.search(r'Ran 400 tests in [0-9.]+s\s+OK',tests) and (ROOT/'validation/all-tests-final.exit').read_text().strip()=='0'
    summary={'verification_passed':True,'analyzer_version':'0.5.0','v041_baseline_hashes_intact':True,
             'production_source_hashes':production,'unit_tests_passed':400,'checked_corpus_cases':len(results),
             'complete_section_gzip_files_checked':gzip_count,'saved_mesh_array_pairs_checked':arrays,
             'known_missed_supports_recovered':support_checks,'retro_local_candidates_refit_with_unchanged_v041':refits,
             'real_baseline_direction_files_with_identical_exact_failures':exact_failures_equal,
             'real_primary_direction_files_identical':primary_identical,
             'real_primary_direction_files_changed_from_new_candidate_guided_frame':primary_changed,
             'event_probe_sides_checked_all_corpus':probe_sides,'cases':results,
             'limitations':['Known four patches are two positions repeated in two instances, not full-model recall ground truth.',
                            'Unchanged original fitter acceptance is not independent semantic or bidirectional CAD ground truth.',
                            'No SLDPRT, Boolean material, trim topology or manufacturing settings have been recovered.']}
    write(ROOT/'validation/verification.json',summary);write(ROOT/'validation/local-candidate-inventory.json',all_local)
    print({key:summary[key] for key in ('verification_passed','checked_corpus_cases','complete_section_gzip_files_checked','saved_mesh_array_pairs_checked')})


if __name__=='__main__':main()
