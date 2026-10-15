"""Immutable-input full pipeline, strict-reader comparison, no tuned per-case options."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from assistant import VERSION,process
from threemf_io import load_document


def main():
    parser=argparse.ArgumentParser();parser.add_argument('id');parser.add_argument('input',type=Path)
    parser.add_argument('--runs-root',type=Path,default=ROOT/'validation/reruns',
                        help='Fresh output root; existing case directories are refused')
    args=parser.parse_args()
    out=args.runs_root/args.id
    out.mkdir(parents=True,exist_ok=False)
    state={'case_id':args.id,'source_name':args.input.name,'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
           'analyzer_version':VERSION,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')},
           'default_options_unchanged':True,'source_repair_performed':False}
    for strict in [True,False]:
        start=time.perf_counter()
        try:
            d=load_document(args.input,geometry_only=not strict,isolate_build_errors=not strict)
            result={'readable':True,'instances':len(d.instances),'faces':sum(len(i.mesh.faces) for i in d.instances),
                    'document_coverage_complete':d.metadata['document_coverage_complete'],
                    'role_mapping_status':d.metadata['role_mapping_status'],'slicer_roles':d.metadata['slicer_metadata']['resolved_role_occurrence_counts']}
        except Exception as exc:result={'readable':False,'error_type':type(exc).__name__,'error':str(exc)}
        result['elapsed_seconds']=time.perf_counter()-start
        state['strict_reader' if strict else 'compatible_reader']=result
    start=time.perf_counter()
    try:
        r=process(args.input,out/'analysis')
        result={'produced_report':True,'status':r['status'],'instance_count':r['instance_count'],'shell_count':r['body_count'],
                'source_faces':sum(b['triangles'] for b in r['bodies']),
                'surface_counts':{k:sum(len(b[k]) for b in r['bodies']) for k in ('planes','cylinders','cones','spheres')},
                'local_cylinder_count':sum(sum(c.get('segmentation_method')=='local_normal_axis_consensus' for c in b['cylinders']) for b in r['bodies']),
                'successful_primary_sections':sum(e.get('summary',{}).get('sample_count',0) for b in r['bodies'] for e in b['section_analyses']),
                'failed_primary_sections':sum(e.get('summary',{}).get('failed_sample_count',0) for b in r['bodies'] for e in b['section_analyses']),
                'failed_directions':sum(e['status']=='failed' for b in r['bodies'] for e in b['section_analyses']),
                'event_probe_summary':r['event_probe_summary'],
                'candidate_analysis_coverage_complete':r['candidate_analysis_coverage_complete'],
                'candidate_recall_certified':r['candidate_recall_certified'],'has_candidate_search_limits':r['has_candidate_search_limits'],
                'has_candidate_analysis_errors':r['has_candidate_analysis_errors'],'has_section_analysis_errors':r['has_section_analysis_errors'],
                'has_reconstruction_blockers':r['has_reconstruction_blockers'],'analysis_budget':r['analysis_budget'],
                'slicer_metadata':{k:r['package']['slicer_metadata'][k] for k in ('role_mapping_status','negative_component_occurrence_count','resolved_role_occurrence_counts','config_issues')},
                'cad_reconstructed':False}
    except Exception as exc:
        traceback.print_exc()
        result={'produced_report':False,'error_type':type(exc).__name__,'error':str(exc)}
    result.update(elapsed_seconds=round(time.perf_counter()-start,3),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    state['pipeline']=result
    (out/'result.json').write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
