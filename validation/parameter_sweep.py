"""Deterministic unseen parameter combinations; synthetic, not exporter ground truth."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from geometry_analysis import cylinder_candidates
from test_cylinder_segmentation import blended_wall


def run():
    start=time.perf_counter();rng=np.random.default_rng(510723);records=[]
    for index in range(24):
        radius=float(rng.uniform(.5,8.));height=float(radius*rng.uniform(.6,4.))
        blend=float(radius*rng.uniform(.08,.4));sections=int(rng.choice([16,24,32,48]))
        coverage=float(rng.choice([80.,135.,270.,360.]));steps=int(rng.integers(4,11))
        mesh=blended_wall(radius,height,blend,sections,coverage,steps)
        transform=trimesh.transformations.rotation_matrix(float(rng.uniform(.1,3.)),rng.normal(size=3))
        transform[:3,3]=rng.uniform(-1e6,1e6,3);mesh.apply_transform(transform)
        before=(mesh.vertices.copy(),mesh.faces.copy());cs=cylinder_candidates(mesh)
        matches=[c for c in cs if abs(c['radius_mm']-radius)<1e-5 and abs(c['axial_length_mm']-height)<1e-5
                 and abs(np.dot(c['axis'],transform[:3,2]))>1-1e-9]
        assert np.array_equal(mesh.vertices,before[0]) and np.array_equal(mesh.faces,before[1])
        records.append({'case':index,'kind':'blended_cylindrical_support','parameters':{'radius':radius,'height':height,'blend':blend,'sections':sections,'coverage':coverage,'steps':steps},
                        'rigid_transform':transform.tolist(),'expected_support_matched':len(matches)==1,'candidate_count':len(cs),
                        'surface_within_005mm':bool(matches and matches[0]['surface_approximation_within_tolerance']),
                        'source_arrays_unchanged':True})
    negative=[]
    for index in range(12):
        major=float(rng.uniform(.25,8.));minor=major*float(rng.uniform(.08,.4))
        mesh=trimesh.creation.torus(major_radius=major,minor_radius=minor,major_sections=int(rng.choice([24,32,48])),minor_sections=int(rng.choice([12,16,24])))
        transform=trimesh.transformations.rotation_matrix(float(rng.uniform(.1,3.)),rng.normal(size=3));mesh.apply_transform(transform)
        cs=cylinder_candidates(mesh)
        negative.append({'case':index,'kind':'smooth_torus_no_cylindrical_run','major_radius':major,'minor_radius':minor,'candidate_count':len(cs)})
    result={'seed':510723,'scope':'synthetic_parameter_sweep_not_population_recall_or_false_positive_rate',
            'new_parameters_not_used_in_implementation':True,'positive_cases':records,'negative_cases':negative,
            'expected_support_matches':sum(c['expected_support_matched'] for c in records),
            'negative_cases_with_any_cylinder':sum(c['candidate_count']>0 for c in negative),'elapsed_seconds':time.perf_counter()-start,
            'production_source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py') if not p.name.startswith('test_')}}
    (ROOT/'validation/parameter-sweep.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print({k:result[k] for k in ['expected_support_matches','negative_cases_with_any_cylinder','elapsed_seconds']})
    return result


if __name__=='__main__':run()
