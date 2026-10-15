"""Bounded local cylinder hypotheses from mesh normal changes, never CAD/repair.

Global smooth bands may mix straight cylinder walls and tangent blends. Local
axis proposals come from neighboring nonparallel source normals, NOT world
XYZ, source component axes, filenames or known sample IDs. All hypotheses must
pass the unchanged strict cylinder fitter on edge-connected original faces.
Recovery also needs an observable axial run, not a one-cell equatorial belt
picked from a sphere/torus merely because two rings happen to have equal radii.
The latter is a conservative observation heuristic, not a universal classifier.
"""
from __future__ import annotations

import numpy as np

# Proposal budgets, not accuracy tolerances or hard CPU/memory guarantees.
MAX_AXIS_REPRESENTATIVES = 4096
MAX_AXES_PER_BAND = 12
MAX_AXES_PER_BODY = 96
MAX_REGION_FITS_PER_BODY = 192
MAX_REGIONS_PER_AXIS = 8
MAX_PROPOSAL_CLUSTERS = 64
MIN_NORMAL_CHANGE_DEGREES = .05
AXIS_CLUSTER_DEGREES = .35
AXIS_FACE_PERPENDICULAR_DEGREES = .05
MIN_AXIAL_RADIUS_RATIO = .25
MIN_RUN_TO_SMOOTH_NEIGHBOR_RATIO = 2.


def _proposals(data, region, links):
    """Area/bend weighted neighboring-normal directions, deterministic and bounded.

    Rounding is ONLY a 1e-6 direction proposal dedup key; the selected direction
    is an actual cross product, never rounded coordinates or repaired normals.
    Final axes/accuracy are refit from original triangles by the strict fitter.
    """
    from geometry_analysis import _canonical
    if not links:
        return [], {'normal_change_edge_count':0,'axis_representative_count':0,
                    'axis_representatives_omitted':0}
    pairs=np.asarray(links,dtype=int)
    ns=data['normals']
    cross=np.cross(ns[pairs[:,0]],ns[pairs[:,1]])
    lengths=np.linalg.norm(cross,axis=1)
    dots=np.einsum('ij,ij->i',ns[pairs[:,0]],ns[pairs[:,1]])
    keep=(lengths>=np.sin(np.deg2rad(MIN_NORMAL_CHANGE_DEGREES))) & (dots>0)
    pairs,cross,lengths=pairs[keep],cross[keep],lengths[keep]
    if not len(pairs):
        return [], {'normal_change_edge_count':0,'axis_representative_count':0,
                    'axis_representatives_omitted':0}
    axes=cross/lengths[:,None]
    major=np.argmax(np.abs(axes),axis=1)
    axes[axes[np.arange(len(axes)),major]<0]*=-1
    weights=np.minimum(data['areas'][pairs[:,0]],data['areas'][pairs[:,1]])*lengths
    _,first,inverse=np.unique(np.round(axes,6),axis=0,return_index=True,return_inverse=True)
    totals=np.bincount(inverse,weights=weights)
    counts=np.bincount(inverse)
    order=np.argsort(-totals,kind='stable')
    omitted=max(0,len(order)-MAX_AXIS_REPRESENTATIVES)
    order=order[:MAX_AXIS_REPRESENTATIVES]
    representatives=axes[first[order]]
    totals,counts=totals[order],counts[order]
    active=np.ones(len(order),dtype=bool)
    choices=[]
    threshold=np.cos(np.deg2rad(AXIS_CLUSTER_DEGREES))
    for index in range(len(order)):
        if not active[index]:continue
        axis=representatives[index]
        neighbors=active & (np.abs(representatives @ axis)>=threshold)
        # Use source votes for an area-weighted line direction, with signs aligned.
        signed=representatives[neighbors]*np.where(representatives[neighbors] @ axis<0,-1.,1.)[:,None]
        refined=_canonical(np.sum(totals[neighbors,None]*signed,axis=0))
        choices.append({'axis':refined,'vote_weight':float(totals[neighbors].sum()),
                        'normal_change_edge_count':int(counts[neighbors].sum())})
        active[neighbors]=False
        if len(choices)>=MAX_PROPOSAL_CLUSTERS:
            break
    choices.sort(key=lambda item:-item['vote_weight'])
    return choices, {'normal_change_edge_count':len(pairs),'axis_representative_count':len(order),
                     'axis_representatives_omitted':omitted,'axis_cluster_count':len(choices),
                     'unclustered_axis_representative_count':int(active.sum())}


def _observable_run(data, region, parent, candidate):
    """Reject under-observed belts; never turn a short smooth-cell strip into CAD."""
    axis=np.asarray(candidate['axis'])
    height=candidate['axial_length_mm']
    radius=candidate['radius_mm']
    stats={'max_axis_normal_residual_degrees':candidate['max_axis_normal_residual_degrees'],
           'maximum_axis_normal_residual_degrees':AXIS_FACE_PERPENDICULAR_DEGREES,
           'axial_length_over_radius':height/radius,
           'minimum_axial_length_over_radius':MIN_AXIAL_RADIUS_RATIO,
           'observation_kind':'local_recovered_support_not_design_feature',
           'smooth_boundary_axial_step_mm':{},'run_over_neighbor_step':{}}
    if candidate['max_axis_normal_residual_degrees']>AXIS_FACE_PERPENDICULAR_DEGREES:
        return False,'local_recovery_axis_normal_evidence_insufficient',stats
    if height<MIN_AXIAL_RADIUS_RATIO*radius:
        return False,'insufficient_axial_run_relative_to_radius',stats
    selected=set(region)
    projection=data['triangles'] @ axis
    lo,hi=float(projection[region].min()),float(projection[region].max())
    numeric=max(1e-10*height,128*np.finfo(float).eps*data['scale'])
    steps={'below':[],'above':[]}
    # Use complete strict coplanar facets, not the immediate triangle's size:
    # subdivision of an unchanged curved cell must not create a longer-looking
    # straight run. This cache contains only derived topology, not mesh edits.
    if '_local_cylinder_strict_facets' not in data:
        from geometry_analysis import _planar_regions
        facets = _planar_regions(data, 0., strict=True)
        labels = np.empty(len(data['faces']), dtype=int)
        for index, facet in enumerate(facets):
            labels[facet] = index
        data['_local_cylinder_strict_facets'] = (facets, labels)
    facets, labels = data['_local_cylinder_strict_facets']
    seen_neighbors=set()
    stats['neighbor_span_basis']='whole_connected_strict_coplanar_facet_not_incident_triangle'
    parent_set=set(parent)
    threshold=np.cos(np.deg2rad(35.))
    # A genuine straight run beside a sampled fillet should be resolved beyond
    # one neighboring curved cell. An accidental symmetric torus/sphere belt
    # has about the same axial length as its immediate curved neighbors.
    for a,b in data['adjacency']:
        if (a in selected)==(b in selected):continue
        inside,outside=(a,b) if a in selected else (b,a)
        if outside not in parent_set or np.dot(data['normals'][inside],data['normals'][outside])<threshold:
            continue
        label=int(labels[outside])
        if label in seen_neighbors:continue
        seen_neighbors.add(label)
        z=projection[facets[label]].ravel()
        span=float(np.ptp(z))
        if span<=numeric:continue
        if z.min()<lo-numeric:steps['below'].append(span)
        if z.max()>hi+numeric:steps['above'].append(span)
    for side,values in steps.items():
        if values:
            step=float(np.median(values))
            stats['smooth_boundary_axial_step_mm'][side]=step
            stats['run_over_neighbor_step'][side]=height/step
    if (len(stats['run_over_neighbor_step'])==2 and
            min(stats['run_over_neighbor_step'].values())<MIN_RUN_TO_SMOOTH_NEIGHBOR_RATIO):
        return False,'axial_run_not_separated_from_two_sided_smooth_cell_belt',stats
    return True,None,stats


def recover_cylinders(data,tolerance,baseline,bands,fit_region):
    """Return new fits + diagnostics; no accepted face is silently reassigned."""
    from geometry_analysis import _regions
    diagnostics={'method':'neighbor_normal_cross_axes_then_connected_perpendicular_subregions',
                 'source_triangle_id_scope':'body_local_faces',
                 'candidate_geometry_only':True,'mesh_repair_performed':False,
                 'axis_proposal_source':'source_manifold_adjacent_face_normals_not_world_or_component_axes',
                 'limits':{'max_axis_representatives_per_band':MAX_AXIS_REPRESENTATIVES,
                           'max_axes_per_band':MAX_AXES_PER_BAND,'max_axes_per_body':MAX_AXES_PER_BODY,
                           'max_region_fits_per_body':MAX_REGION_FITS_PER_BODY,
                           'max_regions_per_axis':MAX_REGIONS_PER_AXIS,
                           'max_axis_proposal_clusters':MAX_PROPOSAL_CLUSTERS},
                 'budget_allocation':'round_robin_bands_first_axis_before_second; largest_connected_supports_first',
                 'normal_cluster_degrees':AXIS_CLUSTER_DEGREES,
                 'face_axis_perpendicular_gate_degrees':AXIS_FACE_PERPENDICULAR_DEGREES,
                 'baseline_candidate_count':len(baseline),'bands_considered':0,
                 'axis_hypotheses_attempted':0,'connected_region_fit_attempts':0,
                 'axis_hypotheses_omitted':0,'axis_representatives_omitted':0,
                 'region_fits_omitted':0,'body_axis_budget_reached':False,
                 'accepted_local_candidate_count':0,'rejection_counts':{},'bands':[],
                 'search_exhaustive':False,'candidate_recall_certified':False,
                 'limitations':['Bounded deterministic axis proposals can omit valid local supports.',
                                'Short axial runs and one-cell curved belts are conservatively unresolved.',
                                'One-sided analytic support error is not CAD/trim/role equivalence.']}
    covered={i for c in baseline for i in c['source_triangle_ids']}
    accepted=[]
    seen_regions=set()
    threshold=np.cos(np.deg2rad(35.))
    face_gate=np.sin(np.deg2rad(AXIS_FACE_PERPENDICULAR_DEGREES))
    labels=np.full(len(data['faces']),-1,dtype=int)
    for index,band in enumerate(bands):labels[band]=index
    band_links=[[] for _ in bands]
    for a,b in data['adjacency']:
        index=labels[a]
        if index>=0 and labels[b]==index and np.dot(data['normals'][a],data['normals'][b])>=threshold:
            band_links[index].append((a,b))
    def reject(reason):
        diagnostics['rejection_counts'][reason]=diagnostics['rejection_counts'].get(reason,0)+1
    work=[]
    for band_index,parent in enumerate(bands):
        if len(parent)<6 or set(parent).issubset(covered):continue
        diagnostics['bands_considered']+=1
        proposals,stats=_proposals(data,parent,band_links[band_index])
        omitted=max(0,len(proposals)-MAX_AXES_PER_BAND)
        diagnostics['axis_hypotheses_omitted']+=omitted
        diagnostics['axis_representatives_omitted']+=stats['axis_representatives_omitted']
        diagnostics['bands'].append({'source_triangle_ids':parent,'triangle_count':len(parent),
                                     **stats,'axis_hypotheses_retained':min(len(proposals),MAX_AXES_PER_BAND),
                                     'axis_hypotheses_omitted':omitted})
        work.append((band_index,parent,proposals[:MAX_AXES_PER_BAND]))
    # A huge first mixed band must not consume every local-fit budget before
    # small later features get even their best axis hypothesis tried.
    for rank in range(MAX_AXES_PER_BAND):
        for band_index,parent,proposals in work:
            if rank>=len(proposals):continue
            hypothesis=proposals[rank]
            if diagnostics['axis_hypotheses_attempted']>=MAX_AXES_PER_BODY:
                diagnostics['body_axis_budget_reached']=True
                diagnostics['axis_hypotheses_omitted']+=1
                continue
            diagnostics['axis_hypotheses_attempted']+=1
            ids=np.asarray(parent,dtype=int)
            ids=ids[np.abs(data['normals'][ids] @ hypothesis['axis'])<=face_gate]
            eligible=set(ids.tolist())-covered
            links=[(a,b) for a,b in band_links[band_index] if a in eligible and b in eligible]
            groups=[g for g in _regions(eligible,links) if len(g)>=6 and tuple(g) not in seen_regions]
            groups.sort(key=lambda g:(-len(g),-float(data['areas'][g].sum()),g[0]))
            diagnostics['region_fits_omitted']+=max(0,len(groups)-MAX_REGIONS_PER_AXIS)
            for region in groups[:MAX_REGIONS_PER_AXIS]:
                key=tuple(region)
                seen_regions.add(key)
                if diagnostics['connected_region_fit_attempts']>=MAX_REGION_FITS_PER_BODY:
                    diagnostics['region_fits_omitted']+=1
                    continue
                diagnostics['connected_region_fit_attempts']+=1
                fits=fit_region(region)
                if not fits:
                    reject('strict_cylinder_fit_rejected')
                    continue
                for fit in fits:
                    observable,reason,observation=_observable_run(data,region,parent,fit)
                    if not observable:
                        reject(reason)
                        continue
                    if set(fit['source_triangle_ids']) & covered:
                        reject('overlap_with_previously_accepted_cylinder')
                        continue
                    fit.update(segmentation_method='local_normal_axis_consensus',
                               local_segmentation_evidence={
                                   'source_triangle_id_scope':'body_local_faces',
                                   'parent_normal_band':{'source_triangle_ids':list(parent),
                                                         'source_triangle_id_scope':'body_local_faces'},
                                   'axis_proposal':hypothesis['axis'].tolist(),
                                   'axis_proposal_basis':'adjacent_source_face_normal_cross_product_consensus',
                                   'supporting_normal_change_edge_count':hypothesis['normal_change_edge_count'],
                                   'strict_fitter_unchanged':True,'axial_observability':observation,
                                   'geometry_or_normal_repair_performed':False,
                                   'global_parent_fit_failure_not_ignored':True})
                    accepted.append(fit)
                    covered.update(fit['source_triangle_ids'])
    diagnostics['accepted_local_candidate_count']=len(accepted)
    return accepted,diagnostics
