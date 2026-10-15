"""Integrated new-evidence contracts, not CAD execution or recall certification."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import trimesh

from assistant import process
from geometry_analysis import inspect_mesh
from reverse_assistant import section_stack_json, write_section_evidence
from section_analysis import analyze_sections
from test_section_event_probes import layered_prism
from test_threemf_io import model_xml, mesh_xml, assembly


class GeneralizationPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.source=self.root/'part.3mf'

    def package(self,mesh,config=None):
        obj='<object id="1">'+mesh_xml(mesh.vertices,mesh.faces)+'</object>'
        if config is not None:
            obj+=assembly(10,'<component objectid="1"/>');items='<item objectid="10"/>'
        else:items='<item objectid="1"/>'
        with zipfile.ZipFile(self.source,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('3D/3dmodel.model',model_xml(obj,items))
            if config is not None:archive.writestr('Metadata/model_settings.config',config)

    def test_event_probe_summary_pointers_are_lossless_and_export_is_pure(self):
        stack=analyze_sections(layered_prism([0,.5,.5,1],[1,1,1.2,1.2]))
        evidence=section_stack_json(stack);original=deepcopy(evidence)
        summary=write_section_evidence(evidence,self.root/'evidence')
        self.assertEqual(evidence,original)
        path=self.root/'evidence/sections.full.json.gz'
        self.assertEqual(summary['full_evidence']['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        with gzip.open(path,'rt',encoding='utf-8') as stream:full=json.load(stream)
        self.assertEqual(full,evidence)
        for event in summary['event_probes']:
            for side in event['sides']:
                self.assertEqual(side['status'],'success')
                for attempt in side['attempts']:
                    self.assertNotIn('evidence',attempt)
                    target=full
                    for token in attempt['full_evidence_json_pointer'].split('/')[1:]:
                        target=target[int(token)] if isinstance(target,list) else target[token]
                    self.assertIn('intersection_nodes',target)
                    self.assertEqual(target['z_mm'],attempt['actual_z_mm'])

    def test_finite_probes_cannot_promote_pipeline_completion_or_plan(self):
        self.package(layered_prism([0,.25,.25,1],[1,1,1.2,1.2]))
        # Force only the already validated local frame, making this an exact
        # integration fixture rather than relying on OBB eigenvalue ties.
        frame={'origin_mm':[0,0,0],'local_to_source_rotation':np.eye(3).tolist()}
        direction={'id':'primary','frame':frame,'is_plate_direction':True}
        with patch('assistant.section_directions',return_value=[direction]):
            report=process(self.source,self.root/'output')
        self.assertEqual(report['event_probe_summary']['successful_side_count'],2)
        self.assertFalse(report['event_probe_summary']['exact_failures_replaced'])
        self.assertEqual(report['status'],'analysis_partial_not_cad_complete')
        self.assertTrue(report['has_reconstruction_blockers'])
        self.assertTrue(report['has_section_analysis_errors'])
        self.assertFalse(report['section_analysis_attempts_complete'])
        self.assertFalse(report['candidate_recall_certified'])
        self.assertFalse(report['candidate_search_exhaustive'])
        self.assertTrue(all('plate_plan' not in body for body in report['bodies']))

    def test_body_roles_html_escape_and_whole_instance_veto(self):
        mesh=trimesh.creation.box(extents=[30,20,1])
        config=('<config><object id="10"><part id="1" subtype="negative_part">'
                '<metadata key="name" value="&lt;script&gt;do-not-run&lt;/script&gt;"/>'
                '</part></object></config>')
        self.package(mesh,config)
        report=process(self.source,self.root/'output',max_section_directions=1)
        body=report['bodies'][0]
        self.assertEqual(body['source_component_roles'][0]['slicer_role'],'negative_part')
        self.assertIn('slicer_non_material_role:negative_part',body['source_reconstruction_blockers'])
        self.assertTrue(report['has_reconstruction_blockers'])
        self.assertNotIn('plate_plan',body)
        text=(self.root/'output/index.html').read_text()
        self.assertNotIn('<script>',text)
        self.assertIn('negative_part',text)
        self.assertEqual(body['source_geometry_semantics'],
                         'retained_core_leaf_geometry_not_evaluated_printed_material')

    def test_local_budget_flag_does_not_claim_candidate_recall(self):
        mesh=trimesh.creation.icosphere(subdivisions=1)
        self.package(mesh)
        base=inspect_mesh(mesh,0,0)
        base['cylinder_segmentation_diagnostics']['axis_hypotheses_omitted']=1
        with patch('assistant.inspect_mesh',return_value=base):
            report=process(self.source,self.root/'output',max_section_directions=1)
        self.assertTrue(report['candidate_analysis_coverage_complete'])
        self.assertTrue(report['has_candidate_search_limits'])
        self.assertFalse(report['candidate_search_exhaustive'])
        self.assertFalse(report['candidate_recall_certified'])
        self.assertEqual(report['candidate_analysis_coverage_scope'],
                         'scheduled_source_faces_and_shells_not_candidate_recall')


if __name__=='__main__':unittest.main()
