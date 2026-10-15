import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import trimesh
from reverse_assistant import transform, fit_circle, load_3mf, analyze
from build_solidworks import validate

class ReverseTests(unittest.TestCase):
    def test_transform(self):
        m=transform('0 1 0 -1 0 0 0 0 1 10 20 30')
        np.testing.assert_allclose(m@np.array([1,0,0,1]),[10,21,30,1])
    def test_circle(self):
        t=np.linspace(0,2*np.pi,120,endpoint=False)
        c=fit_circle(np.column_stack((3+2*np.cos(t),4+2*np.sin(t))),.01)
        self.assertAlmostEqual(c['diameter_mm'],4)
        self.assertIsNone(fit_circle([[0,0],[3,0],[3,2],[0,2]],.01))
    def test_build_gate(self):
        with self.assertRaises(ValueError):
            validate({'schema_version':1,'strategy':'review_required_plate_profile','approved_for_build':False})
    def test_3mf_units_components_and_analysis(self):
        mesh=trimesh.creation.box(extents=[2,3,.1])
        vertices=''.join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x,y,z in mesh.vertices)
        faces=''.join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a,b,c in mesh.faces)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'box.3mf'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('3D/3dmodel.model','<model unit="centimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"><resources><object id="2"><components><component objectid="1" p:path="/3D/Objects/part.model" transform="1 0 0 0 1 0 0 0 1 1 2 3"/></components></object></resources><build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 4 5 6"/></build></model>')
                z.writestr('3D/Objects/part.model',f'<model unit="centimeter"><resources><object id="1"><mesh><vertices>{vertices}</vertices><triangles>{faces}</triangles></mesh></object></resources></model>')
            m=load_3mf(p)[0]
            np.testing.assert_allclose(m.extents,[20,30,1])
            np.testing.assert_allclose(m.bounds.mean(axis=0),[50,70,90])
            plan=analyze(p,Path(d)/'out')
            self.assertAlmostEqual(plan['volume_difference_percent'],0)
            self.assertTrue((Path(d)/'out/report.html').stat().st_size>0)

if __name__=='__main__': unittest.main()
