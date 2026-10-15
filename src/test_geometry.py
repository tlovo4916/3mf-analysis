import unittest
import numpy as np
import trimesh
from geometry_analysis import plate_frame, cylinder_candidates, inspect_mesh

class GeometryTests(unittest.TestCase):
    def test_rotated_plate(self):
        m=trimesh.creation.box(extents=[100,70,2])
        rotation=trimesh.transformations.rotation_matrix(.83,[1,2,3])
        m.apply_transform(rotation);m.apply_translation([17,-9,35])
        r,o,local=plate_frame(m)
        self.assertAlmostEqual(local.extents[2],2,places=6)
        self.assertAlmostEqual(np.linalg.det(r),1,places=6)
        np.testing.assert_allclose(local.vertices@r.T+o,m.vertices,atol=1e-7)
        self.assertEqual(inspect_mesh(m,0,0)['strategy'],'plate_profile_review')
    def test_cylinder(self):
        m=trimesh.creation.cylinder(radius=5,height=20,sections=128)
        m.apply_transform(trimesh.transformations.rotation_matrix(.7,[1,1,0]))
        candidates=cylinder_candidates(m)
        self.assertTrue(candidates)
        self.assertAlmostEqual(candidates[0]['radius_mm'],5,places=5)
        self.assertAlmostEqual(candidates[0]['axial_length_mm'],20,places=5)
    def test_box_not_cylinder(self):
        m=trimesh.creation.box(extents=[10,20,30])
        self.assertEqual(cylinder_candidates(m),[])
        self.assertEqual(inspect_mesh(m,0,0)['strategy'],'mechanical_feature_review')
    def test_open_mesh(self):
        m=trimesh.creation.box();m.update_faces(np.arange(len(m.faces)-1))
        self.assertEqual(inspect_mesh(m,0,0)['strategy'],'repair_required')

if __name__=='__main__':unittest.main()
