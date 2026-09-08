import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d

from vectra3d import photo_compare as pc, quality


def face_plane(bump=False):
    x,y=np.meshgrid(np.arange(-70,71,5.),np.arange(-90,91,5.))
    z=np.full(x.shape,60.)
    if bump:
        z+=3*np.exp(-((x+35)**2+(y+25)**2)/100)
    vertices=np.c_[x.ravel(),y.ravel(),z.ravel()]
    grid=np.arange(x.size).reshape(x.shape)
    a,b,c,d=grid[:-1,:-1].ravel(),grid[:-1,1:].ravel(),grid[1:,1:].ravel(),grid[1:,:-1].ravel()
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                   o3d.utility.Vector3iVector(np.vstack((np.c_[a,b,c],np.c_[a,c,d]))))
    mesh.compute_vertex_normals()
    landmarks=np.tile([0.,0.,60.],(478,1))
    angle=np.linspace(np.pi/2,np.pi/2-2*np.pi,len(pc.FACE_OVAL),endpoint=False)
    landmarks[pc.FACE_OVAL]=np.c_[60*np.cos(angle),80*np.sin(angle),np.full(len(angle),60.)]
    coords=np.array([[-35,30],[-15,30],[15,30],[35,30],[0,25],[0,20],
                     [0,45],[0,65],[0,80],[-20,50],[20,50],[-30,55],[30,55],[-40,45],[40,45]])
    landmarks[pc.STABLE_LANDMARKS]=np.c_[coords,np.full(len(coords),60.)]
    return mesh,landmarks


class PhotoComparisonTests(unittest.TestCase):
    def test_rigid_fit_has_no_scale_and_rejects_degenerate_landmarks(self):
        source=np.array([[0.,0,0],[1,0,0],[0,2,0],[0,0,3]])
        transform,rms=pc.rigid_fit(source,source+[2,4,6])
        np.testing.assert_allclose(transform[:3,:3],np.eye(3),atol=1e-12)
        self.assertLess(rms,1e-12)
        with self.assertRaises(quality.QualityError):
            pc.rigid_fit(np.zeros((4,3)),np.zeros((4,3)))

    def test_intercanthal_normalization_removes_scale_without_mutating_inputs(self):
        before,bl=face_plane();after=o3d.geometry.TriangleMesh(before).scale(1.2,center=np.zeros(3))
        al=bl*1.2
        original=np.asarray(after.vertices).copy()
        b,a,_,_,calibration,checks=pc.align_surfaces(before,after,bl,al,32.)
        np.testing.assert_allclose(np.asarray(a.vertices),np.asarray(b.vertices),atol=1e-6)
        np.testing.assert_array_equal(np.asarray(after.vertices),original)
        self.assertAlmostEqual(calibration['relative_after_scale'],1/1.2)
        self.assertEqual(calibration['source'],'user_measurement')
        self.assertTrue(checks['alignment']['eligible'])

    def test_lower_face_bump_survives_upper_face_alignment(self):
        before,bl=face_plane();after,al=face_plane(bump=True)
        b,a,points,_,_,_=pc.align_surfaces(before,after,bl,al,30.)
        regions=pc.projected_region_volumes(b,a,points)
        self.assertGreater(regions[0]['volume_ml'],0.8)
        self.assertLess(regions[0]['volume_ml'],1.1)
        self.assertAlmostEqual(regions[1]['volume_ml'],0.,places=2)

    def test_missing_surface_never_becomes_zero_volume(self):
        before,bl=face_plane();after,_=face_plane()
        after.remove_vertices_by_mask(np.asarray(after.vertices)[:,0]<0)
        regions=pc.projected_region_volumes(before,after,bl)
        self.assertIsNone(regions[0]['volume_ml'])
        self.assertLess(regions[0]['coverage'],1)

    def test_back_facing_exit_is_not_a_duplicate_but_overlapping_entries_are(self):
        before,landmarks=face_plane()
        back=o3d.geometry.TriangleMesh(before).translate([0,0,-5])
        back.triangles=o3d.utility.Vector3iVector(np.asarray(back.triangles)[:,::-1].copy())
        shell=before+back
        regions=pc.projected_region_volumes(shell,shell,landmarks)
        self.assertTrue(all(r['volume_ml']==0 for r in regions))
        duplicate=before+o3d.geometry.TriangleMesh(before).translate([0,0,-2])
        regions=pc.projected_region_volumes(before,duplicate,landmarks)
        self.assertTrue(all(r['volume_ml'] is None for r in regions))

    def test_uncalibrated_comparison_has_no_cc_and_no_bias_subtraction(self):
        before,bl=face_plane();after,al=face_plane(bump=True)
        with tempfile.TemporaryDirectory() as out, \
                patch.object(pc,'closeup_landmarks',side_effect=[(before,bl,'b'),(after,al,'a')]), \
                patch.object(pc.quality,'mesh_fingerprint',side_effect=['b','a']), \
                patch.object(pc.analyze,'subtract_bias_field') as bias:
            result=pc.compare_assets('before.npz','after.npz',out)
            self.assertTrue(result['diagnostic_only'])
            self.assertIsNone(result['net_significant_volume_ml'])
            self.assertNotIn('regional_volumes',result)
            self.assertEqual(result['source'],'photo')
            self.assertTrue((Path(out)/'heatmap.ply').is_file())
            self.assertTrue(json.loads((Path(out)/'result.json').read_text())['diagnostic_only'])
            bias.assert_not_called()

    def test_point_to_surface_check_is_independent_of_tessellation(self):
        mesh,landmarks=face_plane()
        denser=mesh.subdivide_midpoint(2)
        result=pc.surface_alignment(mesh,denser,landmarks,1.)
        self.assertTrue(result['eligible'])
        self.assertLess(max(r['rms_reference_mm'] for r in result['directions']),1e-4)


if __name__=='__main__':
    unittest.main()
