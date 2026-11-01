"""Small independent checks of the algebra and evidence comparisons."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io
import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce
from metamaterials.certify import certify
from metamaterials.constructions import hs_corner, xy_of
from metamaterials.elastic import solid_moduli, iso_tensor, rotate_mandel, V_H, LAM_S
from metamaterials.evidence import read_designs
from metamaterials.geometry import wall_section
from metamaterials.laminate import gamma_op, laminate, mp_laminate
from metamaterials.metrics import finite_scale_table
from metamaterials.zero_mode import dilation_residual


class AlgebraChecks(unittest.TestCase):
    def test_hs_attainment(self):
        for nu in (.3, .45):
            for theta in (.01, .3, .7):
                x, y, defect = xy_of(hs_corner(nu, theta), nu, theta)
                np.testing.assert_allclose([x, y], [1, 1], atol=2e-12, rtol=0)
                self.assertLess(defect, 1e-12)

    def test_green_operator_invariants(self):
        for nu in (.3, .45):
            k, m = solid_moduli(nu)
            solid = iso_tensor(k, m)
            for n in ([1, 0, 0], [1, 2, -3], [-2, .1, 1]):
                g = gamma_op(n, solid)
                self.assertAlmostEqual(V_H @ g @ V_H, 1 / (3*k+4*m), places=12)
                self.assertAlmostEqual(np.trace(LAM_S @ g), 3*(k+2*m)/(m*(3*k+4*m)), places=12)
                self.assertGreater(np.linalg.eigvalsh(np.linalg.inv(solid)-g).min(), -1e-12)

    def test_uniform_material(self):
        c = iso_tensor(*solid_moduli(.3))
        with mp.workdps(45):
            for n in ([1, 2, 3], [0, 0, 1]):
                actual = np.array(mp_laminate([c, c], [.37, .63], n, mp.mpf('1e-30')).tolist(), float)
                np.testing.assert_allclose(actual, c, atol=1e-14, rtol=0)

    def test_diagonal_layer_closed_form(self):
        a, b = np.diag([3., 4., 5., 6., 7., 8.]), np.diag([9., 10., 11., 12., 13., 14.])
        fraction = .3
        expected = fraction * a + (1-fraction) * b
        for i in [2, 3, 4]:
            expected[i, i] = 1 / (fraction / a[i,i] + (1-fraction) / b[i,i])
        np.testing.assert_allclose(laminate([a,b], [fraction,1-fraction], [0,0,1]), expected, atol=1e-13)

    def test_rotation_covariance(self):
        a = iso_tensor(*solid_moduli(.3)); b = .2 * a
        q, _ = np.linalg.qr(np.array([[1.,2.,4.],[-3.,1.,2.],[2.,0.,1.]]))
        n = np.array([1., 2., 3.])
        actual = laminate([rotate_mandel(a,q),rotate_mandel(b,q)], [.4,.6], q@n)
        expected = rotate_mandel(laminate([a,b],[.4,.6],n),q)
        np.testing.assert_allclose(actual, expected, atol=2e-14)

    def test_small_geometry_certification(self):
        design = dict(D=1,K=1,base='B',leaf_n=[[[0,0,1]],[[0,0,1]]],
                      leaf_p=[[[1,0,0]],[[1,0,0]]],leaf_eta=[[0.],[0.]],
                      m=[[[0,1,0]]],f=[[.4]])
        value = certify(design,.3,.2,eps_list=[1e-6],deltas=['1e-25'],dps=45)
        self.assertAlmostEqual(value['runs'][0]['theta_total'],.2,places=14)
        self.assertTrue(np.isfinite(value['runs'][0]['y']))
        self.assertNotIn('torch',sys.modules)

    def test_compatible_dilation_is_sensitive(self):
        record = next(r for r in read_designs() if r['file']=='zero_bulk_aniso_nu0.3_th0.1.json')
        d = record['design']; alpha = list(d['zero_mode_alpha'])
        with mp.workdps(50):
            good = np.linalg.norm([float(x) for x in dilation_residual(d,alpha)])
            alpha[0] += .01
            bad = np.linalg.norm([float(x) for x in dilation_residual(d,alpha)])
        self.assertLess(good, 5e-13)
        self.assertGreater(bad,1e-4)

    def test_finite_scale_reapplication(self):
        rows=finite_scale_table()
        self.assertEqual(len(rows),11)
        self.assertLess(max(abs(r['relative_error']-r['expected_error']) for r in rows),1e-12)
        self.assertLess(max(r['formula_tensor_difference'] or 0 for r in rows),1e-12)

    def test_reflection_of_binary_geometry(self):
        plus = wall_section(32, p2=16, level='A', sign=1)
        minus = wall_section(32, p2=16, level='A', sign=-1)
        np.testing.assert_array_equal(minus, plus[(3 - np.arange(32)) % 32])

    def test_mismatch_exits_nonzero(self):
        records=read_designs()
        changed=next(r for r in records if r['file']=='zero_bulk_aniso_nu0.3_th0.1.json')
        for run in changed['runs']:
            run['y'] += .2
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'results') as tmp:
            target=Path(tmp)/'result.json'
            with patch.object(reproduce,'read_designs',return_value=records), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code=reproduce.main(['--stored-only','--output',str(target)])
            self.assertEqual(code,1)
            self.assertFalse(json.loads(target.read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
