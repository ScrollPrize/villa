import unittest
import numpy as np

from vc_sampling_audit.slab import CONTRACT, report_grid


class SlabReportTests(unittest.TestCase):
    def grid(self):
        y,x=np.meshgrid([0.,1.],[0.,1.],indexing="ij")
        return np.stack([x,y,np.zeros_like(x)],axis=-1)

    def call(self, q, n, low=0., high=3., **kwargs):
        return report_grid(q,n,[0.,0.,1.],low,high,contract=kwargs.get("contract",CONTRACT),units="test-units")

    def test_interior_failure_missed_by_endpoints(self):
        q=self.grid(); n=np.stack([-.5*q[...,0],-.8*q[...,1],np.ones((2,2))],axis=-1)
        for d in self.call(q,n)["diagonals"].values():
            self.assertEqual(d["endpoint_only_projected_failures"],0)
            self.assertEqual(d["interior_projected_failures_missed_by_endpoints"],2)
            self.assertAlmostEqual(d["minimum_projected_area_ratio"],-.05625,places=12)
            self.assertAlmostEqual(d["failure_witnesses"][0]["projected_min_depth"],1.625,places=12)

    def test_constant_directions_nonunit(self):
        q=self.grid(); n=np.zeros_like(q); n[...,2]=2
        self.assertEqual(self.call(q,n)["diagonals"]["AC"]["volume_nonpositive_triangles"],0)

    def test_global_normal_reversal_is_only_depth_sign(self):
        q=self.grid(); n=np.zeros_like(q); n[...,2]=-1
        r=self.call(q,n)
        self.assertEqual(r["diagonals"]["AC"]["volume_nonpositive_triangles"],0)
        self.assertEqual(r["diagonals"]["AC"]["audit_direction_sign"],-1)

    def test_endpoint_double_root(self):
        q=self.grid(); n=-q.copy(); n[...,2]=1
        r=self.call(q,n,0,1)
        self.assertEqual(r["diagonals"]["AC"]["projected_nonpositive_triangles"],2)

    def test_interior_double_root(self):
        q=self.grid(); n=-q.copy(); n[...,2]=1
        r=self.call(q,n,0,2)
        self.assertEqual(r["diagonals"]["AC"]["interior_projected_failures_missed_by_endpoints"],2)

    def test_unsupported_normalization_contract_refused(self):
        with self.assertRaisesRegex(ValueError,"UNSUPPORTED_INTERPOLATION_CONTRACT"):
            self.call(self.grid(),self.grid(),contract="normalize-after-interpolation")

    def test_invalid_base_refused(self):
        with self.assertRaises(ValueError):
            self.call(np.zeros((2,2,3)),np.ones((2,2,3)))

    def test_nonfinite_input_refused(self):
        q=self.grid(); q[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            self.call(q,np.ones_like(q))

    def test_local_pass_never_certifies_global_or_anatomy(self):
        # Two separately supplied copies overlap exactly yet each is local-pass.
        q=self.grid(); n=np.zeros_like(q); n[...,2]=1
        for _ in range(2):
            r=self.call(q,n)
            self.assertEqual(r["global_injectivity"],"NOT_ASSESSED")
            self.assertEqual(r["anatomical_sheet_identity"],"NOT_ASSESSED")


if __name__ == "__main__":
    unittest.main()
