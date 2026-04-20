import unittest

import numpy as np
import torch

from aemvc_fusion import AEMVCFusion


class AEMVCFusionTest(unittest.TestCase):
    def test_forward_shapes_and_loss(self):
        modal_dims = {"m1": 6, "m2": 4}
        model = AEMVCFusion(modal_dims=modal_dims, latent_dim=8)
        x1 = torch.tensor(np.random.randn(10, 6), dtype=torch.float32)
        x2 = torch.tensor(np.random.randn(10, 4), dtype=torch.float32)
        mask = {
            "modal_feature_observed": {
                "m1": np.ones((10, 6), dtype=bool),
                "m2": np.ones((10, 4), dtype=bool),
            }
        }
        mask["modal_feature_observed"]["m1"][0, :] = False
        out_x, out_k, loss = model({"m1": x1, "m2": x2}, mask)
        self.assertEqual(out_x["m1"].shape, x1.shape)
        self.assertEqual(out_k["m1"].shape, (10, 10))
        self.assertTrue(torch.is_tensor(loss))


if __name__ == "__main__":
    unittest.main()
