import unittest

import numpy as np

from missing_impute_utils import generate_modal_missing


class MissingUtilsTest(unittest.TestCase):
    def test_mixed_missing_and_mask_schema(self):
        x = {
            "m1": np.ones((20, 5), dtype=np.float32),
            "m2": np.ones((20, 4), dtype=np.float32),
        }
        miss, mask = generate_modal_missing(
            x,
            missing_rate=0.3,
            target_modalities=["m1", "m2"],
            missing_type="mixed",
            mixed_sample_ratio=0.7,
            seed=7,
            return_mask_dict=True,
        )
        self.assertIn("modal_feature_observed", mask)
        self.assertIn("modal_sample_observed", mask)
        self.assertEqual(mask["modal_feature_observed"]["m1"].shape, x["m1"].shape)
        self.assertTrue(np.isnan(miss["m1"]).any() or np.isnan(miss["m2"]).any())


if __name__ == "__main__":
    unittest.main()
