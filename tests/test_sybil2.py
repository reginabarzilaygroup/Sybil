import glob
import os
import unittest

import pytest
import torch

from sybil import Sybil2, Serie
from sybil.serie import InputV2

T0_DIR = (
    "/data/rbg/shared/datasets/NLST/NLST/all_nlst-ct/set2/batch1/122361"
    "/T0/1.2.840.113654.2.55.81641439048624235905678753284956900652"
)
T1_DIR = (
    "/data/rbg/shared/datasets/NLST/NLST/all_nlst-ct/set2/batch1/122361"
    "/T1/1.2.840.113654.2.55.210451208063625047828616019396666958685"
)
MODEL_CACHE_DIR = "/data/rbg/users/pgmikhael/mammogram/sybil2_checkpoints"
SERIE_CACHE_DIR = os.path.expanduser("~/.sybil/test_sybil2_cache")


def glob_dcm(directory):
    return sorted(glob.glob(os.path.join(directory, "*.dcm")))


def skip_unless_regression():
    if not os.environ.get("SYBIL_TEST_RUN_REGRESSION", "false").lower() == "true":
        pytest.skip("Skipping long-running Sybil2 test. Set SYBIL_TEST_RUN_REGRESSION=true to run.")


class TestSybil2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_unless_regression()
        os.makedirs(SERIE_CACHE_DIR, exist_ok=True)
        cls.dicoms = {
            0: glob_dcm(T0_DIR),
            1: glob_dcm(T1_DIR),
        }

    def test_serie_v2_init(self):
        """Serie v2 initializes correctly with a two-timepoint dict."""
        skip_unless_regression()
        serie = Serie(self.dicoms, version="v2", cache_dir=SERIE_CACHE_DIR)

        self.assertTrue(serie._is_version2)
        self.assertEqual(set(serie._meta.keys()), {0, 1})
        for tp, meta in serie._meta.items():
            self.assertGreater(len(meta.paths), 0, f"No paths loaded for timepoint {tp}")
            self.assertGreater(meta.thickness, 0, f"Slice thickness <= 0 for timepoint {tp}")

    def test_serie_v2_get_volume(self):
        """serie.get_volume() returns a dict of InputV2 named tuples."""
        skip_unless_regression()
        serie = Serie(self.dicoms, version="v2", cache_dir=SERIE_CACHE_DIR)
        volume = serie.get_volume()

        self.assertIsInstance(volume, dict)
        self.assertEqual(set(volume.keys()), {0, 1})
        for tp, inp in volume.items():
            self.assertIsInstance(inp, InputV2, f"Expected InputV2 for timepoint {tp}")
            self.assertIsInstance(
                inp.segmentation_volume,
                torch.Tensor,
                f"segmentation_volume is not a Tensor for timepoint {tp}",
            )
            # rve_volume and lungmask_volume are optional
            if inp.rve_volume is not None:
                self.assertIsInstance(inp.rve_volume, torch.Tensor)
            if inp.lungmask_volume is not None:
                self.assertIsInstance(inp.lungmask_volume, torch.Tensor)

    def test_sybil2_predict(self):
        """End-to-end Sybil2.predict() returns 6 calibrated risk scores in [0, 1]."""
        skip_unless_regression()
        model = Sybil2(cache=MODEL_CACHE_DIR)
        serie = Serie(self.dicoms, version="v2", cache_dir=SERIE_CACHE_DIR)

        prediction = model.predict([serie])

        self.assertEqual(len(prediction.scores), 1)
        scores = prediction.scores[0]
        self.assertEqual(len(scores), 6, f"Expected 6 yearly risk scores, got {len(scores)}")
        for i, s in enumerate(scores):
            self.assertGreaterEqual(s, 0.0, f"Score at year {i+1} is negative: {s}")
            self.assertLessEqual(s, 1.0, f"Score at year {i+1} exceeds 1.0: {s}")
        self.assertEqual(
            scores,
            sorted(scores),
            f"Risk scores are not monotonically non-decreasing: {scores}",
        )


if __name__ == "__main__":
    unittest.main()
