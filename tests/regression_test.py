import datetime
import io
import json
import math
import os
import shutil
import time
import unittest

import numpy as np
import requests
import tqdm
import warnings
import zipfile

warnings.filterwarnings("ignore", category=DeprecationWarning)

import sybil.model
import sybil.models.calibrator
from sybil import Serie, Sybil, visualize_attentions

script_directory = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(script_directory)


def myprint(instr):
    print(f"{datetime.datetime.now()} - {instr}")


def download_and_extract_zip(zip_file_name, cache_dir, url, demo_data_dir):
    # Check and construct the full path of the zip file
    zip_file_path = os.path.join(cache_dir, zip_file_name)

    # 1. Check if the zip file exists
    if not os.path.exists(zip_file_path):
        # 2. Download the file
        response = requests.get(url)
        with open(zip_file_path, 'wb') as file:
            file.write(response.content)

    # 3. Check if the output directory exists
    if not os.path.exists(demo_data_dir):
        # 4. Extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(demo_data_dir)
    else:
        pass
        # myprint(f"Output directory {demo_data_dir} already exists. No extraction needed.")


def get_sybil_model_path(model_name_or_path, cache_dir="~/.sybil"):
    cache_dir = os.path.expanduser(cache_dir)
    if os.path.exists(model_name_or_path):
        path = model_name_or_path
    elif model_name_or_path in sybil.model.NAME_TO_FILE:
        paths = sybil.model.NAME_TO_FILE[model_name_or_path]["checkpoint"]
        assert len(paths) == 1, "Can only save 1 model at a time, no ensembles"
        path = os.path.join(cache_dir, paths[0] + ".ckpt")
    else:
        raise ValueError(f"Model name or path not found: {model_name_or_path}")

    return path


class TestPredict(unittest.TestCase):
    def test_demo_data(self):
        if not os.environ.get("SYBIL_TEST_RUN_REGRESSION", "false").lower() == "true":
            import pytest
            pytest.skip(f"Skipping long-running test in {type(self)}.")

        # Download demo data
        demo_data_url = "https://www.dropbox.com/sh/addq480zyguxbbg/AACJRVsKDL0gpq-G9o3rfCBQa?dl=1"
        expected_scores = [
            0.021628819563619374,
            0.03857256315036462,
            0.07191945816622261,
            0.07926975188037134,
            0.09584583525781108,
            0.13568094038444453
        ]

        zip_file_name = "SYBIL.zip"
        cache_dir = os.path.expanduser("~/.sybil")
        demo_data_dir = os.path.join(cache_dir, "SYBIL")
        image_data_dir = os.path.join(demo_data_dir, "sybil_demo_data")
        os.makedirs(cache_dir, exist_ok=True)
        download_and_extract_zip(zip_file_name, cache_dir, demo_data_url, demo_data_dir)

        dicom_files = os.listdir(image_data_dir)
        dicom_files = [os.path.join(image_data_dir, x) for x in dicom_files]
        num_files = len(dicom_files)

        # Load a trained model
        model = Sybil()

        myprint(f"Beginning prediction using {num_files} files from {image_data_dir}")

        # Get risk scores
        serie = Serie(dicom_files)
        series = [serie]
        prediction = model.predict(series, return_attentions=True)
        actual_scores = prediction.scores[0]
        count = len(actual_scores)

        myprint(f"Prediction finished. Results\n{actual_scores}")

        assert len(expected_scores) == len(actual_scores), f"Unexpected score length {count}"

        all_elements_match = True
        for exp_score, act_score in zip(expected_scores, actual_scores):
            does_match = math.isclose(exp_score, act_score, rel_tol=1e-6)
            assert does_match, f"Mismatched scores. {exp_score} != {act_score}"
            all_elements_match &= does_match

        print(f"Data URL: {demo_data_url}\nAll {count} elements match: {all_elements_match}")

        series_with_attention = visualize_attentions(
            series,
            attentions=prediction.attentions,
            save_directory="regression_test_output",
            gain=3,
        )

    def test_calibrator(self):
        """
        Test the calibrator against previous known calibrations.
        """

        default_baseline_path = os.path.join(PROJECT_DIR, "tests", "sybil_ensemble_v1.4.0_calibrations.json")
        baseline_path = os.environ.get("SYBIL_TEST_BASELINE_PATH", default_baseline_path)

        new_cal_dict_path = os.environ.get("SYBIL_TEST_COMPARE_PATH", "~/.sybil/sybil_ensemble_simple_calibrator.json")
        new_cal_dict_path = os.path.expanduser(new_cal_dict_path)
        raw_new_cal_dict = json.load(open(new_cal_dict_path, "r"))
        new_cal_dict = {}
        for key, val in raw_new_cal_dict.items():
            new_cal_dict[key] = sybil.models.calibrator.SimpleClassifierGroup.from_json(val)

        baseline_preds = json.load(open(baseline_path, "r"))
        test_probs = np.array(baseline_preds["x"]).reshape(-1, 1)
        year_keys = [key for key in baseline_preds.keys() if key.startswith("Year")]
        for year_key in year_keys:
            baseline_scores = np.array(baseline_preds[year_key]).flatten()
            new_cal = new_cal_dict[year_key]
            new_scores = new_cal.predict_proba(test_probs).flatten()

            self.assertTrue(np.allclose(baseline_scores, new_scores, atol=1e-10), f"Calibration mismatch for {year_key}")


if __name__ == "__main__":
    unittest.main()
