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
from sybil.utils import device_utils

script_directory = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(script_directory)

nlst_test_series_uids = """
1.2.840.113654.2.55.117165331353985769278030759027968557175
1.2.840.113654.2.55.125761431810488169605478704683628260210
1.2.840.113654.2.55.141145605876336438705007116410698504988
1.2.840.113654.2.55.172973285539665405130180217312651302726
1.2.840.113654.2.55.177114075868256371370044474147630945288
1.2.840.113654.2.55.210451208063625047828616019396666958685
1.2.840.113654.2.55.22343358537878328490619391877977879745
1.2.840.113654.2.55.250355771186119178528311921318050236359
1.2.840.113654.2.55.264036959200244122726184171100390477201
1.2.840.113654.2.55.270666838959776453521953970167166965589
1.2.840.113654.2.55.5405951206377419400128917954731813327
1.2.840.113654.2.55.83074092506605340087865221843273784687
1.2.840.113654.2.55.9114064256331314804445563449996729696
1.3.6.1.4.1.14519.5.2.1.7009.9004.102050757680671140089992182963
1.3.6.1.4.1.14519.5.2.1.7009.9004.140916852551836049221836980755
1.3.6.1.4.1.14519.5.2.1.7009.9004.145444099046834219014840219889
1.3.6.1.4.1.14519.5.2.1.7009.9004.160633847701259284025259919227
1.3.6.1.4.1.14519.5.2.1.7009.9004.219693265059595773200467950221
1.3.6.1.4.1.14519.5.2.1.7009.9004.228293333306602707645036607751
1.3.6.1.4.1.14519.5.2.1.7009.9004.230644512623268816899910856967
1.3.6.1.4.1.14519.5.2.1.7009.9004.234524223570882184991800514748
1.3.6.1.4.1.14519.5.2.1.7009.9004.252281466173937391895189766240
1.3.6.1.4.1.14519.5.2.1.7009.9004.310293448890324961317272491664
1.3.6.1.4.1.14519.5.2.1.7009.9004.330739122093904668699523188451
1.3.6.1.4.1.14519.5.2.1.7009.9004.338644625343131376124729421878
1.3.6.1.4.1.14519.5.2.1.7009.9004.646014655040104355409047679769
"""

test_series_uids = nlst_test_series_uids


def myprint(instr):
    print(f"{datetime.datetime.now()} - {instr}")


def download_file(url, filename):
    response = requests.get(url)

    target_dir = os.path.dirname(filename)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Check if the request was successful
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

    return filename


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
        demo_data_url = "https://www.dropbox.com/scl/fi/covbvo6f547kak4em3cjd/sybil_example.zip?rlkey=7a13nhlc9uwga9x7pmtk1cf1c&st=dqi0cf9k&dl=1"
        expected_scores = [
            0.021628819563619374,
            0.03857256315036462,
            0.07191945816622261,
            0.07926975188037134,
            0.09584583525781108,
            0.13568094038444453
        ]

        zip_file_name = "sybil_example.zip"
        cache_dir = os.path.expanduser("~/.sybil")
        demo_data_dir = os.path.join(cache_dir, "sybil_example")
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


def _get_nlst(series_instance_uid, cache_dir=".cache"):
    base_url = "https://nlst.cancerimagingarchive.net/nbia-api/services/v1"
    series_dir = os.path.join(cache_dir, series_instance_uid)
    if os.path.exists(series_dir):
        return series_dir

    action = "getImage"
    remote_url = f"{base_url}/{action}"
    print(f"Downloading {series_instance_uid} from {remote_url}")
    response = requests.get(remote_url, params={"SeriesInstanceUID": series_instance_uid})
    # The response is a zip file, I want to unzip it into a directory
    os.makedirs(series_dir, exist_ok=True)

    if response.status_code == 200:
        zip_file_bytes = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_file_bytes) as zip_file:
            zip_file.extractall(series_dir)
        print(f"Files extracted to {series_dir}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

    return series_dir


class TestPredictionRegression(unittest.TestCase):

    def test_nlst_predict(self, allow_resume=True, delete_downloaded_files=False):
        if not os.environ.get("SYBIL_TEST_RUN_REGRESSION", "false").lower() == "true":
            import pytest
            pytest.skip(f"Skipping long-running test in {type(self)}.")

        test_series_list = test_series_uids.split("\n")
        test_series_list = [x.strip() for x in test_series_list if x.strip()]
        print(f"About to test {len(test_series_list)} series")

        # Whether to allow resuming from a previous run,
        # or to overwrite the existing results file.
        # Operates on a per-series basis.
        model_name = "sybil_ensemble"

        # True ->  send web requests to the ARK server (must be launched separately).
        # False -> to run inference directly.
        use_ark = os.environ.get("SYBIL_TEST_USE_ARK", "false").lower() == "true"
        ark_host = os.environ.get("SYBIL_ARK_HOST", "http://localhost:5000")

        version = sybil.__version__

        out_fi_name = f"nlst_predictions_{model_name}_v{version}.json"
        info_data = {}
        if use_ark:
            # Query the ARK server to get the version
            print(f"Will use ark server {ark_host} for prediction")
            resp = requests.get(f"{ark_host}/info")
            info_data = resp.json()["data"]
            print(f"ARK server response: {resp.text}")
            version = info_data["modelVersion"]
            out_fi_name = f"nlst_predictions_ark_v{version}.json"

        output_dir = os.path.join(PROJECT_DIR, "tests", "nlst_predictions")

        metadata = {
                "modelName": model_name,
                "modelVersion": version,
                "start_time": datetime.datetime.now().isoformat(),
            }
        metadata.update(info_data)
        all_results = {"__metadata__":  metadata}

        os.makedirs(output_dir, exist_ok=True)
        cur_pred_results = os.path.join(output_dir, out_fi_name)
        cache_dir = os.path.join(PROJECT_DIR, ".cache")

        if os.path.exists(cur_pred_results):
            if allow_resume:
                with open(cur_pred_results, 'r') as f:
                    all_results = json.load(f)
            else:
                os.remove(cur_pred_results)

        if use_ark:
            model = device = None
        else:
            model = Sybil(model_name)

        device = device_utils.get_default_device()
        if bool(model) and bool(device):
            model.to(device)

        num_to_process = len(test_series_list)
        for idx, series_uid in enumerate(tqdm.tqdm(test_series_list)):
            print(f"{datetime.datetime.now()} Processing {series_uid} ({idx}/{num_to_process})")
            if series_uid in all_results:
                print(f"Already processed {series_uid}, skipping")
                continue

            series_dir = _get_nlst(series_uid, cache_dir=cache_dir)
            dicom_files = os.listdir(series_dir)
            dicom_files = sorted([os.path.join(series_dir, x) for x in dicom_files if x.endswith(".dcm")])

            if len(dicom_files) < 20:
                print(f"Skipping {series_uid} due to insufficient files ({len(dicom_files)})")
                continue

            try:
                prediction = all_results.get(series_uid, {})
                if use_ark:
                    # Submit prediction to ARK server.
                    files = [('dicom', open(file_path, 'rb')) for file_path in dicom_files]
                    r = requests.post(f"{ark_host}/dicom/files", files=files)
                    _ = [f[1].close() for f in files]
                    if r.status_code != 200:
                        print(f"An error occurred while processing {series_uid}: {r.text}")
                        prediction["error"] = r.text
                        continue
                    else:
                        r_json = r.json()
                        prediction = r_json["data"]
                        prediction["runtime"] = r_json["runtime"]
                        prediction["predictions"] = prediction["predictions"][0]
                else:
                    serie = Serie(dicom_files)
                    start_time = time.time()
                    pred_result = model.predict([serie], return_attentions=False)
                    runtime = "{:.2f}s".format(time.time() - start_time)

                    scores = pred_result.scores
                    prediction = {"predictions": scores, "runtime": runtime}

                if delete_downloaded_files:
                    shutil.rmtree(series_dir)

            except Exception as e:
                print(f"Failed to process {series_uid}: {e}")
                continue

            cur_dict = {
                "series_uid": series_uid,
                "num_slices": len(dicom_files),
            }

            if prediction:
                cur_dict.update(prediction)

            all_results[series_uid] = cur_dict

            # Save as we go
            with open(cur_pred_results, 'w') as f:
                json.dump(all_results, f, indent=2)

    def test_compare_predict_scores(self):
        if not os.environ.get("SYBIL_TEST_RUN_REGRESSION", "false").lower() == "true":
            import pytest
            pytest.skip(f"Skipping long-running test '{type(self)}'.")

        default_baseline_preds_path = os.path.join(PROJECT_DIR, "tests",
                                                   "nlst_predictions", "nlst_predictions_ark_v1.4.0.json")
        baseline_preds_path = os.environ.get("SYBIL_TEST_BASELINE_PATH", default_baseline_preds_path)

        version = sybil.__version__
        default_new_preds_path = os.path.join(PROJECT_DIR, "tests",
                                                "nlst_predictions", f"nlst_predictions_sybil_ensemble_v{version}.json")
        new_preds_path = os.environ.get("SYBIL_TEST_COMPARE_PATH", default_new_preds_path)
        assert new_preds_path, "SYBIL_TEST_COMPARE_PATH must be set to the path of the new predictions file."
        pred_key = "predictions"
        num_compared = 0

        with open(baseline_preds_path, 'r') as f:
            baseline_preds = json.load(f)
        with open(new_preds_path, 'r') as f:
            new_preds = json.load(f)

        ignore_keys = {"__metadata__"}
        overlap_keys = set(baseline_preds.keys()).intersection(new_preds.keys()) - ignore_keys
        union_keys = set(baseline_preds.keys()).union(new_preds.keys()) - ignore_keys
        print(f"{len(overlap_keys)} / {len(union_keys)} patients in common between the two prediction files.")

        all_mismatches = []
        for series_uid_key in overlap_keys:
            if series_uid_key in ignore_keys:
                continue

            if pred_key not in baseline_preds[series_uid_key]:
                print(f"{pred_key} not found in baseline predictions for {series_uid_key}")
                assert pred_key not in new_preds[series_uid_key]
                continue

            compare_keys = ["predictions"]
            for comp_key in compare_keys:
                cur_baseline_preds = baseline_preds[series_uid_key][comp_key][0]
                cur_new_preds = new_preds[series_uid_key][comp_key][0]
                for ind in range(len(cur_baseline_preds)):
                    year = ind + 1
                    baseline_score = cur_baseline_preds[ind]
                    new_score = cur_new_preds[ind]
                    does_match = math.isclose(baseline_score, new_score, abs_tol=1e-6)
                    if not does_match:
                        err_str = f"Scores for {series_uid_key}, {comp_key} differ for year {year}.\n"
                        err_str += f"Diff: {new_score - baseline_score:0.4e}. Baseline: {baseline_score:0.4e}, New: {new_score:0.4e}"
                        all_mismatches.append(err_str)

            num_compared += 1

        assert num_compared > 0
        print(f"Compared {num_compared} patients.")

        if all_mismatches:
            print(f"Found {len(all_mismatches)} mismatches.")
            for err in all_mismatches:
                print(err)

        num_mismatches = len(all_mismatches)
        assert num_mismatches == 0, f"Found {num_mismatches} mismatches between the two prediction files."

    def test_calibrator(self):
        """
        Test the calibrator against previous known calibrations.
        """

        default_baseline_path = os.path.join(PROJECT_DIR, "tests", "sybil_ensemble_v1.4.0_calibrations.json")
        baseline_path = os.environ.get("SYBIL_TEST_BASELINE_PATH", default_baseline_path)

        if not os.path.exists(baseline_path) and baseline_path == default_baseline_path:
            os.makedirs(os.path.dirname(default_baseline_path), exist_ok=True)
            reference_calibrations_url = "https://www.dropbox.com/scl/fi/2fx6ukmozia7y3u8mie97/sybil_ensemble_v1.4.0_calibrations.json?rlkey=tquids9qo4mkkuf315nqdq0o7&dl=1"
            download_file(reference_calibrations_url, default_baseline_path)

        default_cal_dict_path = os.path.expanduser("~/.sybil/sybil_ensemble_simple_calibrator.json")
        compare_calibrator_path = os.environ.get("SYBIL_TEST_COMPARE_PATH", default_cal_dict_path)
        compare_calibrator_path = os.path.expanduser(compare_calibrator_path)
        if not os.path.exists(compare_calibrator_path) and compare_calibrator_path == default_cal_dict_path:
            test_model = Sybil("sybil_ensemble")

        raw_calibrator_dict = json.load(open(compare_calibrator_path, "r"))
        new_calibrator_dict = {}
        for key, val in raw_calibrator_dict.items():
            new_calibrator_dict[key] = sybil.models.calibrator.SimpleClassifierGroup.from_json(val)

        baseline_preds = json.load(open(baseline_path, "r"))
        test_probs = np.array(baseline_preds["x"]).reshape(-1, 1)
        year_keys = [key for key in baseline_preds.keys() if key.startswith("Year")]
        for year_key in year_keys:
            baseline_scores = np.array(baseline_preds[year_key]).flatten()
            new_cal = new_calibrator_dict[year_key]
            new_scores = new_cal.predict_proba(test_probs).flatten()

            self.assertTrue(np.allclose(baseline_scores, new_scores, atol=1e-10), f"Calibration mismatch for {year_key}")


if __name__ == "__main__":
    unittest.main()
