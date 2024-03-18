import datetime
import math
import os
import requests
import zipfile

from sybil import Serie, Sybil, visualize_attentions

script_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.dirname(script_directory)


def myprint(instr):
    print(f"{datetime.datetime.now()} - {instr}")


def download_and_extract_zip(zip_file_name, cache_dir, url, demo_data_dir):
    # Check and construct the full path of the zip file
    zip_file_path = os.path.join(cache_dir, zip_file_name)

    # 1. Check if the zip file exists
    if not os.path.exists(zip_file_path):
        # myprint(f"Zip file not found at {zip_file_path}. Downloading from {url}...")
        # 2. Download the file
        response = requests.get(url)
        with open(zip_file_path, 'wb') as file:
            file.write(response.content)
        # myprint(f"Downloaded zip file to {zip_file_path}")

    # 3. Check if the output directory exists
    if not os.path.exists(demo_data_dir):
        # myprint(f"Output directory {demo_data_dir} does not exist. Creating and extracting...")
        # 4. Extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(demo_data_dir)
        # myprint(f"Extracted zip file to {demo_data_dir}")
    else:
        pass
        # myprint(f"Output directory {demo_data_dir} already exists. No extraction needed.")


def main():
    # Note that this function is named so that pytest will not automatically discover it
    # It takes a long time to run and potentially a lot of disk space

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
    # model = Sybil("sybil_ensemble")
    model = Sybil()

    myprint(f"Beginning prediction using {num_files} files from {image_data_dir}")

    # Get risk scores
    serie = Serie(dicom_files)
    series = [serie]
    prediction = model.predict(series, return_attentions=True)
    actual_scores = prediction.scores[0]
    count = len(actual_scores)

    myprint(f"Prediction finished. Results\n{actual_scores}")

    # pprint.pprint(f"Prediction object: {prediction}")

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


if __name__ == "__main__":
    main()
