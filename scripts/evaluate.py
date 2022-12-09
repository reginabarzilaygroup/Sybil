import os, sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
import csv
import argparse
from sybil import Sybil, Serie
from rich import print


def parse_csv_dataset(file_path):
    """
    Convert a CSV file into a list of Serie objects from the following data:
        - 'patient_id': str
        - 'split': str
        - 'exam_id': str
        - 'series_id': str
        - 'ever_has_future_cancer': bool
        - 'years_to_cancer': int
        - 'years_to_last_negative_followup': int
        - 'paths': [str]
        - 'file_path': str
        - 'file_type': str
        - 'pixel_spacing': [float]
        - 'slice_thickness': float
        - 'file_or_directory': str


    Parameters
    ----------
    file_path : str
        path to csv file

    Returns
    -------
    list
        list of sybil.Serie objects
    """
    dataset_dicts = {}
    _reader = csv.DictReader(open(file_path, "r"))
    for _row in _reader:
        row = {
            k.encode("ascii", "ignore").decode(): v.encode("ascii", "ignore").decode()
            for k, v in _row.items()
        }
        patient_id, exam_id, series_id = (
            row["patient_id"],
            row["exam_id"],
            row["series_id"],
        )
        unique_id = "{}_{}_{}".format(patient_id, exam_id, series_id)

        if not (unique_id in dataset_dicts):
            dataset_dicts[unique_id] = {
                "unique_id": unique_id,
                "patient_id": patient_id,
                "exam_id": exam_id,
                "series_id": series_id,
                "ever_has_future_cancer": row["ever_has_future_cancer"],
                "years_to_cancer": row["years_to_cancer"],
                "years_to_last_negative_followup": row[
                    "years_to_last_negative_followup"
                ],
                "file_type": row["file_type"],
                "pixel_spacing": row["pixel_spacing"],
                "slice_thickness": row["slice_thickness"],
            }
            if row["file_or_directory"] == "directory":
                assert (
                    row["file_type"] == "dicom"
                ), "Only DICOM files when a directory is provided"
                dicom_files = os.listdir(row["file_path"])
                dataset_dicts[unique_id]["paths"] = [
                    os.path.join(row["file_path"], f) for f in dicom_files
                ]

            elif row["file_or_directory"] == "file":
                dataset_dicts[unique_id]["paths"] = [row["file_path"]]

            else:
                raise ValueError("file_or_directory must be 'file' or 'directory'")

        elif (unique_id in dataset_dicts) and (row["file_or_directory"] == "file"):
            dataset_dicts[unique_id]["paths"].append(row["file_path"])

    dataset = []
    for unique_id, series_dict in dataset_dicts.items():
        censor_time = (
            row["years_to_cancer"]
            if row["ever_has_future_cancer"]
            else row["years_to_last_negative_followup"]
        )
        voxel = series_dict["pixel_spacing"] + [series_dict["slice_thickness"]]
        dataset.append(
            Serie(
                dicoms=series_dict["paths"],
                voxel_spacing=voxel,
                label=row["ever_has_future_cancer"],
                censor_time=censor_time,
                file_type=row["file_type"],
            )
        )

    return dataset


def print_performance(results):
    """Print performance metrics

    Args:
        results (NamedTuple): Evaluation named tuple from Sybil.evaluate()
    """
    performance = "Performance:\n{}\n".format("-" * 12)

    for i, auc in enumerate(results.auc):
        performance += f"* {i+1}-Year ROC-AUC: {auc}\n"

    performance += f"* C-Index: {results.c_index}"

    print(performance)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv_dataset_path",
    type=str,
    default="../files/lung_cancer_dataset.csv",
    help="Path to csv dataset",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # make dataset from csv
    series = parse_csv_dataset(args.csv_dataset_path)

    # Load a trained model
    model = Sybil("sybil_ensemble")

    # You can also evaluate by providing labels
    results = model.evaluate(series)

    print_performance(results)
