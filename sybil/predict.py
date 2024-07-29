#!/usr/bin/env python

__doc__ = """
Use Sybil to run inference on a single exam.
"""

import argparse
import json
import os
import pickle
import typing
from typing import Literal

import sybil.utils.logging_utils
import sybil.datasets.utils
from sybil import Serie, Sybil, visualize_attentions, __version__


def _get_parser():
    description = __doc__ + f"\nVersion: {__version__}\n"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "image_dir",
        default=None,
        help="Path to directory containing DICOM/PNG files (from a single exam) to run inference on. "
             "Every file in the directory will be included.",
    )

    parser.add_argument(
        "--output-dir",
        default="sybil_result",
        dest="output_dir",
        help="Output directory in which to save prediction results. "
        "Prediction will be printed to stdout as well.",
    )

    parser.add_argument(
        "--return-attentions",
        default=False,
        action="store_true",
        help="Return hidden vectors and attention scores, write them to a pickle file.",
    )

    parser.add_argument(
        "--write-attention-images",
        default=False,
        action="store_true",
        help="Generate images with attention overlap. Sets --return-attentions (if not already set).",
    )

    parser.add_argument(
        "--file-type",
        default="auto",
        dest="file_type",
        choices={"dicom", "png", "auto"},
        help="File type of input images. "
        "If not provided, the file type will be inferred from input extensions.",
    )

    parser.add_argument(
        "--model-name",
        default="sybil_ensemble",
        dest="model_name",
        help="Name of the model to use for prediction. Default: sybil_ensemble",
    )

    parser.add_argument("-l", "--log", "--loglevel", "--log-level",
                        default="INFO", dest="loglevel")

    parser.add_argument('--threads', type=int, default=0,
                        help="Number of threads to use for PyTorch inference. "
                             "Default is 0 (use all available cores)."
                             "Set to a negative number to use Pytorch default.")

    parser.add_argument("-v", "--version", action="version", version=__version__)

    return parser


def predict(
    image_dir,
    output_dir,
    model_name="sybil_ensemble",
    return_attentions=False,
    write_attention_images=False,
    file_type: Literal["auto", "dicom", "png"] = "auto",
    threads: int = 0,
):
    logger = sybil.utils.logging_utils.get_logger()

    return_attentions |= write_attention_images

    input_files = os.listdir(image_dir)
    input_files = [os.path.join(image_dir, x) for x in input_files if not x.startswith(".")]
    input_files = [x for x in input_files if os.path.isfile(x)]

    voxel_spacing = None
    if file_type == "auto":
        extensions = {os.path.splitext(x)[1] for x in input_files}
        extension = extensions.pop()
        if len(extensions) > 1:
            raise ValueError(
                f"Multiple file types found in {image_dir}: {','.join(extensions)}"
            )

        file_type = "dicom"
        if extension.lower() in {".png", "png"}:
            file_type = "png"
            voxel_spacing = sybil.datasets.utils.VOXEL_SPACING
            logger.debug(f"Using default voxel spacing: {voxel_spacing}")
    assert file_type in {"dicom", "png"}
    file_type = typing.cast(Literal["dicom", "png"], file_type)

    num_files = len(input_files)

    logger.debug(f"Beginning prediction using {num_files} {file_type} files from {image_dir}")

    # Load a trained model
    model = Sybil(model_name)

    # Get risk scores
    serie = Serie(input_files, voxel_spacing=voxel_spacing, file_type=file_type)
    series = [serie]
    prediction = model.predict(series, return_attentions=return_attentions, threads=threads)
    prediction_scores = prediction.scores[0]

    logger.debug(f"Prediction finished. Results:\n{prediction_scores}")

    prediction_path = os.path.join(output_dir, "prediction_scores.json")
    pred_dict = {"predictions": prediction.scores}
    with open(prediction_path, "w") as f:
        json.dump(pred_dict, f, indent=2)

    series_with_attention = None
    if return_attentions:
        attention_path = os.path.join(output_dir, "attention_scores.pkl")
        with open(attention_path, "wb") as f:
            pickle.dump(prediction, f)

    if write_attention_images:
        series_with_attention = visualize_attentions(
            series,
            attentions=prediction.attentions,
            save_directory=output_dir,
            gain=3,
        )

    return pred_dict, series_with_attention


def main():
    args = _get_parser().parse_args()
    sybil.utils.logging_utils.configure_logger(args.loglevel)

    os.makedirs(args.output_dir, exist_ok=True)

    pred_dict, series_with_attention = predict(
        args.image_dir,
        args.output_dir,
        args.model_name,
        args.return_attentions,
        args.write_attention_images,
        file_type=args.file_type,
        threads=args.threads,
    )

    print(json.dumps(pred_dict, indent=2))


if __name__ == "__main__":
    main()
