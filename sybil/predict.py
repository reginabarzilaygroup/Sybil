#!/usr/bin/env python

__doc__ = """
Use Sybil or Sybil2 to run inference.

Modes
-----
  single   Score one exam (Sybil v1, single DICOM directory).
  batch    Score a cohort from a CSV manifest (Sybil2, supports multi-GPU).
"""

import argparse
import json
import os
import pickle
import typing
from typing import Literal

import torch
from loguru import logger

import sybil.datasets.utils
import sybil.utils.logging_utils
from sybil import Serie, Sybil, Sybil2, SybilV2Dataset, visualize_attentions, __version__


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name",
        default="sybil_ensemble",
        dest="model_name",
        help="Pretrained model alias or path. Default: sybil_ensemble",
    )
    parser.add_argument(
        "-l", "--log", "--loglevel", "--log-level",
        default="INFO",
        dest="loglevel",
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)


def _get_parser() -> argparse.ArgumentParser:
    description = __doc__ + f"\nVersion: {__version__}\n"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ------------------------------------------------------------------
    # single – v1 single-exam inference (backwards-compatible)
    # ------------------------------------------------------------------
    single = subparsers.add_parser(
        "single",
        help="Score a single exam (Sybil v1).",
    )
    single.add_argument(
        "image_dir",
        help="Directory containing DICOM or PNG files for one exam.",
    )
    single.add_argument(
        "--output-dir",
        default="sybil_result",
        dest="output_dir",
        help="Directory in which to save prediction_scores.json. Default: sybil_result",
    )
    single.add_argument(
        "--return-attentions",
        default=False,
        action="store_true",
        dest="return_attentions",
        help="Save attention scores to attention_scores.pkl.",
    )
    single.add_argument(
        "--write-attention-images",
        default=False,
        action="store_true",
        dest="write_attention_images",
        help="Overlay attention on slices and write GIFs. Implies --return-attentions.",
    )
    single.add_argument(
        "--file-type",
        default="auto",
        dest="file_type",
        choices={"dicom", "png", "auto"},
        help="File type of input images. Default: auto-detect.",
    )
    single.add_argument(
        "--threads",
        type=int,
        default=0,
        help="PyTorch CPU threads. 0 = all cores, negative = PyTorch default.",
    )
    _add_common_args(single)

    # ------------------------------------------------------------------
    # batch – Sybil2 cohort inference from CSV
    # ------------------------------------------------------------------
    batch = subparsers.add_parser(
        "batch",
        help="Score a cohort from a CSV manifest (Sybil2).",
    )
    batch.add_argument(
        "csv",
        help=(
            "CSV manifest with columns: patient_id, timepoint, ct_dir "
            "[, label, censor_time]."
        ),
    )
    batch.add_argument(
        "--output",
        required=True,
        dest="output_path",
        help="Output file path (.csv or .json).",
    )
    batch.add_argument(
        "--cache-dir",
        required=True,
        dest="cache_dir",
        help="Directory for intermediate NIfTI files.",
    )
    batch.add_argument(
        "--batch-size",
        type=int,
        default=4,
        dest="batch_size",
        help="Patients per preprocessing batch. Default: 4",
    )
    batch.add_argument(
        "--num-workers",
        type=int,
        default=4,
        dest="num_workers",
        help="DataLoader worker processes. Default: 4",
    )
    batch.add_argument(
        "--file-type",
        default="dicom",
        dest="file_type",
        choices={"dicom", "png"},
        help="CT file type. Default: dicom",
    )
    batch.add_argument(
        "--distributed",
        default=False,
        action="store_true",
        help=(
            "Enable multi-GPU distributed inference. "
            "Requires torch.distributed to be initialised (use torchrun to launch)."
        ),
    )
    _add_common_args(batch)

    return parser


# ---------------------------------------------------------------------------
# Single-exam inference (Sybil v1)
# ---------------------------------------------------------------------------

def predict_single(
    image_dir: str,
    output_dir: str,
    model_name: str = "sybil_ensemble",
    return_attentions: bool = False,
    write_attention_images: bool = False,
    file_type: Literal["auto", "dicom", "png"] = "auto",
    threads: int = 0,
):
    """Score one exam with the Sybil v1 model."""
    return_attentions |= write_attention_images

    input_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if not f.startswith(".")
    ]
    input_files = [f for f in input_files if os.path.isfile(f)]

    voxel_spacing = None
    if file_type == "auto":
        extensions = {os.path.splitext(f)[1] for f in input_files}
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
    logger.debug(f"Scoring {len(input_files)} {file_type} file(s) from {image_dir}")

    model = Sybil(model_name)
    serie = Serie(input_files, voxel_spacing=voxel_spacing, file_type=file_type)
    prediction = model.predict([serie], return_attentions=return_attentions, threads=threads)

    pred_dict = {"predictions": prediction.scores}
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "prediction_scores.json")
    with open(pred_path, "w") as f:
        json.dump(pred_dict, f, indent=2)
    logger.debug(f"Scores written to {pred_path}")

    series_with_attention = None
    if return_attentions:
        attn_path = os.path.join(output_dir, "attention_scores.pkl")
        with open(attn_path, "wb") as f:
            pickle.dump(prediction, f)

    if write_attention_images:
        series_with_attention = visualize_attentions(
            [serie],
            attentions=prediction.attentions,
            save_directory=output_dir,
            gain=3,
        )

    return pred_dict, series_with_attention


# ---------------------------------------------------------------------------
# Batch / distributed inference (Sybil2)
# ---------------------------------------------------------------------------

def predict_batch(
    csv: str,
    output_path: str,
    cache_dir: str,
    model_name: str = "sybil2",
    batch_size: int = 4,
    num_workers: int = 4,
    file_type: str = "dicom",
    distributed: bool = False,
):
    """Score a cohort from a CSV manifest using Sybil2."""
    if distributed:
        torch.distributed.init_process_group(backend="nccl")

    model = Sybil2(model_name)
    results = model.predict_dataset(
        dataset=csv,
        output_path=output_path,
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        file_type=file_type,
        distributed=distributed,
    )

    if distributed:
        torch.distributed.destroy_process_group()

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = _get_parser()
    args = parser.parse_args()
    sybil.utils.logging_utils.configure_logger(args.loglevel)

    if args.mode == "single":
        os.makedirs(args.output_dir, exist_ok=True)
        pred_dict, _ = predict_single(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            return_attentions=args.return_attentions,
            write_attention_images=args.write_attention_images,
            file_type=args.file_type,
            threads=args.threads,
        )
        print(json.dumps(pred_dict, indent=2))

    elif args.mode == "batch":
        results = predict_batch(
            csv=args.csv,
            output_path=args.output_path,
            cache_dir=args.cache_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            file_type=args.file_type,
            distributed=args.distributed,
        )
        # On rank 0 (or single-GPU), print a summary
        if not args.distributed or torch.distributed.get_rank() == 0:
            print(f"Scored {len(results)} patient(s). Results written to {args.output_path}")


if __name__ == "__main__":
    main()
