import argparse
import datetime
import json
import logging
import os
import pickle

from sybil import Serie, Sybil, visualize_attentions

script_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.dirname(script_directory)


def _get_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--output-dir', default="sybil_result", dest="output_dir",
                        help="Output directory in which to save prediction results."
                             "Prediction will be printed to stdout as well.")

    parser.add_argument('--return-attentions', default=False, action="store_true",
                        help="Generate an image which overlaps attention scores.")

    parser.add_argument('dicom_dir', default=None,
                        help="Path to directory containing DICOM files (from a single exam) to run inference on."
                             "Every dcm file in the directory will be included.")

    parser.add_argument('--model-name', default="sybil_ensemble", dest="model_name",
                        help="Name of the model to use for prediction. Default: sybil_ensemble")

    parser.add_argument('-l', '--log', '--loglevel', default="INFO", dest="loglevel")

    return parser


def logging_basic_config(args):
    info_fmt = "[%(asctime)s] - %(message)s"
    debug_fmt = "[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s"
    fmt = debug_fmt if args.loglevel.upper() == "DEBUG" else info_fmt

    logging.basicConfig(format=fmt,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=args.loglevel.upper())


def inference(dicom_dir, output_dir, model_name="sybil_ensemble", return_attentions=False):
    logger = logging.getLogger('inference')

    dicom_files = os.listdir(dicom_dir)
    dicom_files = [os.path.join(dicom_dir, x) for x in dicom_files]
    dicom_files = [x for x in dicom_files if x.endswith(".dcm") and os.path.isfile(x)]
    num_files = len(dicom_files)

    # Load a trained model
    model = Sybil(model_name)

    logger.debug(f"Beginning prediction using {num_files} files from {dicom_dir}")

    # Get risk scores
    serie = Serie(dicom_files)
    series = [serie]
    prediction = model.predict(series, return_attentions=return_attentions)
    prediction_scores = prediction.scores[0]

    logger.debug(f"Prediction finished. Results:\n{prediction_scores}")

    prediction_path = os.path.join(output_dir, "prediction_scores.json")
    pred_dict = {"predictions": prediction.scores}
    with open(prediction_path, "w") as f:
        json.dump(pred_dict, f, indent=2)

    if return_attentions:
        attention_path = os.path.join(output_dir, "attention_scores.pkl")
        with open(attention_path, "wb") as f:
            pickle.dump(prediction, f)

        series_with_attention = visualize_attentions(
            series,
            attentions=prediction.attentions,
            save_directory=output_dir,
            gain=3,
        )

    return pred_dict


def main():
    args = _get_parser().parse_args()
    logging_basic_config(args)

    os.makedirs(args.output_dir, exist_ok=True)

    pred_dict = inference(args.dicom_dir, args.output_dir, args.model_name, args.return_attentions)

    print(json.dumps(pred_dict, indent=2))


if __name__ == "__main__":
    main()
