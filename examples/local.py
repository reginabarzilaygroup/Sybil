#!/usr/bin/env python

__doc__ = """
Simple example script showing how to use the Sybil library locally to predict risk scores for a set of DICOM files.
"""

import sybil
from sybil import visualize_attentions

from utils import get_demo_data


def main():
    # Load a trained model
    model = sybil.Sybil("sybil_ensemble")

    dicom_files = get_demo_data()

    # Get risk scores
    serie = sybil.Serie(dicom_files)
    print(f"Processing {len(dicom_files)} DICOM files")
    prediction = model.predict([serie], return_attentions=True)
    scores = prediction.scores

    print(f"Risk scores: {scores}")

    # Visualize attention maps
    output_dir = "sybil_attention_output"

    print(f"Writing attention images to {output_dir}")
    series_with_attention = visualize_attentions(
        serie,
        attentions=prediction.attentions,
        save_directory=output_dir,
        gain=3,
    )

    print(f"Finished writing attention images to {output_dir}")

if __name__ == "__main__":
    main()
