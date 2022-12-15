[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pgmikhael/Sybil/blob/main/LICENSE.txt) ![version](https://img.shields.io/badge/version-1.0.1-success)

# Sybil

Lung Cancer Risk Prediction

## Run the model

You can load our pretrained model trained on the NLST dataset, and score a given DICOM serie as follows:

```python
from sybil import Serie, Sybil

# Load a trained model
model = Sybil("sybil_base")

# Get risk scores
serie = Serie([dicom_path_1, dicom_path_2, ...])
scores = model.predict([serie])

# You can also evaluate by providing labels
serie = Serie([dicom_path_1, dicom_path_2, ...], label=1)
results = model.evaluate([serie])
```

Models available include: `sybil_base` and `sybil_ensemble`.

All model files are available [here](https://drive.google.com/drive/folders/1nBp05VV9mf5CfEO6W5RY4ZpcpxmPDEeR?usp=sharing).

## Replicating results

You can replicate the results from our model using our training script:

```sh
python train.py
```

See our [documentation](docs/readme.md) for a full description of Sybil's training parameters.

## LDCT Orientation

The model expects the input to be an Axial LDCT, where the first frame is of the abdominal region and the last frame is along the clavicles.

## Cite

Coming soon.

