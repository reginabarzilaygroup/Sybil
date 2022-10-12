[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pgmikhael/Sybil/blob/main/LICENSE.txt) <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="88.3" height="20"><linearGradient id="smooth" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient><clipPath id="round"><rect width="88.3" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#round)"><rect width="50.3" height="20" fill="#555"/><rect x="50.3" width="38.0" height="20" fill="darkgreen"/><rect width="88.3" height="20" fill="url(#smooth)"/></g><g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110"><text x="261.5" y="150" fill="#010101" fill-opacity=".3" transform="scale(0.1)" textLength="403.0" lengthAdjust="spacing">version</text><text x="261.5" y="140" transform="scale(0.1)" textLength="403.0" lengthAdjust="spacing">version</text><text x="683.0" y="150" fill="#010101" fill-opacity=".3" transform="scale(0.1)" textLength="280.0" lengthAdjust="spacing">0.2.0</text><text x="683.0" y="140" transform="scale(0.1)" textLength="280.0" lengthAdjust="spacing">0.2.0</text></g></svg> 

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

## Replicating results

You can replicate the results from our model using our training script:

```sh
python train.py
```

See our [documentation](docs/readme.md) for a full description of Sybil's training parameters.

## Cite

Coming soon.

