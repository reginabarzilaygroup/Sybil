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


## Annotations

To help train the model, two fellowship-trained thoracic radiologists jointly annotated suspicious lesions on NLST LDCTs using [MD.AI](https://md.ai) software for all participants who developed cancer within 1 year after an LDCT. Each lesion’s volume was marked with bounding boxes on contiguous thin-cut axial images. The “ground truth” annotations were informed by the imaging appearance and the clinical data provided by the NLST, i.e., the series and image number of cancerous nodules and the anatomical location of biopsy-confirmed lung cancers. For these participants, lesions in the location of subsequently diagnosed cancers were also annotated, even if the precursor lesion lacked imaging features specific for cancer. 

Annotations are availble to download in JSON format [here](https://drive.google.com/file/d/19aa5yIHPWu3NtjqvXDc8NYB2Ub9V-4WM/view?usp=share_link). The JSON file is structured as below, where `(x,y)` refers to the top left corner of the bounding box, and all values are normlized to the image size (512,512). 

```
{
  series1_id: {   # Series Instance UID
    image1_id: [  # SOP Instance UID / file name
      {"x": x_axis_value, "y": y_axis_value, "height": bounding_box_heigh, "width": bounding_box_width}, # bounding box 1
      {"x": x_axis_value, "y": y_axis_value, "height": bounding_box_heigh, "width": bounding_box_width}, # bounding box 2
      ...
      ],
    image2_id: [],
    ...
  }
  series2_id: {},
  ...
}
```

## Cite

Coming soon.

