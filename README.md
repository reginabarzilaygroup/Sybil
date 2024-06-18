[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pgmikhael/Sybil/blob/main/LICENSE.txt) ![version](https://img.shields.io/badge/version-1.2.0-success)

# Sybil

Lung Cancer Risk Prediction.

Additional documentation can be found on the [GitHub Wiki](https://github.com/reginabarzilaygroup/Sybil/wiki).

# Run a regression test

```shell
python tests/regression_test.py
```

This will download the`sybil_ensemble` model and sample data, and compare the results to what has previously been calculated.


# Run the model

You can load our pretrained model trained on the NLST dataset, and score a given DICOM serie as follows:

```python
from sybil import Serie, Sybil

# Load a trained model
model = Sybil("sybil_ensemble")

# Get risk scores
serie = Serie([dicom_path_1, dicom_path_2, ...])
scores = model.predict([serie])

# You can also evaluate by providing labels
serie = Serie([dicom_path_1, dicom_path_2, ...], label=1)
results = model.evaluate([serie])
```

Models available include: `sybil_1`, `sybil_2`, `sybil_3`, `sybil_4`, `sybil_5` and `sybil_ensemble`.

All model files are available on [GitHub releases](https://github.com/reginabarzilaygroup/Sybil/releases) as well as [here](https://drive.google.com/drive/folders/1nBp05VV9mf5CfEO6W5RY4ZpcpxmPDEeR?usp=sharing).

# Replicating results

You can replicate the results from our model using our training script:

```sh
python train.py
```

See our [documentation](docs/readme.md) for a full description of Sybil's training parameters. Additional information on the training process can be found on the [train](https://github.com/reginabarzilaygroup/Sybil/tree/train) branch of this repository.


# LDCT Orientation

The model expects the input to be an Axial LDCT, where the first frame is of the abdominal region and the last frame is along the clavicles.

When the input is of the `dicom` type, the frames will be automatically sorted. However, for `png` inputs, the path of the PNG files must be in the right anatomical order. 


# Annotations

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

# Attention Scores

The multi-attention pooling layer aims to learn the importance of each slice in the 3D volume and the importance of each pixel in the 2D slice. During training, these are supervised by bounding boxes of the cancerous nodules. This is a soft attention mechanism, and the model's primary task is to predict the risk of lung cancer. However, the attention scores can be extracted and used to visualize the model's focus on the 3D volume and the 2D slices. 

To extract the attention scores, you can use the  `return_attentions` argument as follows:

```python

results = model.predict([serie], return_attentions=True)

attentions = results.attentions

```

The `attentions` will be a list of length equal to the number of series. Each series has a dictionary with the following keys:

- `image_attention_1`: attention scores (as logits) over the pixels in the 2D slice. This will be a list of length equal to the size of the model ensemble.
- `volume_attention_1`: attention scores (as logits) over each slice in the 3D volume. This will be a list of length equal to the size of the model ensemble.

To visualize the attention scores, you can use the following code. This will return a list of 2D images, where the attention scores are overlaid on the original images. If you provide a `save_directory`, the images will be saved as a GIF. If multiple series are provided, the function will return a list of lists, one for each series.

```python

from sybil import visualize_attentions

series_with_attention = visualize_attentions(
    series,
    attentions = attentions,
    save_directory = "path_to_save_directory",
    gain = 3, 
)

```

# Training Data

The Sybil model was trained using the National Lung Screening Trial (NLST) dataset:

National Lung Screening Trial Research Team. (2013). Data from the National Lung Screening Trial (NLST) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.HMQ8-J677

# Cite

```
@article{mikhael2023sybil,
  title={Sybil: a validated deep learning model to predict future lung cancer risk from a single low-dose chest computed tomography},
  author={Mikhael, Peter G and Wohlwend, Jeremy and Yala, Adam and Karstens, Ludvig and Xiang, Justin and Takigami, Angelo K and Bourgouin, Patrick P and Chan, PuiYee and Mrah, Sofiane and Amayri, Wael and Juan, Yu-Hsiang and Yang, Cheng-Ta and Wan, Yung-Liang and Lin, Gigin and Sequist, Lecia V and Fintelmann, Florian J. and Barzilay, Regina},
  journal={Journal of Clinical Oncology},
  pages={JCO--22},
  year={2023},
  publisher={Wolters Kluwer Health}
}
```
