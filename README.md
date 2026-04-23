# Sybil

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pgmikhael/Sybil/blob/main/LICENSE.txt) ![version](https://img.shields.io/badge/version-1.7.2-success)

Lung Cancer Risk Prediction.

Additional documentation can be found on the [GitHub Wiki](https://github.com/reginabarzilaygroup/Sybil/wiki).

## Run a regression test

```shell
python tests/regression_test.py
```

This will download the `sybil_ensemble` model and sample data, and compare the results to what has previously been calculated.

## Run the model

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

All model files are available on [GitHub releases](https://github.com/reginabarzilaygroup/Sybil/releases) as well as on [Google Drive](https://drive.google.com/drive/folders/1nBp05VV9mf5CfEO6W5RY4ZpcpxmPDEeR?usp=sharing).

## Replicating results

You can replicate the results from our model using our training script:

```sh
python train.py
```

See our [documentation](docs/readme.md) for a full description of Sybil's training parameters. Additional information on the training process can be found on the [train branch](https://github.com/reginabarzilaygroup/Sybil/tree/train) of this repository.

## Sybil-2

Sybil-2 is a new version of the Sybil model that can ingest multiple scans and performs nodule segmentation and tracking.

### Environment

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. The pinned `uv.lock` is the single source of truth for versions; it targets Linux x86_64 with CUDA.

1. Install uv:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create the venv and install Sybil v2 dependencies. `rad-vision-engine` is already declared as a git source in `pyproject.toml`, so `uv sync` pulls it automatically — no separate clone step:

```sh
uv sync --extra v2
```

3. Install `torch-scatter`. It's not in the lock file because it ships CUDA-specific prebuilt wheels from PyG that aren't indexed on PyPI; the URL must match the pinned torch version (2.7.1+cu126):

```sh
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu126.html
```

4. Download the Pillar model checkpoint from [YalaLab/Pillar0-Sybil-1.5](https://huggingface.co/YalaLab/Pillar0-Sybil-1.5) and save it as `pillar_seed0_epoch=2.ckpt` in the Sybil cache directory (default: `~/.sybil/`). The Sybil2 core checkpoints will be downloaded automatically on first use:

```sh
huggingface-cli download YalaLab/Pillar0-Sybil-1.5 pillar_seed0_epoch=2.ckpt --local-dir ~/.sybil/
```

Activate the venv with `source .venv/bin/activate`, or run commands through `uv run <command>`.

Alternatively, use the pre-built Docker image — see `Dockerfile` and `docs/docker_guide.md`. The image includes all dependencies and bakes in the checkpoints.

### Run the model on a single exam

```python
from sybil import Serie, Sybil2

model = Sybil2()

serie = Serie(
    dicoms={
        "baseline": [dicom_path_1, dicom_path_2, ...],
        "followup": [dicom_path_1, dicom_path_2, ...],
    },
    version="v2",
    cache_dir="/tmp/sybil_cache",
)
prediction = model.predict([serie])
print(prediction.scores)   # [[year1_risk, year2_risk, ...]]
```

### Batch inference from a CSV

For large cohorts, use `SybilV2Dataset` together with `Sybil2.predict_dataset`.

#### CSV format

Each row is one CT scan (timepoint) for a patient. Multiple rows with the same
`patient_id` are treated as a longitudinal series.

| column | required | description |
| --- | --- | --- |
| `patient_id` | ✓ | Unique patient identifier |
| `timepoint` | ✓ | Integer ordering the scans chronologically (e.g. `0` for baseline, `1` for the first followup) |
| `ct_dir` | ✓ | Path to the directory containing the DICOM files for that scan |
| `label` | | Cancer label (0 / 1) |
| `censor_time` | | Years to cancer diagnosis (required when `label` is provided) |

```csv
patient_id,timepoint,ct_dir
P001,0,/data/P001/scan_2018
P001,1,/data/P001/scan_2019
P002,0,/data/P002/scan_2020
```

#### Single-GPU

```python
from sybil import Sybil2

model = Sybil2()
results = model.predict_dataset(
    dataset="cohort.csv",
    output_path="predictions.csv",
    cache_dir="/tmp/sybil_cache",
    batch_size=4,       # patients per batch
    num_workers=4,      # DataLoader workers
)
# results is a list of dicts: patient_id, scores, year_1_risk, ...
```

#### Multi-GPU (single node)

Create a launcher script, e.g. `run_inference.py`:

```python
import torch
from sybil import Sybil2

torch.distributed.init_process_group(backend="nccl")

model = Sybil2()
results = model.predict_dataset(
    dataset="cohort.csv",
    output_path="predictions.csv",   # written once by rank 0
    cache_dir="/tmp/sybil_cache",
    batch_size=4,
    num_workers=4,
    distributed=True,
)

torch.distributed.destroy_process_group()
```

Then launch with `torchrun`:

```sh
# 4 GPUs on one machine
torchrun --standalone --nproc_per_node=4 run_inference.py
```

#### Multi-node

```sh
# 2 nodes, 4 GPUs each (8 GPUs total)
# Run on node 0:
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=0 --master_addr=<node0_ip> --master_port=29500 \
         run_inference.py

# Run on node 1:
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=1 --master_addr=<node0_ip> --master_port=29500 \
         run_inference.py
```

#### How batching works

| Step | Scope | Parallelism |
| --- | --- | --- |
| DICOM → NIfTI + volume loading | per patient | DataLoader workers |
| Lung mask + nodule segmentation (nnUNet) | per patient, per timepoint | Sequential on GPU (variable volume sizes) |
| Confidence model (fixed 128×128×32 patches) | **all patients in batch** | Single batched forward pass |
| CT registration (ANTs rigid) | per patient | `ThreadPoolExecutor` — parallel across patients |
| Sybil2 risk forward | per patient | Sequential |
| Patient assignment across GPUs | across GPUs | `DistributedSampler` |

### Command-line interface

The `sybil-predict` CLI supports two sub-commands.

#### Single exam (Sybil v1)

```sh
sybil-predict single /path/to/dicoms --output-dir results/
# with attention visualisation
sybil-predict single /path/to/dicoms --output-dir results/ --write-attention-images
```

#### Batch cohort (Sybil2) via CLI

```sh
# Single GPU
sybil-predict batch cohort.csv \
    --output results/predictions.csv \
    --cache-dir /tmp/sybil_cache

# Multi-GPU (4 GPUs, one node)
torchrun --standalone --nproc_per_node=4 \
    -m sybil.predict batch cohort.csv \
    --output results/predictions.csv \
    --cache-dir /tmp/sybil_cache \
    --distributed
```

Full option reference: `sybil-predict batch --help` / `sybil-predict single --help`.

## LDCT Orientation

The model expects the input to be an Axial LDCT, where the first frame is of the abdominal region and the last frame is along the clavicles.

When the input is of the `dicom` type, the frames will be automatically sorted. However, for `png` inputs, the path of the PNG files must be in the right anatomical order.

## Annotations

To help train the model, two fellowship-trained thoracic radiologists jointly annotated suspicious lesions on NLST LDCTs using [MD.AI](https://md.ai) software for all participants who developed cancer within 1 year after an LDCT. Each lesion's volume was marked with bounding boxes on contiguous thin-cut axial images. The "ground truth" annotations were informed by the imaging appearance and the clinical data provided by the NLST, i.e., the series and image number of cancerous nodules and the anatomical location of biopsy-confirmed lung cancers. For these participants, lesions in the location of subsequently diagnosed cancers were also annotated, even if the precursor lesion lacked imaging features specific for cancer.

Annotations are available to download in JSON format from [Google Drive](https://drive.google.com/file/d/19aa5yIHPWu3NtjqvXDc8NYB2Ub9V-4WM/view?usp=share_link). The JSON file is structured as below, where `(x,y)` refers to the top left corner of the bounding box, and all values are normalized to the image size (512,512).

```json
{
  "series1_id": {
    "image1_id": [
      {"x": 0.1, "y": 0.2, "height": 0.05, "width": 0.05},
      {"x": 0.3, "y": 0.4, "height": 0.05, "width": 0.05}
    ],
    "image2_id": []
  },
  "series2_id": {}
}
```

## Attention Scores

The multi-attention pooling layer aims to learn the importance of each slice in the 3D volume and the importance of each pixel in the 2D slice. During training, these are supervised by bounding boxes of the cancerous nodules. This is a soft attention mechanism, and the model's primary task is to predict the risk of lung cancer. However, the attention scores can be extracted and used to visualize the model's focus on the 3D volume and the 2D slices.

To extract the attention scores, use the `return_attentions` argument:

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
    attentions=attentions,
    save_directory="path_to_save_directory",
    gain=3,
)
```

## Training Data

The Sybil model was trained using the National Lung Screening Trial (NLST) dataset:

National Lung Screening Trial Research Team. (2013). Data from the National Lung Screening Trial (NLST) [Data set]. The Cancer Imaging Archive. <https://doi.org/10.7937/TCIA.HMQ8-J677>

## Cite

```bibtex
@article{mikhael2023sybil,
  title={Sybil: a validated deep learning model to predict future lung cancer risk from a single low-dose chest computed tomography},
  author={Mikhael, Peter G and Wohlwend, Jeremy and Yala, Adam and Karstens, Ludvig and Xiang, Justin and Takigami, Angelo K and Bourgouin, Patrick P and Chan, PuiYee and Mrah, Sofiane and Amayri, Wael and Juan, Yu-Hsiang and Yang, Cheng-Ta and Wan, Yung-Liang and Lin, Gigin and Sequist, Lecia V and Fintelmann, Florian J. and Barzilay, Regina},
  journal={Journal of Clinical Oncology},
  pages={JCO--22},
  year={2023},
  publisher={Wolters Kluwer Health}
}
```
