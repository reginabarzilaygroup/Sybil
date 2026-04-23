# Docker Guide — Sybil / Sybil2

## Prerequisites

1. **HuggingFace token** with access to:
   - [YalaLab/Pillar0-Sybil-1.5](https://huggingface.co/YalaLab/Pillar0-Sybil-1.5) (gated — accept gating conditions first)
   - [YalaLab/Pillar0-ChestCT](https://huggingface.co/YalaLab/Pillar0-ChestCT) (public atlas foundation model used by `MultimodalAtlas`; token still required for HF auth)

2. **Zenodo share-link token** for the restricted Sybil2 checkpoints record (record `19323196`). Ask the record owner for a share link with **download access to restricted files** — the URL's `?token=...` query parameter is the value you need.

3. **Docker with BuildKit and GPU support**:
   ```bash
   docker --version          # >= 20.10
   nvidia-smi                # CUDA drivers visible
   docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
   ```

4. **Test data**: two NLST DICOM series for the same patient (patient 122361):
   - T0: `1.2.840.113654.2.55.81641439048624235905678753284956900652`
   - T1: `1.2.840.113654.2.55.210451208063625047828616019396666958685`

   Download them before building (so they're available for bind-mounting):
   ```bash
   mkdir -p /tmp/sybil_test_data
   python -c "
   from tests.test_sybil2 import _get_nlst
   _get_nlst('1.2.840.113654.2.55.81641439048624235905678753284956900652', '/tmp/sybil_test_data')
   _get_nlst('1.2.840.113654.2.55.210451208063625047828616019396666958685', '/tmp/sybil_test_data')
   "
   ```

---

## Phase 0: Build the image

```bash
export HF_TOKEN=hf_...
export ZENODO_TOKEN=...   # share-link token for the restricted Zenodo record

DOCKER_BUILDKIT=1 docker build \
  --network=host \
  --secret id=hf_token,env=HF_TOKEN \
  --secret id=zenodo_token,env=ZENODO_TOKEN \
  -t sybil2 .
```

If you're running as root, use `sudo -E` so the env vars survive the sudo call:

```bash
sudo -E DOCKER_BUILDKIT=1 docker build \
  --network=host \
  --secret id=hf_token,env=HF_TOKEN \
  --secret id=zenodo_token,env=ZENODO_TOKEN \
  -t sybil2 .
```

**Pass criteria**: build completes without error. All 3 model download steps succeed.

---

## Phase 1: Smoke tests

These verify the image is well-formed and the CLI loads correctly.

### 1a. CLI help

```bash
docker run --rm sybil2 --help
```

**Expected**: prints the usage message with `single` and `batch` subcommands.

### 1b. Version

```bash
docker run --rm sybil2 --version
```

**Expected**: prints `1.7.2`.

### 1c. Python imports

```bash
docker run --rm --gpus all --entrypoint python sybil2 -c "
from sybil import Sybil, Sybil2, Serie, __version__
import torch, transformers, ants, lungmask, rve
print(f'sybil={__version__}, torch={torch.__version__}, cuda={torch.cuda.is_available()}')
"
```

**Expected**: all imports succeed, `cuda=True`.

Without `--gpus all` the container has no GPU device exposed, so `torch.cuda.is_available()` returns `False` even though torch itself is a CUDA build (you'll see `+cu130` in the version string). That's the GPU-passthrough flag, not a torch install issue.

### 1d. CLI mode/version validation

These should all exit non-zero with a clear argparse error *before* any model loading:

```bash
# followup needs v2
docker run --rm sybil2 single /tmp --followup /tmp
# cache-dir needed for v2
docker run --rm sybil2 single /tmp --model-version v2
# cache-dir rejected for v1
docker run --rm sybil2 single /tmp --model-version v1 --cache-dir /tmp
# attentions are v1-only
docker run --rm sybil2 single /tmp --model-version v2 --cache-dir /tmp --return-attentions
# batch rejects v1
docker run --rm sybil2 batch /tmp/cohort.csv --model-version v1
```

**Expected**: each prints a short `usage: ... error: ...` line and exits 2. None of them should get as far as loading PyTorch or Sybil — the errors come from argparse / `parser.error()` in `sybil/predict.py`.

---

## Phase 2: Sybil v1 — single exam

Runs the original Sybil ensemble on one DICOM directory (T0 scan). `--model-version v1` is the default for `single`, so it's omitted here — pass it explicitly if you want to be unambiguous.

```bash
docker run --rm --gpus all \
  -v /tmp/sybil_test_data:/data:ro \
  -v /tmp/sybil_results:/output \
  sybil2 \
  single /data/1.2.840.113654.2.55.81641439048624235905678753284956900652 \
    --output-dir /output/v1 \
    --file-type dicom
```

**Pass criteria**:
- [ ] Exit code 0
- [ ] `/tmp/sybil_results/v1/prediction_scores.json` exists
- [ ] JSON contains `"predictions"` key with a list of 1 patient, each having 6 floats (year 1-6 risk scores)
- [ ] All scores are in `[0, 1]` and monotonically non-decreasing

**Verify**:
```bash
cat /tmp/sybil_results/v1/prediction_scores.json | python -m json.tool
```

---

## Phase 3: Sybil v2 — single patient (CLI)

The `single` subcommand now accepts `--model-version v2` for longitudinal inference on a baseline + optional followup DICOM directory.

```bash
docker run --rm --gpus all \
  -v /tmp/sybil_test_data:/data:ro \
  -v /tmp/sybil_results:/output \
  sybil2 \
    single /data/1.2.840.113654.2.55.81641439048624235905678753284956900652 \
    --model-version v2 \
    --followup /data/1.2.840.113654.2.55.210451208063625047828616019396666958685 \
    --cache-dir /tmp/cache \
    --output-dir /output/v2_single \
    --file-type dicom
```

**Pass criteria**:
- [ ] Exit code 0
- [ ] Prints the prediction JSON to stdout (6 risk scores)
- [ ] `/tmp/sybil_results/v2_single/prediction_scores.json` exists
- [ ] All 6 scores are in `[0, 1]` and monotonically non-decreasing

### 3b. Baseline-only (no followup)

```bash
docker run --rm --gpus all \
  -v /tmp/sybil_test_data:/data:ro \
  -v /tmp/sybil_results:/output \
  sybil2 \
    single /data/1.2.840.113654.2.55.81641439048624235905678753284956900652 \
    --model-version v2 \
    --cache-dir /tmp/cache \
    --output-dir /output/v2_single_baseline \
    --file-type dicom
```

Sybil2 supports single-timepoint inference too. Same pass criteria as above.

---

## Phase 4: Sybil2 — batch cohort (CSV manifest)

`batch` runs Sybil v2 only — `--model-version v2` is the default. Passing `--model-version v1` is rejected by argparse at parse time since Sybil v1 has no cohort runner.

### 4a. Create test CSV

```bash
cat > /tmp/sybil_test_data/cohort.csv <<'EOF'
patient_id,timepoint,ct_dir,label,censor_time
122361,0,/data/1.2.840.113654.2.55.81641439048624235905678753284956900652,0,6
122361,1,/data/1.2.840.113654.2.55.210451208063625047828616019396666958685,0,6
EOF
```

### 4b. Run batch inference (single GPU)

```bash
docker run --rm --gpus all \
  -v /tmp/sybil_test_data:/data:ro \
  -v /tmp/sybil_results:/output \
  sybil2 \
  batch /data/cohort.csv \
    --output /output/v2_batch/predictions.csv \
    --cache-dir /tmp/cache \
    --batch-size 1 \
    --num-workers 0
```

**Pass criteria**:
- [ ] Exit code 0
- [ ] Prints `Scored 1 patient(s). Results written to /output/v2_batch/predictions.csv`
- [ ] `/tmp/sybil_results/v2_batch/predictions.csv` exists and contains a row for patient `122361`
- [ ] Risk scores are in `[0, 1]`

**Verify**:
```bash
cat /tmp/sybil_results/v2_batch/predictions.csv
```

### 4c. Run batch inference (multi-GPU, if available)

Only run this if the host has 2+ GPUs.

```bash
docker run --rm --gpus all \
  -v /tmp/sybil_test_data:/data:ro \
  -v /tmp/sybil_results:/output \
  --entrypoint torchrun sybil2 \
    --standalone --nproc_per_node=2 \
    -m sybil.predict batch /data/cohort.csv \
      --output /output/v2_multi/predictions.csv \
      --cache-dir /tmp/cache \
      --batch-size 1 \
      --num-workers 0 \
      --distributed
```

**Pass criteria**:
- [ ] Exit code 0
- [ ] Output file is identical to the single-GPU result (same patient, same scores within floating-point tolerance)

---

## Phase 5: Resource and cleanup checks

### 5a. No HF_TOKEN leaked in the image

```bash
docker run --rm --entrypoint env sybil2 | grep -i hf_token
```

**Expected**: either no output or `HF_TOKEN=` (empty). Token must not be baked in.

### 5b. Model checkpoints are present

```bash
docker run --rm --entrypoint bash sybil2 -c "ls -lh /root/.sybil/"
```

**Expected**: checkpoint files for risk, segmentation, confidence, lungmask, malignancy, calibrator, and `pillar_seed0_epoch=2.ckpt`.

### 5c. Atlas model is cached

```bash
docker run --rm --entrypoint python sybil2 -c "
from transformers import AutoModel
m = AutoModel.from_pretrained(
    'YalaLab/Pillar0-ChestCT',
    revision='main',
    trust_remote_code=True,
    local_files_only=True
)
print(f'Atlas model loaded from cache: {type(m).__name__}')
"
```

**Expected**: loads without network access (`local_files_only=True`). Prints model class name.

---

## Cleanup

```bash
rm -rf /tmp/sybil_test_data /tmp/sybil_results
docker rmi sybil2
```
