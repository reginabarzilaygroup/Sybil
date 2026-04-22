# Changelog

## Version 1.7.1

## 2026-04-22 — Atlas backbone switched to public repo

`sybil.models.pillar.backbones.mmatlas.MultimodalAtlas` is now hardcoded to
`YalaLab/Pillar0-ChestCT` (public, token-auth) instead of the private
`YalaLab/PILLAR0-atlas_small_qwen8b_clip_ucsf_chest_ct_inpt_ep300__exp172`.

- **`mmatlas.py`**: `__init__` still accepts `model_repo_id` / `model_revision` / `pretrained` so checkpoints whose `args.model["kwargs"]` carry those fields keep loading, but the values are ignored — the repo id, revision, and `pretrained=True` are set internally.
- **`Dockerfile`**: step 3 (atlas pre-cache) pulls `YalaLab/Pillar0-ChestCT` at revision `main`.
- **`docs/docker_test_plan.md`**: prerequisites and Phase 5c updated to reference the public repo.

## 2026-04-21 — uv migration & Dockerization

### 1. `pyproject.toml` — Migrated to uv

Consolidated `setup.cfg` + old `pyproject.toml` + `environment-v2.yml` into a single modern `pyproject.toml`:

- **Python**: `>=3.10,<3.12` (nnunetv2 requires >=3.10; environment uses 3.11)
- **Base deps**: Sybil v1 essentials (torch, torchvision, pydicom, imageio, etc.)
- **`[v2]` extra**: Full Sybil2 stack — antspyx, nnunetv2, lungmask, transformers, monai, etc.
- **`[train]`/`[testing]` extras**: training and dev tooling
- **`rad-vision-engine`** sourced from `github.com/YalaLab/rave` via `[tool.uv.sources]` (public HTTPS)
- **`torch-scatter`** excluded from lock file — requires CUDA-specific prebuilt wheels from PyG's find-links index, documented inline with the install command
- **Resolution restricted** to `linux x86_64` via `[tool.uv.environments]` since this is a CUDA-only project

### 2. `uv.lock` — 183 packages resolved

Fully reproducible lock file for all dependencies.

### 3. `Dockerfile` — Sybil2 deployment image

- **Base**: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
- **uv** copied from official image, installs Python 3.11 into a venv
- **Deps**: `uv sync --frozen --extra v2` from lock file, then `torch-scatter` from PyG wheels
- **rave**: installed from GitHub
- **3 model download steps** baked into the image:
  1. Sybil2 checkpoints from Zenodo (nnUNet, lungmask, confidence, risk, calibrator)
  2. Pillar0-Sybil-1.5 checkpoint from HuggingFace (gated)
  3. Atlas foundation model (`PILLAR0-atlas_small_qwen8b_clip_ucsf_chest_ct_inpt_ep300__exp172`) pre-cached from HuggingFace
- **BuildKit secrets** (`--secret id=hf_token`) for HF_TOKEN — never baked into image layers
- **Layer caching**: dependency install is separate from source copy for fast rebuilds

### 4. `.dockerignore`

Excludes `.git`, tests, docs, notebooks, `CLAUDE.md` from build context.

## Version 1.7.0

Sybil2 multi-timepoint model, training code, and Docker image.
