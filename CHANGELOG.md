# Changelog

## Version 1.7.2

## 2026-04-23 — Pin torch + transformers + route torch through PyTorch cu126 index

`pyproject.toml` now pins the deep-learning stack explicitly:

- `torch==2.7.1` (was `>=2.0`) — the unbounded range resolved to `2.11.0`, which bundles CUDA 13.0 and requires NVIDIA driver ≥ 580. 2.7.1 matches the tested local environment and works against the far-more-common cu126 driver line.
- `torchvision==0.22.1` — the sibling wheel for torch 2.7.1.
- `transformers>=4.40,<5` — transformers 5.x refactored its weight-tying API (`all_tied_weights_keys` vs `_tied_weights_keys`), which broke custom models loaded via `trust_remote_code=True` (notably the `YalaLab/Pillar0-ChestCT` atlas backbone). Upper bound stays on the 4.x line.

A new `[[tool.uv.index]]` block and `[tool.uv.sources]` entries route `torch` and `torchvision` through `https://download.pytorch.org/whl/cu126` instead of PyPI's default cu13x wheels. The PyG torch-scatter find-links URL is updated to `torch-2.7.1+cu126.html` to match.

## 2026-04-23 — Drop conda environment files

Deleted `environment.yml` and `environment-v2.yml`. `pyproject.toml` + `uv.lock` is now the single source of truth for dependencies:

- Sybil v1 deps live in the `[project]` table (used by `uv sync`).
- Sybil v2 deps live under the `[v2]` extra (`uv sync --extra v2`).
- `rad-vision-engine` is declared as a git source in `[tool.uv.sources]`, so no separate clone + `pip install -e` step is needed.
- `torch-scatter` still has to be installed via `uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu126.html` (CUDA-specific wheel not on PyPI).

The conda files were drifting from `pyproject.toml` (version pins like `torch==2.7.1`, `transformers<5` were never mirrored back into the YAMLs). Updated `README.md` §Environment and `CLAUDE.md` §"Sybil2 environment" to show the `uv` workflow.

## 2026-04-22 — Zenodo download is back in the image

The Zenodo `RUN` block is re-enabled in the Dockerfile, so `docker build` once again bakes the ~2 GB checkpoints into the image. No runtime mounts or env-var overrides are needed to invoke Sybil2 — the container has `/root/.sybil/` populated out of the box.

**Fallback if a specific run needs a different checkpoint set** (e.g. a pinned Zenodo revision): `sybil.model.CHECKPOINT2_URL` is read at import via `os.getenv("SYBIL_CHECKPOINT2_URL", <default>)`, so `docker run -e SYBIL_CHECKPOINT2_URL="...?token=..." -v /tmp/cache:/root/.sybil` still works as an override — but this is no longer the default testing path.

## 2026-04-22 — CLI: decouple mode from model version

`sybil-predict` previously conflated `single` with Sybil v1 and `batch` with Sybil v2. Mode and version are now independent:

- **`single`** accepts `--model-version {v1, v2}` (default `v1`, backwards-compatible).
  - `v2` mode takes the positional `image_dir` as the baseline timepoint and adds `--followup <dir>` for optional longitudinal inference.
  - `v2` mode requires `--cache-dir` (intermediate NIfTI files).
  - `--return-attentions` / `--write-attention-images` remain v1-only and raise a parse error when combined with `v2`.
- **`batch`** accepts `--model-version {v2}` only — argparse rejects `v1` at parse time because Sybil v1 has no cohort runner.
- **Root parser** gets its own `--version` flag so `sybil-predict --version` works without a subcommand.

See `docs/docker_guide.md` Phase 3 for the updated v2 single-patient test commands.

## 2026-04-22 — Zenodo checkpoints record is restricted

The Sybil2 core-checkpoints Zenodo record (`19323196`) is access-restricted. The Dockerfile now:

- Accepts a second BuildKit secret `id=zenodo_token` for a Zenodo share-link token
- Sets `SYBIL_CHECKPOINT2_URL` at the checkpoint-download step so `sybil.model.CHECKPOINT2_URL` (read via `os.getenv`, see `sybil/model.py:97`) picks up the tokenised URL — no code change needed
- Build command now requires `--secret id=zenodo_token,env=ZENODO_TOKEN` alongside the existing `hf_token` secret

See `docs/docker_guide.md` for the updated prerequisites and Phase 0 build command.

## Version 1.7.1

## 2026-04-22 — Atlas backbone switched to public repo

`sybil.models.pillar.backbones.mmatlas.MultimodalAtlas` is now hardcoded to
`YalaLab/Pillar0-ChestCT` (public, token-auth) instead of the private
`YalaLab/PILLAR0-atlas_small_qwen8b_clip_ucsf_chest_ct_inpt_ep300__exp172`.

- **`mmatlas.py`**: `__init__` still accepts `model_repo_id` / `model_revision` / `pretrained` so checkpoints whose `args.model["kwargs"]` carry those fields keep loading, but the values are ignored — the repo id, revision, and `pretrained=True` are set internally.
- **`Dockerfile`**: step 3 (atlas pre-cache) pulls `YalaLab/Pillar0-ChestCT` at revision `main`.
- **`docs/docker_guide.md`**: prerequisites and Phase 5c updated to reference the public repo.

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
