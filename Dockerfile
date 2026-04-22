# ---------------------------------------------------------------------------
# Sybil2 inference image
# ---------------------------------------------------------------------------
# Build:
#   docker build \
#     --secret id=hf_token,env=HF_TOKEN \
#     -t sybil2 .
#
# Run (single exam, v1):
#   docker run --gpus all -v /data:/data sybil2 \
#     sybil-predict single /data/dicoms --output-dir /data/results
#
# Run (batch cohort, v2):
#   docker run --gpus all -v /data:/data sybil2 \
#     sybil-predict batch /data/cohort.csv --output /data/results/predictions.csv
#
# Run (multi-GPU batch):
#   docker run --gpus all -v /data:/data sybil2 \
#     torchrun --standalone --nproc_per_node=4 \
#       -m sybil.predict batch /data/cohort.csv --distributed
# ---------------------------------------------------------------------------

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# System dependencies for ANTs, OpenCV, ITK, and medical imaging libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# uv + Python
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN uv python install 3.11

# Virtual environment on PATH so all RUN/ENTRYPOINT use it
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
RUN uv venv --python 3.11

# ---------------------------------------------------------------------------
# Dependency layer (cached until pyproject.toml or uv.lock changes)
# ---------------------------------------------------------------------------
COPY pyproject.toml uv.lock setup.py setup.cfg README.md LICENSE.txt ./
RUN mkdir -p sybil && echo '__version__ = "1.7.1"' > sybil/__init__.py

# Install locked dependencies (v2 extra) without the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra v2 --no-install-project

# torch-scatter requires CUDA-specific wheels from PyG (not in lock file)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch-scatter \
        -f https://data.pyg.org/whl/torch-2.7.0+cu126.html

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra v2 --no-deps

# ---------------------------------------------------------------------------
# Model checkpoints
# ---------------------------------------------------------------------------
ENV SYBIL_CACHE=/root/.sybil

# 1. Sybil2 core checkpoints from Zenodo (nnUNet, confidence, lungmask, risk, calibrator)
RUN python -c " \
from sybil.model import download_and_extract, CHECKPOINT2_URL; \
import os; os.makedirs(os.environ['SYBIL_CACHE'], exist_ok=True); \
download_and_extract(CHECKPOINT2_URL, os.environ['SYBIL_CACHE'])"

# 2. Pillar0-Sybil-1.5 (gated HF repo — requires HF_TOKEN at build time)
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null) && \
    python -c " \
import os, shutil; \
from huggingface_hub import hf_hub_download; \
path = hf_hub_download( \
    repo_id='YalaLab/Pillar0-Sybil-1.5', \
    filename='seed0/epoch=2.ckpt', \
    token=os.environ.get('HF_TOKEN', None), \
    local_dir='/tmp/pillar'); \
shutil.copy(path, os.path.join(os.environ['SYBIL_CACHE'], 'pillar_seed0_epoch=2.ckpt'))"

# 3. Atlas foundation model (pre-cache so it is not downloaded at runtime)
# MultimodalAtlas is hardcoded to YalaLab/Pillar0-ChestCT (public, token-auth).
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null) && \
    python -c " \
import os; \
from transformers import AutoModel; \
AutoModel.from_pretrained( \
    'YalaLab/Pillar0-ChestCT', \
    revision='main', \
    token=os.environ.get('HF_TOKEN', None), \
    trust_remote_code=True)"

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
ENTRYPOINT ["sybil-predict"]
CMD ["--help"]
