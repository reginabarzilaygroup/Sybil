# ---------------------------------------------------------------------------
# Sybil2 inference image
# ---------------------------------------------------------------------------
# Build:
#   export HF_TOKEN=...          # HuggingFace (gated Pillar0-Sybil-1.5 + Pillar0-ChestCT)
#   export ZENODO_TOKEN=...      # Zenodo share-link token for the restricted checkpoints record
#   docker build \
#     --secret id=hf_token,env=HF_TOKEN \
#     --secret id=zenodo_token,env=ZENODO_TOKEN \
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

# System dependencies for ANTs, OpenCV, ITK, and medical imaging libs.
# ffmpeg is required at runtime by rad-vision-engine (sybil.loaders.rve) for
# video encoding of CT volumes — it shells out via subprocess.run.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        vim nano less \
        ffmpeg \
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
RUN mkdir -p sybil && echo '__version__ = "1.7.2"' > sybil/__init__.py

# Install locked dependencies (v2 extra) without the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra v2 --no-install-project

# torch-scatter requires CUDA-specific wheels from PyG (not in lock file).
# The wheel MUST match the pinned torch version (2.7.1) and its CUDA variant
# (cu126 — selected via [tool.uv.sources] pytorch-cu126 index in pyproject.toml).
# A mismatch segfaults at import or raises a runtime error on first call.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch-scatter \
        -f https://data.pyg.org/whl/torch-2.7.1+cu126.html

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -e .

# ---------------------------------------------------------------------------
# Model checkpoints
# ---------------------------------------------------------------------------
ENV SYBIL_CACHE=/root/.sybil

# 1. Sybil2 core checkpoints from Zenodo (nnUNet, confidence, lungmask, risk, calibrator).
# Restricted record — requires a Zenodo share-link token (passed via BuildKit secret).
# The zip is ~2 GB, so we download via curl with resume + retries rather than
# sybil.model.download_and_extract's urlopen.read() — that one pull frequently
# drops mid-stream in BuildKit and raises http.client.IncompleteRead.
RUN --mount=type=secret,id=zenodo_token \
    ZENODO_TOKEN=$(cat /run/secrets/zenodo_token) && \
    mkdir -p "${SYBIL_CACHE}" && \
    curl --fail --location --continue-at - \
         --retry 10 --retry-delay 10 --retry-max-time 3600 --retry-all-errors \
         -o /tmp/sybil2_checkpoints.zip \
         "https://zenodo.org/records/19378950/files/sybil2_checkpoints.zip?token=${ZENODO_TOKEN}" && \
    python -c "\
import os, shutil, zipfile; \
cache = os.environ['SYBIL_CACHE']; \
tmp = '/tmp/sybil2_extract'; \
zipfile.ZipFile('/tmp/sybil2_checkpoints.zip').extractall(tmp); \
entries = os.listdir(tmp); \
inner = os.path.join(tmp, entries[0]); \
src = inner if len(entries) == 1 and os.path.isdir(inner) else tmp; \
[shutil.move(os.path.join(src, f), os.path.join(cache, f)) for f in os.listdir(src)]; \
shutil.rmtree(tmp)" && \
    rm -f /tmp/sybil2_checkpoints.zip

# 2. Pillar0-Sybil-1.5 (gated HF repo — requires HF_TOKEN at build time)
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) && \
    export HF_TOKEN && \
    python -c "\
import os, shutil; \
from huggingface_hub import hf_hub_download; \
path = hf_hub_download( \
    repo_id='YalaLab/Pillar0-Sybil-1.5', \
    filename='seed0/epoch=2.ckpt', \
    token=os.environ['HF_TOKEN'], \
    local_dir='/tmp/pillar'); \
shutil.copy(path, os.path.join(os.environ['SYBIL_CACHE'], 'pillar_seed0_epoch=2.ckpt'))"

# 3. Atlas foundation model (pre-cache so it is not downloaded at runtime)
# MultimodalAtlas is hardcoded to YalaLab/Pillar0-ChestCT (public, token-auth).
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) && \
    export HF_TOKEN && \
    python -c "\
import os; \
from transformers import AutoModel; \
AutoModel.from_pretrained( \
    'YalaLab/Pillar0-ChestCT', \
    revision='main', \
    token=os.environ['HF_TOKEN'], \
    trust_remote_code=True)"

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
ENTRYPOINT ["sybil-predict"]
CMD ["--help"]
