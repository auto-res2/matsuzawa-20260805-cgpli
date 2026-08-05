# Execution environment for the CG-PLI experiment.
#
# NOTE TO ANY AGENT REGENERATING THIS FILE: every instruction below states
# why it is here. The target runtime is aarch64 (NVIDIA GB200, Grace-Blackwell)
# running this image as an Apptainer SIF, so please preserve the intent:
#   * do NOT add unpinned `uv pip install` lines -- every dependency is
#     already pinned in pyproject.toml and locked in uv.lock, and each one was
#     verified to publish a cp311 manylinux aarch64 wheel. An unpinned install
#     resolves to whatever is newest and will pull a package with no aarch64
#     wheel (this is how torch 2.7 drags in triton 3.3.0, which is x86-only).
#   * do NOT mix `uv pip install --system` with `uv run`; `uv run` creates its
#     own .venv and would not see system-installed packages.
#   * this workload needs no CUDA, no torch and no nvcc. It is CPU-bound
#     cheminformatics scoring.

# Python 3.11 as required by the repository contract; slim keeps the SIF small
# because the image is pulled onto every compute node.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git: not used at run time but some tooling probes for it.
# curl + ca-certificates: the run downloads PLINDER data over HTTPS from
#   storage.googleapis.com, so the CA bundle must be present.
# libgomp1: OpenMP runtime that the scipy and rdkit aarch64 wheels link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv from PyPI rather than `COPY --from=ghcr.io/astral-sh/uv`, because
# a cross-registry COPY --from has failed under the Kaniko builder used here.
RUN pip install --no-cache-dir uv==0.7.2

WORKDIR /workspace

# Copy the dependency manifest and the lock together. The lock is committed
# on purpose: `uv sync --frozen` must fail loudly if it is missing rather than
# silently re-resolving to versions that have no aarch64 wheels.
COPY pyproject.toml uv.lock ./

# --frozen with no fallback. A bare `|| uv sync` would swallow a stale lock,
# which is exactly the failure this pinning is meant to prevent.
RUN uv sync --frozen --no-install-project

COPY . .

# results_dir the CLI contract writes into.
RUN mkdir -p .research/results

CMD ["bash"]
