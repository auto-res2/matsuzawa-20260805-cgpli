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
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install uv from PyPI rather than `COPY --from=ghcr.io/astral-sh/uv`, because
# a cross-registry COPY --from has failed under the Kaniko builder used here.
RUN pip install --no-cache-dir uv==0.7.2

# --- OpenStructure, via micromamba -------------------------------------
# NOTE TO ANY AGENT REGENERATING THIS FILE: keep this whole block.
# OpenStructure is the reference implementation of lDDT-PLI and is what
# PLINDER's own evaluation uses. It is NOT pip-installable -- the PyPI
# package named `ost` is an unrelated subtitle library -- so bioconda is
# the only source. bioconda publishes linux-aarch64 builds, but 2.11.1 on
# aarch64 requires Python 3.12, which is why this is a SEPARATE conda
# prefix rather than the uv venv (that one is Python 3.11, fixed by the
# repository's CLI contract). The two never share a prefix: scoring runs
# as a subprocess against /opt/ost/bin/python and hands JSON back.
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
RUN set -eux; \
    case "$(uname -m)" in \
      aarch64) MARCH=linux-aarch64 ;; \
      x86_64)  MARCH=linux-64 ;; \
      *) echo "unsupported arch $(uname -m)" >&2; exit 1 ;; \
    esac; \
    curl -Ls "https://micro.mamba.pm/api/micromamba/${MARCH}/latest" \
      | tar -xvj -C /usr/local bin/micromamba; \
    micromamba create -y -p /opt/ost -c conda-forge -c bioconda \
      python=3.12 openstructure=2.11.1; \
    micromamba clean --all --yes; \
    /opt/ost/bin/python -c "from ost.mol.alg.ligand_scoring_lddtpli import LDDTPLIScorer; print('openstructure ok')"

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
