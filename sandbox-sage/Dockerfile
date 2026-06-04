# Sandbox image with SageMath via conda-forge. Natively multi-arch
# (linux/arm64 + linux/amd64) so this same image works on Apple Silicon
# Macs and x86_64 Linux without emulation.
#
# Why conda-forge rather than upstream sagemath/sagemath:
#   - sagemath/sagemath:latest is published only for linux/amd64 — no
#     arm64 manifest, so it requires QEMU/Rosetta emulation on M-series
#     Macs. Significant runtime cost.
#   - conda-forge ships `sage` as a first-class package (maintained by
#     the Sage team at conda-forge/sage-feedstock) with native builds
#     for both arches.
#   - miniforge3 base is multi-arch and small (~200 MB before sage).
#
# Tradeoff: conda-forge's sage can trail upstream releases by 1–2
# point versions. Not relevant here — we're calling sage as a
# computation tool, not developing on the newest Sage internals.
#
# Layer order: heavy/stable layers first (apt, node, sage, extras),
# CLI install LAST. Since claude-code and codex bump every few days,
# keeping them on the final layer means a version bump is a ~30s
# rebuild instead of re-running the full conda solve.
#
# User: we rename the existing UID 1000 user (miniforge3 ships one,
# typically `mambauser`) to `node` so sandbox.py's hardcoded
# /home/node/.claude and /home/node/.codex mount targets work
# unchanged across both images.

FROM condaforge/miniforge3:latest

USER root

# ── OS packages: node prereqs + build-essential + git/curl ──────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Node 22 via NodeSource (matches sandbox/Dockerfile).
ARG NODE_MAJOR=22
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

# ── SageMath via conda-forge ────────────────────────────────────
# Install into the base env. mamba (bundled in miniforge3) is
# substantially faster than conda for large solves like sage.
# Clean tarballs + indices after install to cut image size.
RUN mamba install -y -n base -c conda-forge sage \
 && mamba clean -afy

# Extra Python packages Sage doesn't include by default. Versions
# are left unpinned so conda's solver picks the ones compatible
# with sage's constraints on numpy/scipy/etc. — pinning conflicting
# versions here would just produce unsatisfiable solves.
RUN mamba install -y -n base -c conda-forge \
        z3-solver \
        biopython \
        astropy \
        rdkit \
        requests \
        beautifulsoup4 \
        lxml \
        scikit-learn \
        statsmodels \
        normaliz \
        pdfplumber \
        pypdf \
 && mamba clean -afy

# PyNormaliz + ortools: install via pip rather than conda.
#   - PyNormaliz on conda-forge is linux-64 / osx-64 only (no arm64);
#     pip pulls source dist or arm64 wheels and links to the `normaliz`
#     binary installed via conda above.
#   - ortools-python on conda-forge tops out at Python 3.11 (last build
#     was 9.6); Sage pins Python 3.13, so the conda package conflicts.
#     Google publishes ortools wheels on PyPI for Python 3.13 + arm64.
# Falls back gracefully on either failure.
RUN /opt/conda/bin/pip install --no-cache-dir ortools \
    || echo "ortools install failed — boolean MILP runs will fall back to Z3"
RUN /opt/conda/bin/pip install --no-cache-dir PyNormaliz \
    || echo "PyNormaliz install failed — LR strategies will fall back to Sage's lattice point counter"

# ── Runtime user + dirs ─────────────────────────────────────────
# miniforge3 ships with an existing user at UID 1000 (typically
# `mambauser`). Rename it to `node` so sandbox.py's hardcoded
# /home/node/.claude and /home/node/.codex mount targets work
# without per-image branching. If no user exists at UID 1000 yet
# (unlikely for this base image but defensive), create one.
#
# Conda's state lives at /opt/conda, not in the user's home, so
# renaming + moving the home dir doesn't break sage/python/mamba.
#
# miniforge3 already puts /opt/conda/bin first on PATH for all
# users, so `sage`, `python3`, `mamba`, `conda` resolve without
# needing explicit activation.
RUN set -eux; \
    if getent passwd 1000 >/dev/null; then \
        oldname=$(getent passwd 1000 | cut -d: -f1); \
        if [ "$oldname" != "node" ]; then \
            usermod -l node "$oldname"; \
            groupmod -n node "$oldname" 2>/dev/null || true; \
            usermod -d /home/node -m node; \
        fi; \
    else \
        useradd -m -u 1000 -s /bin/bash node; \
    fi; \
    mkdir -p /workspace /home/node/.claude /home/node/.local; \
    chown -R node:node /workspace /home/node

# ── CLIs (LAST so version bumps don't invalidate sage layers) ───
# claude-code and codex release every few days. Keeping this step
# last means bumping either version only re-runs npm install-g
# (~30s) instead of the 15-minute conda solve above.
ARG CLAUDE_VERSION=2.1.150
ARG CODEX_VERSION=0.133.0
RUN npm install -g --no-audit --no-fund \
        @anthropic-ai/claude-code@${CLAUDE_VERSION} \
        @openai/codex@${CODEX_VERSION}

USER node
WORKDIR /workspace
