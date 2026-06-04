# Sandbox image for running claude -p and codex exec per problem.
#
# One container per problem (SWE-bench style). All LLM calls for that
# problem run inside via `docker exec`. Common math/science packages
# are baked in so no per-call pip install cost.
#
# User: runs as `node` (UID 1000, standard in the node:* base image).
# Claude's --dangerously-skip-permissions refuses to run as root by
# design, and Anthropic's own devcontainer uses the node user too.
#
# Auth: host bind-mounts ~/.codex and a Keychain-extracted claude
# credentials file into /home/node/. Subscription tokens, no API keys.
#
# Layer order: heavy/stable layers first (apt, venv, pip), CLI install
# LAST. claude-code and codex release every few days; keeping them on
# the final layer means a version bump is a ~30s rebuild instead of
# re-running every pip install above.

# Base: Debian 13 (trixie, current stable as of 2025-08). glibc 2.41
# and Python 3.13 mean modern manylinux wheels (e.g. z3-solver's
# manylinux_2_38_aarch64) install directly rather than falling back
# to slow source compilation.
FROM node:22-trixie-slim

# ── OS packages ─────────────────────────────────────────────────
# python runtime + venv + dev headers (for pip installs with native
# deps), pari-gp (number-theory CLI, useful on HLE math problems),
# usual tooling, build-essential so agents can compile arbitrary
# native extensions at runtime via `pip install --user`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        python3-dev \
        pari-gp \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Shared Python venv ──────────────────────────────────────────
# PEP 668 in Debian 13 blocks system-level pip. A venv at /opt/venv
# sidesteps this and keeps package locations predictable. We chown
# the venv to the runtime user (node) so agents can add packages at
# runtime with a plain `pip install <pkg>` — useful for problems
# needing a package we didn't bake in. Since all agents in a
# container share one UID, installs by one agent are visible to all
# subsequent agents (solver → formalizer → judges).
#
# Note: `pip install --user` is blocked by venv by design; in-venv
# install is the equivalent. `pip install --target=<dir>` still
# works for per-call-dir isolation if needed.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Pinned latest-stable versions as of 2026-04-16. Rebuild to update.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        sympy==1.14.0 \
        numpy==2.4.4 \
        scipy==1.17.1 \
        pandas==3.0.2 \
        networkx==3.6.1 \
        z3-solver==4.16.0.0 \
        requests==2.33.1 \
        beautifulsoup4==4.14.3 \
        lxml==6.0.4 \
        scikit-learn==1.8.0 \
        statsmodels==0.14.6 \
        matplotlib==3.10.8 \
        biopython==1.87 \
        astropy==7.2.0 \
        rdkit==2026.3.1
# Notes on what's NOT baked in and why:
#   - mpmath: transitive dep of sympy, pinned to <1.4 by sympy 1.14.0.
#     Sympy picks a compatible version automatically.
#   - Pillow/PIL: HLE image problems are filtered out by the loader.
#   - sagemath: NOT in Debian 13 Trixie's apt repo (dropped). The
#     pragmatic alternatives are: (a) conda-forge sagemath in a
#     variant image, or (b) a multi-stage build from
#     sagemath/sagemath. For the HLE benchmark, sympy + pari/gp
#     cover ~90% of math problem computations. Problems that
#     specifically require SageMath's advanced algebraic geometry,
#     specialized group theory, or full number-field support will
#     not complete optimally — document this limitation in
#     run_summary.md. Build sagemath variant only when needed.
#   - qiskit, torch, tensorflow, jax: huge; add to a variant image
#     if problem mix shifts toward quantum / ML-training problems.

# ── Runtime user + directories ──────────────────────────────────
# /workspace is node-owned so agents can write there. We pre-create
# /home/node/.claude so the credential bind-mount has a proper
# parent. The venv is chowned so agents can install packages into it
# at runtime (see note above).
RUN mkdir -p /workspace /home/node/.claude /home/node/.local \
 && chown -R node:node /workspace /home/node/.claude /home/node/.local ${VIRTUAL_ENV}

# ── CLIs (LAST so version bumps don't invalidate pip layers) ────
# claude-code and codex release every few days. Keeping this step
# last means bumping either version only re-runs npm install-g
# instead of every pip install above.
ARG CLAUDE_VERSION=2.1.150
ARG CODEX_VERSION=0.133.0
RUN npm install -g --no-audit --no-fund \
        @anthropic-ai/claude-code@${CLAUDE_VERSION} \
        @openai/codex@${CODEX_VERSION}

USER node
WORKDIR /workspace
