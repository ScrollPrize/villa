# syntax=docker/dockerfile:1.7

FROM ubuntu:26.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive

# Link the published ghcr image to the repo so repo admins manage it
# (vs. needing org-owner access to package settings).
LABEL org.opencontainers.image.source="https://github.com/ScrollPrize/villa"
LABEL org.opencontainers.image.description="volume-cartographer CI builder (Ubuntu 26.04 resolute)"
LABEL org.opencontainers.image.licenses="GPL-3.0"

# Build deps live in scripts/install_build_deps.sh — the single source of
# truth shared with scripts/ec2_setup.sh. The BuildKit cache mount stays
# here so apt layer caching still applies.
COPY scripts/install_build_deps.sh /tmp/install_build_deps.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    bash /tmp/install_build_deps.sh \
 && rm -f /tmp/install_build_deps.sh

FROM builder AS build
COPY . /src
WORKDIR /src
RUN cmake --preset ci-release-gcc \
 && cmake --build --preset ci-release-gcc \
 && cp build/ci-release-gcc/bin/* /usr/local/bin/

FROM build AS runtime

RUN install -m 0755 /dev/stdin /usr/local/bin/vc3d <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
exec env OMP_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE OMP_NESTED=FALSE nice ionice VC3D "$@"
BASH

COPY docker_s3_entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV WANDB_ENTITY="vesuvius-challenge"
