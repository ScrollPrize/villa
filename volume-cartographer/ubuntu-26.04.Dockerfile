# syntax=docker/dockerfile:1.7

FROM ubuntu:26.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive

# Link the published ghcr image to the repo so repo admins manage it
# (vs. needing org-owner access to package settings).
LABEL org.opencontainers.image.source="https://github.com/ScrollPrize/villa"
LABEL org.opencontainers.image.description="volume-cartographer CI builder (Ubuntu 26.04 resolute)"
LABEL org.opencontainers.image.licenses="GPL-3.0"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update -y \
 && apt-get install -y --no-install-recommends software-properties-common ca-certificates curl unzip \
 && add-apt-repository -y universe \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
        build-essential clang lld libclang-rt-21-dev mold git cmake ninja-build pkg-config \
        qt6-base-dev \
        libboost-system-dev libboost-program-options-dev \
        libceres-dev libsuitesparse-dev \
        libopencv-dev libopencv-contrib-dev \
        libcgal-dev libmpfr-dev libgmp-dev \
        libblosc2-dev libzstd-dev libcurl4-openssl-dev \
        nlohmann-json3-dev libavahi-client-dev \
        liblz4-dev libtiff-dev \
        zlib1g-dev gfortran libopenblas-dev liblapack-dev liblapacke-dev \
        libscotch-dev libhwloc-dev \
        file bzip2 wget jq \
        gcovr lcov \
        python3 python3-venv

RUN arch="$(uname -m)" \
 && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${arch}.zip" -o /tmp/awscli.zip \
 && unzip -q /tmp/awscli.zip -d /tmp \
 && /tmp/aws/install \
 && rm -rf /tmp/awscli.zip /tmp/aws

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
