# syntax=docker/dockerfile:1.7

FROM alpine:3.21 AS builder

RUN --mount=type=cache,target=/var/cache/apk \
    apk add --no-cache \
        bash ca-certificates curl unzip \
        build-base clang lld git cmake ninja-build pkgconfig \
        qt6-qtbase-dev \
        boost-dev \
        ceres-solver-dev suitesparse-dev \
        opencv-dev \
        blosc2-dev curl-dev \
        nlohmann-json avahi-dev \
        lz4-dev tiff-dev \
        zlib-dev gfortran \
        openblas-dev lapack-dev \
        file bzip2 wget jq \
        gcovr lcov \
        aws-cli

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
