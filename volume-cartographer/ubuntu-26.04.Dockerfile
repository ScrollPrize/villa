# syntax=docker/dockerfile:1.7

# ============================================================================
# Stage 1: builder — toolchain + dev libraries.
# CI uses this stage via `docker build --target builder` for a cached
# environment to compile + test in.
# ============================================================================
FROM ubuntu:26.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
        software-properties-common \
 && add-apt-repository -y universe \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
        build-essential clang lld git cmake ninja-build ccache pkg-config \
        qt6-base-dev qt6-base-private-dev \
        libboost-system-dev libboost-program-options-dev \
        libceres-dev libcgal-dev \
        libopencv-dev libopencv-contrib-dev \
        libblosc-dev libspdlog-dev libgsl-dev libsdl2-dev libcurl4-openssl-dev \
        nlohmann-json3-dev libavahi-client-dev \
        libde265-dev libx265-dev liblz4-dev libtiff-dev \
        flex bison zlib1g-dev gfortran libopenblas-dev liblapack-dev \
        libscotch-dev libhwloc-dev \
        libbacktrace-dev \
        file curl unzip ca-certificates bzip2 wget jq \
        gcovr lcov \
 && rm -rf /var/lib/apt/lists/*

RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
      x86_64)  AWS_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" ;; \
      aarch64) AWS_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" ;; \
      *) echo "Unsupported architecture: $ARCH" >&2 && exit 1 ;; \
    esac && \
    curl -fsSL "$AWS_URL" -o awscliv2.zip && \
    unzip -q awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws

# ============================================================================
# Stage 2: build third-party — scotch, pastix, libigl/flatboi.
# ============================================================================
FROM builder AS thirdparty
ARG LIBIGL_COMMIT=ae8f959ea26d7059abad4c698aba8d6b7c3205e8

COPY libs /src/libs

WORKDIR /src/libs
RUN mkdir -p pastix-install scotch-install \
 && tar -xjf /src/libs/pastix_5.2.3.tar.bz2 -C pastix-install --strip-components=1 \
 && tar -xzf /src/libs/scotch_6.0.4.tar.gz -C scotch-install --strip-components=1

WORKDIR /src/libs/scotch-install/src
RUN cp ./Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc \
 && mkdir -p /usr/local/scotch \
 && make -j"$(nproc --all)" scotch \
 && make prefix=/usr/local/scotch install

WORKDIR /src/libs/pastix-install/src
RUN cp /src/libs/config.in config.in \
 && make -j"$(nproc --all)" SCOTCH_HOME=/usr/local/scotch \
 && make SCOTCH_HOME=/usr/local/scotch install

RUN git clone https://github.com/libigl/libigl.git /src/libs/libigl \
 && cd /src/libs/libigl \
 && git fetch --depth 1 origin ${LIBIGL_COMMIT} \
 && git checkout -q ${LIBIGL_COMMIT} \
 && git submodule update --init --recursive \
 && if [ -d /src/libs/libigl_changes ]; then cp -a /src/libs/libigl_changes/. /src/libs/libigl/; fi

WORKDIR /src/libs/flatboi
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBIGL_WITH_PASTIX=ON \
        -DBLA_VENDOR=OpenBLAS \
        -DCMAKE_PREFIX_PATH=/usr/local/pastix \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_CXX_FLAGS="-DSLIM_CACHED" \
 && cmake --build build -j"$(nproc --all)" \
 && install -m 0755 /src/libs/flatboi/build/flatboi /usr/local/bin/flatboi

# ============================================================================
# Stage 3: build vc3d itself.
# ============================================================================
FROM thirdparty AS build
COPY . /src
WORKDIR /src
RUN cmake --preset ci-release-gcc -B /src/build/ci-release-gcc \
 && cmake --build /src/build/ci-release-gcc --parallel "$(nproc --all)" \
 && cp /src/build/ci-release-gcc/bin/* /usr/local/bin/

# ============================================================================
# Stage 4: runtime — the published image.
# ============================================================================
FROM build AS runtime

RUN install -m 0755 /dev/stdin /usr/local/bin/vc3d <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
exec env OMP_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE OMP_NESTED=FALSE nice ionice VC3D "$@"
BASH

COPY docker_s3_entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV WANDB_ENTITY="vesuvius-challenge"
