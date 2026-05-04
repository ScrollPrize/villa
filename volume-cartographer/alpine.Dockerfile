# syntax=docker/dockerfile:1.7

# ============================================================================
# Stage 1: builder — toolchain + dev libraries (musl libc).
# ============================================================================
FROM alpine:3.21 AS builder

RUN apk add --no-cache \
        build-base clang lld git cmake ninja-build ccache pkgconfig \
        qt6-qtbase-dev qt6-qttools-dev qt6-qtbase-private-dev \
        boost-dev \
        ceres-solver-dev cgal-dev \
        opencv-dev \
        blosc-dev spdlog-dev gsl-dev sdl2-dev curl-dev \
        nlohmann-json avahi-dev \
        libde265-dev x265-dev lz4-dev tiff-dev \
        flex bison zlib-dev gfortran \
        openblas-dev lapack-dev \
        scotch-dev hwloc-dev \
        file curl unzip ca-certificates bzip2 wget jq \
        gcovr lcov \
        aws-cli \
        bash

# ============================================================================
# Stage 2: build third-party — scotch, pastix, libigl/flatboi.
# Note: scotch-dev is installed above for headers, but the upstream PaStiX 5.2
# build expects libscotch built from the vendored tarball.
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
# Stage 4: runtime.
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
