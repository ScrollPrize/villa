FROM ubuntu:noble
ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN apt -y install software-properties-common
RUN add-apt-repository -y universe
RUN apt -y update
RUN apt -y upgrade
RUN apt -y full-upgrade

# --- base toolchain & libs (avoiding xtensor-dev here) ---
RUN apt -y install --no-install-recommends \
    build-essential git cmake ninja-build pkg-config \
    qt6-base-dev libboost-system-dev libboost-program-options-dev libceres-dev \
    libopencv-dev libopencv-contrib-dev \
    libxsimd-dev libblosc-dev libspdlog-dev libgsl-dev libsdl2-dev libcurl4-openssl-dev \
    file curl unzip ca-certificates bzip2 wget fuse jq gimp desktop-file-utils \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    flex bison zlib1g-dev gfortran libopenblas-dev liblapack-dev libscotch-dev libhwloc-dev \
 && rm -rf /var/lib/apt/lists/*

# --- pin specific xtl + xtensor versions from .deb files ---
RUN set -eux; \
    cd /tmp; \
    wget -q http://archive.ubuntu.com/ubuntu/pool/universe/x/xtl/xtl-dev_0.7.7-1_all.deb; \
    wget -q http://archive.ubuntu.com/ubuntu/pool/universe/x/xtensor/libxtensor-dev_0.25.0-2ubuntu1_all.deb; \
    apt-get update; \
    # use apt to install local .debs so dependencies are resolved automatically
    apt-get install -y --no-install-recommends ./xtl-dev_0.7.7-1_all.deb ./libxtensor-dev_0.25.0-2ubuntu1_all.deb; \
    rm -f /tmp/xtl-dev_0.7.7-1_all.deb /tmp/libxtensor-dev_0.25.0-2ubuntu1_all.deb; \
    # prevent later upgrades from bumping these version
    apt-mark hold xtl-dev libxtensor-dev

# ----- Python 3.10 env (micromamba) -----
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        MICROMAMBA_ARCH="linux-64"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        MICROMAMBA_ARCH="linux-aarch64"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    curl -Ls "https://micro.mamba.pm/api/micromamba/${MICROMAMBA_ARCH}/latest" | \
    tar -xvj -C /usr/local/bin bin/micromamba --strip-components=1

RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
SHELL ["/bin/bash", "-lc"]
RUN micromamba create -y -n py310 -c conda-forge python=3.10 pip
RUN micromamba run -n py310 python -m pip install --upgrade pip
RUN micromamba run -n py310 pip install --no-cache-dir numpy==1.26.4 pillow tqdm wandb

# Open3D needs GL/GLib bits even for headless usage.
# These packages are not available on arm64 so skip for now
# TODO: install from source?
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        apt -y install libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 libegl1; \
        micromamba run -n py310 pip install --no-cache-dir libigl==2.5.1 open3d==0.18.0; \
    fi

# Make this Python visible to subsequent steps
ENV PATH="/opt/micromamba/envs/py310/bin:${PATH}"

COPY . /src

# --------------------------- Third-party libs -------------------------------
WORKDIR /src/libs
RUN mkdir -p pastix-install scotch-install
    
# --- libigl: clone and pin to latest commit (as of 2025-10-02) ---
# Latest on main: Aug 1, 2025 — ae8f959ea26d7059abad4c698aba8d6b7c3205e8
ARG LIBIGL_COMMIT=ae8f959ea26d7059abad4c698aba8d6b7c3205e8
RUN git clone https://github.com/libigl/libigl.git libigl \
     && cd libigl \
     && git fetch --depth 1 origin ${LIBIGL_COMMIT} \
     && git checkout -q ${LIBIGL_COMMIT} \
     && git submodule update --init --recursive
    
# --- overlay local changes into the cloned libigl ---
# Copies everything inside /src/libs/libigl_changes into /src/libs/libigl
RUN if [ -d /src/libs/libigl_changes ]; then \
          cp -a /src/libs/libigl_changes/. /src/libs/libigl/; \
        fi
    
# --- pastix & scotch from archives already in repo ---
RUN tar -xjf /src/libs/pastix_5.2.3.tar.bz2 -C pastix-install --strip-components=1
RUN tar -xzf /src/libs/scotch_6.0.4.tar.gz -C scotch-install --strip-components=1
    
WORKDIR /src/libs/scotch-install/src
RUN cp ./Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc
RUN mkdir -p /usr/local/scotch
RUN make scotch
# Create prefixes we will actually keep (not deleted later)
RUN make prefix=/usr/local/scotch install
    
WORKDIR /src/libs/pastix-install/src
RUN cp /src/libs/config.in config.in
RUN make SCOTCH_HOME=/usr/local/scotch
RUN make SCOTCH_HOME=/usr/local/scotch install
    
    # --- build libigl-based target (after overlay) ---
WORKDIR /src/libs/flatboi/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DLIBIGL_WITH_PASTIX=ON \
      -DBLA_VENDOR=OpenBLAS \
      -DCMAKE_PREFIX_PATH=/usr/local/pastix \
      -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_CXX_FLAGS="-DSLIM_CACHED"
RUN cmake --build . -j"$(nproc)"
    
# Install the flatboi binary into PATH before removing /src
RUN install -m 0755 ./flatboi /usr/local/bin/flatboi

# ------------------------- Build the main project ---------------------------
RUN mkdir -p /src/build
WORKDIR /src/build
RUN cmake -DVC_WITH_CUDA_SPARSE=off \
          -DCPACK_DEBIAN_PACKAGE_SHLIBDEPS=ON \
          -DCPACK_DEBIAN_PACKAGE_DEPENDS="" \
          -GNinja /src \
 && ninja \
 && cpack -G DEB -V \
 && dpkg -i /src/build/pkgs/vc3d*.deb

# --------------------------- Cleanup build tree ------------------------------
RUN apt -y autoremove && rm -rf /src

# Wrapper: forwards all args to VC3D with nice/ionice and per-call OMP envs
RUN install -m 0755 /dev/stdin /usr/local/bin/vc3d <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
exec env OMP_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE OMP_NESTED=FALSE nice ionice VC3D "$@"
BASH

COPY docker_s3_entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV WANDB_ENTITY="vesuvius-challenge"
