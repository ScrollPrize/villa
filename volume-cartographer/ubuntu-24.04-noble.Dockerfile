FROM ubuntu:noble

RUN apt -y update
RUN apt -y install software-properties-common
RUN add-apt-repository -y universe
RUN apt -y update
RUN apt -y upgrade
RUN apt -y full-upgrade

# --- install everything EXCEPT xtensor-dev ---
RUN apt -y install build-essential git cmake qt6-base-dev libboost-system-dev libboost-program-options-dev libceres-dev \
    libopencv-dev libxsimd-dev libblosc-dev libspdlog-dev libgsl-dev libsdl2-dev libcurl4-openssl-dev file \
    curl unzip ca-certificates bzip2 wget fuse jq gimp desktop-file-utils ninja-build

RUN apt-get install -y flex bison zlib1g-dev gfortran libopenblas-dev liblapack-dev libscotch-dev libhwloc-dev

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
#These packages are not available on arm64 so skip for now
#TODO: install from source?
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        apt -y install libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 libegl1; \
        micromamba run -n py310 pip install --no-cache-dir libigl==2.5.1 open3d==0.18.0; \
    fi

# Make this Python visible to subsequent steps
ENV PATH="/opt/micromamba/envs/py310/bin:${PATH}"

COPY . /src

# ------------------------- Build your main project ---------------------------
RUN mkdir -p /src/build
WORKDIR /src/build
RUN cmake -DVC_WITH_CUDA_SPARSE=off -GNinja /src && ninja && cpack -G DEB -V && dpkg -i /src/build/pkgs/vc3d*.deb

WORKDIR /src/libs
RUN mkdir pastix-install
RUN mkdir scotch-install
RUN mkdir libigl
RUN wget https://dl.ash2txt.org/other/dev/libigl.tar.bz2
RUN tar -xjf /src/libs/pastix_5.2.3.tar.bz2 -C pastix-install --strip-components=1
RUN tar -xjf /src/libs/libigl.tar.bz2 -C libigl --strip-components=1
RUN tar -xzf /src/libs/scotch_6.0.4.tar.gz -C scotch-install --strip-components=1
WORKDIR scotch-install/src
RUN cp ./Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc
RUN mkdir /usr/local/scotch
RUN make scotch
# Create prefixes we will actually keep (not deleted later)
RUN make prefix=/usr/local/scotch install
WORKDIR /src/libs/pastix-install/src
RUN cp /src/libs/config.in config.in
RUN make SCOTCH_HOME=/usr/local/scotch
RUN make SCOTCH_HOME=/usr/local/scotch install

WORKDIR /src/libs/libigl/tutorial/999_Flatboi/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DLIBIGL_WITH_PASTIX=ON \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH=/usr/local/pastix
RUN cmake --build . -j"$(nproc)"

# Install the flatboi binary into PATH before removing /src
RUN install -m 0755 ./flatboi /usr/local/bin/flatboi

WORKDIR /src/build
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

