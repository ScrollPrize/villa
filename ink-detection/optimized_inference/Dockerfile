# Build separate images for different steps requiring GPU and CPU-only

FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime AS gpu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    libblosc-dev \
    libcurl4-openssl-dev \
    liblz4-dev \
    libzstd-dev \
    libsqlite3-dev \
    libtiff-dev \
    libpng-dev \
    libjpeg-dev \
    zlib1g-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# vc_reader (zarr/image I/O, object stores) is a C++ package built from this monorepo, not
# PyPI; build.sh passes its source as the `vc_reader` build context. Built without CUDA:
# inference only uses it for I/O, torch does the GPU work.
COPY --from=vc_reader pyproject.toml CMakeLists.txt README.md /opt/vc_reader/
COPY --from=vc_reader cmake /opt/vc_reader/cmake
COPY --from=vc_reader include /opt/vc_reader/include
COPY --from=vc_reader src /opt/vc_reader/src
COPY --from=vc_reader third_party /opt/vc_reader/third_party
RUN CMAKE_ARGS="-DVC_READER_ENABLE_CUDA=OFF" uv pip install --system --break-system-packages --no-cache /opt/vc_reader \
    && rm -rf /opt/vc_reader

COPY requirements.txt /app
# should only install additional dependencies, not torch, cuda, etc.
# --break-system-packages: the 2.10 base image uses system Python with PEP 668 marker
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt

COPY entrypoint.py /app
COPY processing.py /app
COPY inference.py /app
COPY profiling.py /app
COPY model_timesformer.py /app
COPY model_resnet3d.py /app
COPY model_resnet3d_3d_decoder.py /app
COPY k8s.py /app
COPY runtime_contracts.py /app
COPY models/ /app/models/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV IMAGE_ROLE=gpu

ENTRYPOINT ["python", "/app/entrypoint.py"]

FROM python:3.11-slim AS cpu
ENV DEBIAN_FRONTEND=noninteractive
# build-essential: some deps (e.g. stringzilla, pulled in transitively by
# albumentations -> albucore) ship no prebuilt aarch64 wheel and must compile
# from source, which needs a C/C++ toolchain. The gpu stage above already
# includes it for the same reason.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    libblosc-dev \
    libcurl4-openssl-dev \
    liblz4-dev \
    libzstd-dev \
    libsqlite3-dev \
    libtiff-dev \
    libpng-dev \
    libjpeg-dev \
    zlib1g-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# vc_reader from the monorepo, as in the gpu stage.
COPY --from=vc_reader pyproject.toml CMakeLists.txt README.md /opt/vc_reader/
COPY --from=vc_reader cmake /opt/vc_reader/cmake
COPY --from=vc_reader include /opt/vc_reader/include
COPY --from=vc_reader src /opt/vc_reader/src
COPY --from=vc_reader third_party /opt/vc_reader/third_party
RUN CMAKE_ARGS="-DVC_READER_ENABLE_CUDA=OFF" uv pip install --system --no-cache /opt/vc_reader \
    && rm -rf /opt/vc_reader

COPY requirements-cpu-only.txt /app
RUN uv pip install --system --no-cache -r requirements-cpu-only.txt

COPY entrypoint.py /app
COPY processing.py /app
COPY profiling.py /app
COPY k8s.py /app
COPY runtime_contracts.py /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV IMAGE_ROLE=cpu

ENTRYPOINT ["python", "/app/entrypoint.py"]
