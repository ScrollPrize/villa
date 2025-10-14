ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The pytorch image has everything preinstalled in the conda environment. If we use that
# one, there's is no need to reinstall everything else
ENV PATH="/opt/conda/bin:$PATH"
WORKDIR /app

COPY requirements.txt /app
# should only install additional dependencies, not torch, cuda, etc.
RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.py /app
COPY inference_timesformer.py /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

ENTRYPOINT ["python", "/app/entrypoint.py"]
