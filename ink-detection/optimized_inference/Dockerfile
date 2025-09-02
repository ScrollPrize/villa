ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

FROM ${BASE_IMAGE} as base
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build venv layer
FROM base as build
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
 && . /opt/venv/bin/activate \
 && pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Runtime image
FROM base as runtime
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

COPY --from=build /opt/venv /opt/venv
COPY entrypoint.py /app
COPY inference_timesformer.py /app

# Ensure minimal size
RUN rm -rf /root/.cache /app/__pycache__

ENTRYPOINT ["python", "/app/entrypoint.py"]