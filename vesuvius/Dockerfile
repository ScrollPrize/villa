FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /workspace/villa/vesuvius

# Install vesuvius core deps (without torch extras, since the container already has PyTorch)
COPY pyproject.toml README.md MANIFEST.in ./
COPY src/vesuvius/_version.py src/vesuvius/_version.py
COPY src/vesuvius/install src/vesuvius/install
RUN pip install --no-cache-dir -e "." \
    && pip install --no-cache-dir -e ".[render]"

# At runtime, mount the full villa tree over /workspace/villa:
#   docker run --gpus all -v /home/sean/villa:/workspace/villa ...
