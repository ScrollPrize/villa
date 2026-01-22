# Optimized Ink Detection Inference

GPU-accelerated, containerized inference for TimeSformer-based ink detection. The container downloads model weights from Hugging Face, fetches input layers from S3, runs inference, uploads a PNG prediction to S3, and writes the output URI to `/tmp/result_s3_url.txt` for workflow orchestration (e.g., Argo).

## What's inside

- **Hugging Face as a model registy** via `MODEL` key
- **S3 I/O** for inputs and outputs

## Inputs and outputs

### Required environment variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL` | Model key or HF repo id. If key, it maps via internal lookup to a repo. | `timesformer-scroll5`
| `S3_PATH` | S3 URI to a segment prefix that contains a `layers/` directory with numbered TIFs. | `s3://my-bucket/scrolls/scroll_001/20231201120000/`
| `START_LAYER` | Inclusive start index of layers to use. | `0`
| `END_LAYER` | Exclusive end index of layers to use. | `26`

### Optional environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` |
| `HF_TOKEN` | Hugging Face token (if model is in private repo) | - |
| `TILE_SIZE` | Tile size for sliding window inference (pixels, also sets network input size) | `64` |
| `STRIDE` | Stride for sliding window (pixels) | `16` |
| `BATCH_SIZE` | Batch size for inference | `256` |
| `MODEL_TYPE` | Model architecture: `timesformer` or `resnet3d-50` | `timesformer` |
| `STEP` | Execution step: `prepare`, `inference`, or `reduce` | `inference` |
| `NUM_PARTS` | Number of partitions for distributed inference | `1` |
| `PART_ID` | Partition ID (0-indexed) when NUM_PARTS > 1 | `0` |
| `SURFACE_VOLUME_ZARR` | Path to pre-created surface volume zarr | - |
| `ZARR_OUTPUT_DIR` | Directory for partition outputs | `/tmp/partitions` |
| `USE_ZARR_COMPRESSION` | Enable zarr compression | `false` |
| `COMPILE` | Enable torch.compile | `1` |
| `COMPILE_MODE` | torch.compile mode | `reduce-overhead` |

**Inference Configuration Notes:**
- `TILE_SIZE`: Sets both the tile extraction size and network input size. Larger values = more context but more memory. Should match training size for best results (typically 64)
- `STRIDE`: Controls overlap between tiles. Smaller stride = more overlap = smoother blending but slower inference
- `BATCH_SIZE`: Number of tiles to process in parallel. Larger values = faster but more GPU memory. Reduce if you encounter OOM errors

### S3 layout (expected)

```
s3://<bucket>/<prefix>/
├── layers/
│   ├── 00.tif
│   ├── 01.tif
│   └── ...
└── (optional extra files)
```

### Output

- Prediction is uploaded to: `s3://<bucket>/<prefix>/predictions/prediction_<MODEL>_<START>_<END>.png`
- The S3 URI is written to: `/tmp/result_s3_url.txt`

## Quick start (local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set env and run:
   ```bash
   export MODEL=timesformer-scroll5
   export S3_PATH=<s3_fragment_path>
   export START_LAYER=0
   export END_LAYER=26

   python entrypoint.py
   ```

## Docker

The `Dockerfile` builds a slim runtime with a prebuilt virtualenv layer.

### Build

```bash
docker build -t ink-detection-optimized-inference .
```

Notes:
- Base image can be overridden via build-arg `BASE_IMAGE` (defaults to `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`).

### Run

```bash
docker run --rm --gpus all \
  -e MODEL=timesformer-scroll5 \
  -e S3_PATH=<s3_fragment_path> \
  -e START_LAYER=0 \
  -e END_LAYER=26 \
  -e AWS_REGION=us-east-1 \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e AWS_ROLE_ARN=<arn_role> \ 
  ink-detection-optimized-inference
```

Authentication options:
- In AWS/Kubernetes, prefer IAM roles (IRSA) for S3 access.
- For local runs, standard AWS env vars/credentials are supported by `boto3`.

## Argo Workflows

Below is a minimal example to run the container as a GPU-enabled Argo Workflow and expose the result S3 URI as an output parameter read from `/tmp/result_s3_url.txt`.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: ink-detection
spec:
  archiveLogs: true
  entrypoint: ink-detection-entrypoint

  artifactRepositoryRef:
    configMap: argo-default-artifact-repository

  podGC:
    strategy: OnPodSuccess
    deleteDelayDuration: 30s

  arguments:
    parameters:
      - name: model
        value: "timesformer-scroll5"
        description: "Predefined model to use. Allowed: timesformer_scroll5_27112024"
      - name: s3-path
        value: "s3://bucket/path/to/input"
        description: "S3 path to input data"
      - name: start-layer
        value: "0"
        description: "Start layer index (inclusive)"
      - name: end-layer
        value: "26"
        description: "End layer index (inclusive)"

  nodeSelector:
    kubernetes.io/arch: amd64

  retryStrategy:
    limit: 3
    retryPolicy: "Always"
    backoff:
      duration: "10s"
      factor: 2
      maxDuration: "1h"

  templates:
    - name: ink-detection-entrypoint
      steps:
        - - name: validate-model
            template: validate-model
        - - name: run-ink-detection
            template: run-ink-detection
            arguments:
              parameters:
                - name: model
                  value: "{{workflow.parameters.model}}"
                - name: s3-path
                  value: "{{workflow.parameters.s3-path}}"
                - name: start-layer
                  value: "{{workflow.parameters.start-layer}}"
                - name: end-layer
                  value: "{{workflow.parameters.end-layer}}"
      outputs:
        parameters:
          - name: result-s3-url
            valueFrom:
              parameter: "{{steps.run-ink-detection.outputs.parameters.result_s3_url}}"

    - name: validate-model
      script:
        image: python:3.11
        command: [python]
        resources:
          requests:
            cpu: "100m"
            memory: 128Mi
          limits:
            memory: 128Mi
        source: |
          import sys

          allowed = {"timesformer-scroll5"}
          model = "{{workflow.parameters.model}}"

          if model not in allowed:
              print(f"Invalid model: {model}. Allowed: {sorted(allowed)}", file=sys.stderr)
              sys.exit(1)
          print(f"Model '{model}' validated.")

    - name: run-ink-detection
      inputs:
        parameters:
        - name: model
        - name: s3-path
        - name: start-layer
        - name: end-layer
      nodeSelector:
        karpenter.k8s.aws/instance-generation: "6"
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 20Gi
      container:
        image: <ink_detection_docker_image>
        resources:
          requests:
            cpu: "6"
            memory: 32Gi
            ephemeral-storage: "10Gi"
            nvidia.com/gpu: "1"
          limits:
            memory: 32Gi
            nvidia.com/gpu: "1"
        env:
          - name: PYTHONUNBUFFERED
            value: "1"
          - name: MODEL
            value: "{{inputs.parameters.model}}"
          - name: S3_PATH
            value: "{{inputs.parameters.s3-path}}"
          - name: START_LAYER
            value: "{{inputs.parameters.start-layer}}"
          - name: END_LAYER
            value: "{{inputs.parameters.end-layer}}"
        volumeMounts:
          - name: shm
            mountPath: /dev/shm
      outputs:
        parameters:
          - name: result_s3_url
            valueFrom:
              path: /tmp/result_s3_url.txt
```

Example submit:

```bash
argo submit wf.yaml \
  -p model=timesformer-scroll5 \
  -p s3_path=<s3_fragment_path> \
  -p start_layer=0 \
  -p end_layer=26
```

Tip:
- `END_LAYER` is exclusive. For layers `00.tif` through `25.tif`, use `START_LAYER=0`, `END_LAYER=26`.

## Notes on models

The `MODEL` key maps to a Hugging Face repo via a small lookup table inside the code. You can look at all possible models in [the HF registry](https://huggingface.co/collections/scrollprize/ink-detection-models-678e2a316597a2e02398357c).

## Troubleshooting

- Missing env var → container exits with an error stating which variable is required
- "No layers found" → verify the `S3_PATH` and the `[START_LAYER, END_LAYER)` range, and ensure files are named numerically under `layers/`
- "Channel mismatch" → the number of layers in the range must match the model's expected channels
