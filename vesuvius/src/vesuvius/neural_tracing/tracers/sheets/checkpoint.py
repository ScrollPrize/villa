"""Sheet tracer checkpoints: the model (EMA weights when the run kept them), its configuration and the checkpoint."""
import torch

from vesuvius.neural_tracing.tracers.sheets import config as C
from vesuvius.neural_tracing.tracers.sheets.model import build_model, load_model_state


def load_checkpoint(path, device, raw: bool = False):
    """Model (the EMA weights if the run kept them, unless ``raw``), config and the checkpoint dict."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg = C.from_dict(checkpoint['config'], strict=False)
    model = build_model(cfg).to(device)
    load_model_state(model, checkpoint['model'] if raw or not checkpoint.get('ema') else checkpoint['ema'])
    return model, cfg, checkpoint
