"""Fixed-decay parameter averages; no extra forwards or optimizer changes."""

from contextlib import contextmanager
import math

import torch


class ModelEMA:
    def __init__(self, model, decays=(), *, start_step=0):
        self.tracks = {}
        for value in decays:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("EMA decays must be numbers in (0, 1)")
            decay = float(value)
            if not math.isfinite(decay) or not 0 < decay < 1:
                raise ValueError("EMA decays must be finite and in (0, 1)")
            name = "ema_" + str(decay).replace(".", "p")
            if name in self.tracks:
                raise ValueError("EMA decays must be unique")
            self.tracks[name] = {
                "decay": decay, "updates": 0, "start_step": int(start_step),
                "best_metric": math.inf,
                "model": {key: tensor.detach().clone().float() if tensor.is_floating_point()
                          else tensor.detach().clone() for key, tensor in model.state_dict().items()},
            }

    @torch.no_grad()
    def update(self, model):
        current = model.state_dict()
        for track in self.tracks.values():
            groups = {}
            for key, shadow in track["model"].items():
                value = current[key].detach()
                if shadow.is_floating_point():
                    values, shadows = groups.setdefault((shadow.device, shadow.dtype), ([], []))
                    values.append(value.to(dtype=shadow.dtype))
                    shadows.append(shadow)
                else:
                    shadow.copy_(value)
            for values, shadows in groups.values():
                torch._foreach_lerp_(shadows, values, 1.0 - track["decay"])
            track["updates"] += 1

    def state_dict(self):
        return {"version": 1, "tracks": self.tracks}

    def load_state_dict(self, state):
        if state.get("version") != 1 or set(state.get("tracks", {})) != set(self.tracks):
            raise ValueError("EMA checkpoint tracks differ from configured decays")
        for name, track in self.tracks.items():
            saved = state["tracks"][name]
            if saved["decay"] != track["decay"] or set(saved["model"]) != set(track["model"]):
                raise ValueError("EMA checkpoint is incompatible with model/config")
            for key, tensor in track["model"].items():
                source = saved["model"][key]
                if source.shape != tensor.shape or source.dtype != tensor.dtype:
                    raise ValueError(f"EMA tensor mismatch: {name}/{key}")
                tensor.copy_(source)
            for key in ("updates", "start_step", "best_metric"):
                track[key] = saved[key]

    @contextmanager
    def apply(self, model, name):
        """Preserve Parameter identity, raw buffers and every module's mode."""
        state = model.state_dict()
        backup = {key: value.detach().clone() for key, value in state.items()}
        modes = [(module, module.training) for module in model.modules()]
        try:
            with torch.no_grad():
                for key, value in state.items():
                    value.copy_(self.tracks[name]["model"][key])
            model.eval()
            yield
        finally:
            with torch.no_grad():
                for key, value in state.items():
                    value.copy_(backup[key])
            for module, training in modes:
                module.training = training
