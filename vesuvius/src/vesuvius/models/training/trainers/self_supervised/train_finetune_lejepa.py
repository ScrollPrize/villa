"""
Fine-tune a supervised model from a pretrained LeJEPA checkpoint.

Loads encoder weights from a LeJEPA checkpoint (discarding the projection head)
and trains on downstream tasks like segmentation.

Features:
- Selective weight loading (encoder only, projector discarded)
- Optional encoder freezing for initial epochs
- Optional differential learning rates (lower LR for pretrained encoder)
- Warmup scheduler support
"""

import os
from pathlib import Path
import torch
from torch._dynamo import OptimizedModule

from vesuvius.models.training.train import BaseTrainer
from vesuvius.models.training.lr_schedulers import WarmupPolyLRScheduler, PolyLRScheduler


class TrainFineTuneLEJEPA(BaseTrainer):
    """
    Fine-tune a supervised model from a pretrained LeJEPA checkpoint.

    Config options:
    - pretrained_lejepa_checkpoint: Path to LeJEPA checkpoint file
    - freeze_encoder_epochs: Number of epochs to freeze encoder (default: 0)
    - encoder_lr_mult: Learning rate multiplier for encoder params (default: 1.0)
    - finetune_warmup_epochs: Warmup epochs before main training (default: 0)
    - load_decoder_from_pretrain: Whether to load decoder weights if present (default: False)
    """

    def __init__(self, mgr=None, verbose: bool = True):
        super().__init__(mgr, verbose)

        # Get checkpoint path from config
        self.pretrained_ckpt = getattr(self.mgr, 'pretrained_lejepa_checkpoint', None)
        if self.pretrained_ckpt is None:
            self.pretrained_ckpt = getattr(self.mgr, 'finetune_from', None)

        # Fine-tuning options
        self.freeze_encoder_epochs = int(getattr(self.mgr, 'freeze_encoder_epochs', 0))
        self.encoder_lr_mult = float(getattr(self.mgr, 'encoder_lr_mult', 1.0))
        self.finetune_warmup_epochs = int(getattr(self.mgr, 'finetune_warmup_epochs', 0))
        self.load_decoder_from_pretrain = bool(getattr(self.mgr, 'load_decoder_from_pretrain', False))

        # Internal state
        self._encoder_frozen = False

        # Disable deep supervision for fine-tuning (simpler setup)
        mgr.enable_deep_supervision = False

    def _get_scheduler(self, optimizer):
        """Get learning rate scheduler with optional warmup."""
        user_specified = hasattr(self.mgr, 'scheduler') and self.mgr.scheduler is not None

        if self.finetune_warmup_epochs > 0 and (not user_specified or str(self.mgr.scheduler).lower() in ("poly", "warmup_poly")):
            print(f"Using Warmup+Poly scheduler for finetuning (warmup_epochs={self.finetune_warmup_epochs})")
            sched = WarmupPolyLRScheduler(
                optimizer=optimizer,
                initial_lr=self.mgr.initial_lr,
                max_steps=self.mgr.max_epoch,
                warmup_steps=self.finetune_warmup_epochs,
                exponent=getattr(self.mgr, 'scheduler_kwargs', {}).get('exponent', 0.9)
            )
            return sched, False

        # Default: plain PolyLR (no warmup)
        sched = PolyLRScheduler(
            optimizer=optimizer,
            initial_lr=self.mgr.initial_lr,
            max_steps=self.mgr.max_epoch,
            exponent=getattr(self.mgr, 'scheduler_kwargs', {}).get('exponent', 0.9)
        )
        print("Using PolyLR scheduler for finetuning (no warmup)")
        return sched, False

    def _update_scheduler_for_epoch(self, scheduler, optimizer, epoch):
        """Handle encoder freezing/unfreezing at epoch boundaries."""
        if self.freeze_encoder_epochs > 0:
            if epoch < self.freeze_encoder_epochs and not self._encoder_frozen:
                self._set_encoder_requires_grad(False)
                self._encoder_frozen = True
                print(f"[Epoch {epoch}] Encoder frozen for finetuning (freeze_encoder_epochs={self.freeze_encoder_epochs})")
            elif epoch >= self.freeze_encoder_epochs and self._encoder_frozen:
                self._set_encoder_requires_grad(True)
                self._encoder_frozen = False
                print(f"[Epoch {epoch}] Encoder unfrozen; training full network")

        return scheduler, getattr(self, '_is_per_iteration_scheduler', False)

    def _set_encoder_requires_grad(self, requires: bool):
        """Enable/disable gradients for encoder parameters."""
        model = getattr(self, '_compiled_model', None)
        if model is None:
            return

        mod = model
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        enc = getattr(mod, 'shared_encoder', None)
        if enc is not None:
            for p in enc.parameters():
                p.requires_grad = requires

    def _get_optimizer(self, model):
        """Create optimizer with optional differential learning rates."""
        # If no LR multiplier requested, use default behavior
        if abs(self.encoder_lr_mult - 1.0) < 1e-8:
            return super()._get_optimizer(model)

        print(f"Using encoder LR multiplier: {self.encoder_lr_mult}")

        # Build param groups: encoder vs the rest
        mod = model
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        enc = getattr(mod, 'shared_encoder', None)
        enc_params = list(enc.parameters()) if enc is not None else []
        enc_param_ids = set(id(p) for p in enc_params)
        other_params = [p for p in mod.parameters() if id(p) not in enc_param_ids]

        from vesuvius.models.training.optimizers import optim, create_optimizer

        # Build optimizer with per-group learning rates
        if str(self.mgr.optimizer).lower() == 'sgd':
            momentum = getattr(self.mgr, 'momentum', 0.99)
            nesterov = getattr(self.mgr, 'nesterov', True)
            groups = []
            if enc_params:
                groups.append({'params': enc_params, 'lr': self.mgr.initial_lr * self.encoder_lr_mult})
            if other_params:
                groups.append({'params': other_params, 'lr': self.mgr.initial_lr})
            return optim.SGD(groups, lr=self.mgr.initial_lr, momentum=momentum, nesterov=nesterov, weight_decay=self.mgr.weight_decay)
        elif str(self.mgr.optimizer).lower() in ('adam', 'adamw'):
            groups = []
            if enc_params:
                groups.append({'params': enc_params, 'lr': self.mgr.initial_lr * self.encoder_lr_mult})
            if other_params:
                groups.append({'params': other_params, 'lr': self.mgr.initial_lr})
            return optim.AdamW(groups, lr=self.mgr.initial_lr, weight_decay=self.mgr.weight_decay)
        else:
            # Fallback to default creator
            optimizer_config = {
                'name': self.mgr.optimizer,
                'learning_rate': self.mgr.initial_lr,
                'weight_decay': self.mgr.weight_decay
            }
            return create_optimizer(optimizer_config, model)

    def _initialize_training(self):
        """Initialize training and load pretrained LeJEPA weights."""
        state = super()._initialize_training()

        # Keep handle to compiled model for freeze helper
        self._compiled_model = state['model']

        if self.pretrained_ckpt is None:
            print("No pretrained LeJEPA checkpoint specified; training from scratch")
            return state

        ckpt_path = Path(self.pretrained_ckpt)
        if not ckpt_path.exists():
            print(f"Warning: Pretrained LeJEPA checkpoint not found: {ckpt_path}")
            return state

        try:
            loaded = torch.load(str(ckpt_path), map_location=self.device)
        except Exception as e:
            print(f"Warning: Failed to load pretrained checkpoint: {e}")
            return state

        # Resolve model state dict from common layouts
        model_state = None
        if isinstance(loaded, dict):
            if 'model' in loaded and isinstance(loaded['model'], dict):
                model_state = loaded['model']
            elif 'state_dict' in loaded and isinstance(loaded['state_dict'], dict):
                model_state = loaded['state_dict']
            else:
                # Heuristic: if all values are tensors, treat as state_dict directly
                try:
                    if all(hasattr(v, 'shape') for v in loaded.values()):
                        model_state = loaded
                except Exception:
                    pass
        elif hasattr(loaded, 'keys'):
            model_state = loaded

        if model_state is None:
            print("Warning: Pretrained checkpoint does not contain a recognizable state_dict; skipping.")
            return state

        # Strip common wrapper prefixes (DDP 'module.', torch.compile '._orig_mod.')
        def _strip_prefixes(sd):
            prefixes = ('module.', '_orig_mod.')
            def strip_key(k: str) -> str:
                changed = True
                while changed:
                    changed = False
                    for p in prefixes:
                        if k.startswith(p):
                            k = k[len(p):]
                            changed = True
                return k
            return {strip_key(k): v for k, v in sd.items()}

        pre_sd = _strip_prefixes(model_state)

        model = state['model']
        mod = model
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        model_sd = mod.state_dict()

        load_keys = []
        skipped_shape = []
        skipped_projector = []

        # LeJEPA checkpoint structure:
        # - encoder.* -> map to shared_encoder.*
        # - projector.* -> skip (not needed for downstream)

        for k, v in pre_sd.items():
            # Skip projector weights (used only for self-supervised training)
            if k.startswith('projector.'):
                skipped_projector.append(k)
                continue

            # Map encoder.* to shared_encoder.*
            if k.startswith('encoder.'):
                new_key = 'shared_encoder.' + k[len('encoder.'):]
                if new_key in model_sd and model_sd[new_key].shape == v.shape:
                    model_sd[new_key] = v
                    load_keys.append(new_key)
                else:
                    skipped_shape.append(k)
            # Also try shared_encoder.* directly (if checkpoint already uses this naming)
            elif k.startswith('shared_encoder.'):
                if k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    load_keys.append(k)
                else:
                    skipped_shape.append(k)
            # Optionally load decoder weights
            elif self.load_decoder_from_pretrain and k.startswith('shared_decoder.'):
                if k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    load_keys.append(k)
                else:
                    skipped_shape.append(k)

        # Load the modified state dict
        mod.load_state_dict(model_sd, strict=False)

        # Report loading results
        if len(load_keys) == 0:
            try:
                ckpt_keys = list(pre_sd.keys())
                tgt_keys = list(model_sd.keys())
                print("Loaded 0 layers from LeJEPA checkpoint. Key mismatch likely. Samples:")
                print(f"  ckpt key sample: {ckpt_keys[:3]} ... {ckpt_keys[-3:]}")
                print(f"  model key sample: {tgt_keys[:3]} ... {tgt_keys[-3:]}")
            except Exception:
                pass
        else:
            print(f"Loaded {len(load_keys)} layers from LeJEPA checkpoint into current model.")

        if skipped_projector:
            print(f"Skipped {len(skipped_projector)} projector params from pretrain (by design).")

        if skipped_shape:
            print(f"Skipped {len(skipped_shape)} params due to shape mismatch.")

        return state


if __name__ == '__main__':
    trainer = TrainFineTuneLEJEPA()
    trainer.run()
