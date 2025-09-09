import os
from pathlib import Path
import torch
from torch._dynamo import OptimizedModule

from vesuvius.models.training.train import BaseTrainer
from vesuvius.models.training.lr_schedulers import WarmupPolyLRScheduler, PolyLRScheduler


class TrainFineTuneMAEUNet(BaseTrainer):
    """
    Fine-tune a supervised UNet from a pretrained UNet-MAE checkpoint produced by TrainUNetMAE.

    - Loads overlapping weights from MAE checkpoint into the current model (encoder always, decoder optional).
    - Skips task-specific heads from the MAE checkpoint.
    - Defaults to no LR warmup, but supports an optional warmup
    - Respects existing BaseTrainer behavior for datasets, losses, logging, etc.
    """

    def __init__(self, mgr=None, verbose: bool = True):
        super().__init__(mgr, verbose)

        self.pretrained_ckpt = getattr(self.mgr, 'pretrained_mae_checkpoint', None)
        if self.pretrained_ckpt is None:
            self.pretrained_ckpt = getattr(self.mgr, 'finetune_from', None)

        self.load_decoder_from_pretrain = bool(getattr(self.mgr, 'load_decoder_from_pretrain', True))
        self.freeze_encoder_epochs = int(getattr(self.mgr, 'freeze_encoder_epochs', 0))
        self.encoder_lr_mult = float(getattr(self.mgr, 'encoder_lr_mult', 1.0))

        # Warmup option (default disabled). Use epochs as steps since BaseTrainer steps per epoch for Poly.
        self.finetune_warmup_epochs = int(getattr(self.mgr, 'finetune_warmup_epochs', 0))

        # Internal flag to avoid re-freezing every call
        self._encoder_frozen = False

        mgr.enable_deep_supervision = False

        # --- scheduler with optional warmup --- #
    def _get_scheduler(self, optimizer):
        # If user explicitly configured a scheduler via config/CLI, defer to BaseTrainer
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

    # --- epoch hook to optionally freeze encoder at the start --- #
    def _update_scheduler_for_epoch(self, scheduler, optimizer, epoch):
        if self.freeze_encoder_epochs > 0:
            if epoch < self.freeze_encoder_epochs and not self._encoder_frozen:
                self._set_encoder_requires_grad(False)
                self._encoder_frozen = True
                print(f"[Epoch {epoch}] Encoder frozen for finetuning (freeze_encoder_epochs={self.freeze_encoder_epochs})")
            elif epoch >= self.freeze_encoder_epochs and self._encoder_frozen:
                self._set_encoder_requires_grad(True)
                self._encoder_frozen = False
                print(f"[Epoch {epoch}] Encoder unfrozen; training full network")

        # No scheduler switching needed; just return current scheduler
        return scheduler, getattr(self, '_is_per_iteration_scheduler', False)

    def _set_encoder_requires_grad(self, requires: bool):
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

    # --- optimizer with optional encoder LR multiplier --- #
    # the purpose of this is if you'd like to set the LR lower on the pretrained encoder
    # while keeping your decoder LR higher
    def _get_optimizer(self, model):
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

        optimizer_config = {
            'name': self.mgr.optimizer,
            'learning_rate': self.mgr.initial_lr,
            'weight_decay': self.mgr.weight_decay
        }

        from vesuvius.models.training.optimizers import optim
        # Build SGD manually to set per-group lr if using SGD; else fall back to default creator
        if str(self.mgr.optimizer).lower() == 'sgd':
            momentum = getattr(self.mgr, 'momentum', 0.99)
            nesterov = getattr(self.mgr, 'nesterov', True)
            groups = []
            if enc_params:
                groups.append({'params': enc_params, 'lr': self.mgr.initial_lr * self.encoder_lr_mult})
            if other_params:
                groups.append({'params': other_params, 'lr': self.mgr.initial_lr})
            return optim.SGD(groups, lr=self.mgr.initial_lr, momentum=momentum, nesterov=nesterov, weight_decay=self.mgr.weight_decay)
        else:
            # For other optimizers, use the default creator (single group), user can adjust lr instead
            from vesuvius.models.training.optimizers import create_optimizer
            return create_optimizer(optimizer_config, model)

    # --- initialize training: load MAE weights after model creation --- #
    def _initialize_training(self):
        state = super()._initialize_training()

        # Keep a handle to compiled model for freeze helper
        self._compiled_model = state['model']

        if self.pretrained_ckpt is None:
            return state

        ckpt_path = Path(self.pretrained_ckpt)
        if not ckpt_path.exists():
            print(f"Warning: Pretrained MAE checkpoint not found: {ckpt_path}")
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
        skipped_head = []

        for k, v in pre_sd.items():
            if k.startswith('task_heads.') or k.startswith('task_decoders.'):
                skipped_head.append(k)
                continue
            if k.startswith('shared_encoder.'):
                if k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    load_keys.append(k)
                else:
                    skipped_shape.append(k)
            elif self.load_decoder_from_pretrain and k.startswith('shared_decoder.'):
                if k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    load_keys.append(k)
                else:
                    skipped_shape.append(k)
            else:
                # Unknown or unsupported key for finetuning
                pass

        # If nothing loaded, try relaxed key remapping:
        #  - remove leading 'module.' once more (in case upstream mapping failed)
        #  - map 'encoder.'/'decoder.' to 'shared_encoder.'/'shared_decoder.'
        if len(load_keys) == 0:
            alt_loaded = []
            for k, v in list(pre_sd.items()):
                new_k = None
                if k.startswith('module.'):
                    k = k[len('module.') :]
                if k.startswith('encoder.'):
                    new_k = 'shared_encoder.' + k[len('encoder.') :]
                elif self.load_decoder_from_pretrain and k.startswith('decoder.'):
                    new_k = 'shared_decoder.' + k[len('decoder.') :]
                if new_k is not None and new_k in model_sd and model_sd[new_k].shape == v.shape:
                    model_sd[new_k] = v
                    alt_loaded.append(new_k)
            if alt_loaded:
                load_keys.extend(alt_loaded)

        missing_before = [k for k in mod.state_dict().keys() if k not in model_sd]
        mod.load_state_dict(model_sd, strict=False)

        if len(load_keys) == 0:
            # Help debug by showing a few keys
            try:
                ch_keys = list(pre_sd.keys())
                tg_keys = list(model_sd.keys())
                print("Loaded 0 layers from MAE checkpoint. Key mismatch likely. Samples:")
                print(f"  ckpt key sample: {ch_keys[:3]} ... {ch_keys[-3:]}")
                print(f"  model key sample: {tg_keys[:3]} ... {tg_keys[-3:]}")
            except Exception:
                pass
        else:
            print(f"Loaded {len(load_keys)} layers from MAE checkpoint into current model.")
        if skipped_head:
            print(f"Skipped {len(skipped_head)} head/decoder params from pretrain (by design).")
        if skipped_shape:
            print(f"Skipped {len(skipped_shape)} params due to shape mismatch.")

        return state
