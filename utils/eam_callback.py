from timm.utils.model_ema import ModelEmaV2
from pytorch_lightning.callbacks import Callback
from timm.utils.model import get_state_dict, unwrap_model
import torch
import warnings
from typing import Optional, Union, List

class EMACallback(Callback):
    """
    EMA Callback with full original features + multi-GPU support.
    
    Args:
        decay (float): EMA decay factor (default: 0.9999)
        apply_to (str/list): Submodule(s) to apply EMA (None = all trainable)
        use_ema_weights (bool): Whether to use EMA weights at training end
        strict (bool): Enforce all specified modules must exist (default: True)
        verbose (bool): Print warnings when modules are skipped (default: True)
    """
    def __init__(self, 
                 decay: float = 0.9999,
                 apply_to: Optional[Union[str, List[str]]] = 'net_desmoke',
                 use_ema_weights: bool = True,
                 strict: bool = True,
                 verbose: bool = True):
        self.decay = decay
        self.apply_to = [apply_to] if isinstance(apply_to, str) else apply_to
        self.use_ema_weights = use_ema_weights
        self.strict = strict
        self.verbose = verbose
        self.ema = None
        self._backup_params = None

    def on_fit_start(self, trainer, pl_module):
        """Initialize EMA for specified submodules"""
        target = self._get_target_modules(pl_module)
        if target is None:
            if self.strict:
                raise ValueError("No valid submodules found for EMA")
            if self.verbose and trainer.is_global_zero:
                warnings.warn("EMA disabled: no trainable submodules found")
            return
            
        self.ema = ModelEmaV2(target, decay=self.decay)
        if self.verbose and trainer.is_global_zero:
            print(f"EMA initialized for: {list(target.keys())}")

    def _get_target_modules(self, pl_module):
        """Get target modules with validation"""
        if not self.apply_to:  # Apply to all trainable params
            return unwrap_model(pl_module)
            
        targets = {}
        for name in self.apply_to:
            module = getattr(pl_module, name, None)
            if module is None:
                if self.strict:
                    raise AttributeError(f"Submodule '{name}' not found")
                continue
            if any(p.requires_grad for p in module.parameters()):
                targets[name] = module
            elif self.verbose:
                warnings.warn(f"Submodule '{name}' has no trainable parameters")
        return torch.nn.ModuleDict(targets) if targets else None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA weights"""
        if self.ema is not None:
            self.ema.update(self._get_target_modules(pl_module))

    def on_validation_epoch_start(self, trainer, pl_module):
        """Apply EMA weights for validation"""
        if self.ema is None:
            return
            
        # Backup original params
        self._backup_params = {
            n: p.detach().clone() 
            for n, p in pl_module.named_parameters()
            if p.requires_grad
        }
        # Apply EMA state
        ema_state = get_state_dict(self.ema.module)
        current_state = pl_module.state_dict()
        for k in ema_state:
            if k in current_state and current_state[k].shape == ema_state[k].shape:
                current_state[k].copy_(ema_state[k])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Restore original weights"""
        if self._backup_params is None:
            return
            
        current_state = pl_module.state_dict()
        for k, v in self._backup_params.items():
            if k in current_state:
                current_state[k].copy_(v)
        self._backup_params = None

    def on_train_end(self, trainer, pl_module):
        """Optionally apply EMA weights at training end"""
        if self.use_ema_weights and self.ema is not None:
            ema_state = get_state_dict(self.ema.module)
            current_state = pl_module.state_dict()
            for k in ema_state:
                if k in current_state and current_state[k].shape == ema_state[k].shape:
                    current_state[k].copy_(ema_state[k])
            if self.verbose and trainer.is_global_zero:
                print("Model weights replaced with EMA version")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save EMA state to checkpoint"""
        if self.ema is not None:
            checkpoint["state_dict_ema"] = get_state_dict(self.ema.module)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Load EMA state from checkpoint"""
        if "state_dict_ema" in checkpoint:
            if self.ema is None:  # Lazy initialization
                self.on_fit_start(trainer, pl_module)
            if self.ema is not None:
                state_dict = checkpoint["state_dict_ema"]
                # Handle DDP prefix if present
                if any(k.startswith('module.') for k in state_dict):
                    state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}
                self.ema.module.load_state_dict(state_dict)