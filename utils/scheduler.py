import torch
import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CosineScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.start_warmup_value = start_warmup_value
        self.freeze_iters = freeze_iters

        # Phase 1 — Freeze: hold lr at zero for the first `freeze_iters` steps
        freeze_schedule = np.zeros(freeze_iters)

        # Phase 2 — Warmup: linearly ramp lr from start_warmup_value to base_value
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        # Phase 3 — Cosine decay: smoothly anneal lr from base_value down to final_value
        # Formula: lr(t) = final_value + 0.5 * (base_value - final_value) * (1 + cos(pi * t / T))
        # At t=0 this equals base_value; at t=T-1 it approaches final_value.
        decay_iters = total_iters - warmup_iters - freeze_iters
        t = np.arange(decay_iters)
        decay_schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * t / decay_iters)
        )

        # Concatenate all three phases into a single lookup table of length total_iters
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, decay_schedule))
        assert len(self.schedule) == self.total_iters

        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch

        # Once training exceeds the scheduled range, pin lr at final_value
        if step >= self.total_iters:
            return [self.final_value for _ in self.optimizer.param_groups]

        # Look up the precomputed lr for the current step
        return [self.schedule[step] for _ in self.optimizer.param_groups]