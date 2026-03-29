import torch
import torch.nn as nn
from models.vgg import Vgg19, Vgg16
from pytorch_msssim import MS_SSIM, SSIM
import torch.nn.functional as F
import random


class Desmoke_Loss(nn.Module):
    """
    Composite loss dispatcher.
    Only the loss modules listed in `loss_name` are instantiated;
    the rest are left as None to avoid unnecessary memory/compute overhead.
    """
    def __init__(self, loss_name=['l1', 'vgg19_loss', 'ms-ssim_loss', 'cr_loss'],
                 vgg_weights=[1.0/5, 1.0/5, 1.0/5, 1.0/5, 1.0/5],
                 weights=None):
        super().__init__()

        # Lazy instantiation — only build submodules that are actually requested
        self.ms_ssim_loss = MS_SSIM_Loss()          if 'ms-ssim_loss' in loss_name else None
        self.vgg19        = Vgg19()                  if 'vgg19_loss'   in loss_name else None
        self.vgg16        = Vgg16()                  if 'vgg16_loss'   in loss_name else None
        self.cr_loss      = ContrastLoss()           if 'cr_loss'      in loss_name else None
        self.fcr_loss     = FCR()                    if 'fcr_loss'     in loss_name else None
        self.smoothl1     = MultiSmoothL1Loss(weights=weights) if 'smoothl1'  in loss_name else None
        self.l1           = MultiL1Loss(weights=weights)       if 'l1'        in loss_name else None
        self.fft_loss     = FFT_Loss(weights=weights)          if 'fft_loss'  in loss_name else None
        self.ssim_loss    = SSIM_Loss()              if 'ssim_loss'    in loss_name else None

        # Per-layer weights for VGG feature matching (shallow → deep)
        self.vgg_weights = vgg_weights  # default: [1/5, 1/5, 1/5, 1/5, 1/5]
        self.loss_name   = loss_name

    def percepetual_loss(self, x, y):
        # Multi-scale inputs: use only the full-resolution output for perceptual comparison
        if isinstance(x, list) and isinstance(y, list):
            x, y = x[0], y[0]

        # Lazily move VGG to the correct device on first call
        if next(self.vgg19.parameters()).device != x.device:
            self.vgg19.to(x.device)

        x_vgg, y_vgg = self.vgg19(x), self.vgg19(y)

        # Weighted MSE across all VGG feature levels
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.vgg_weights[i] * F.mse_loss(x_vgg[i], y_vgg[i])
        return loss

    def forward(self, pred, target, smoky=None, mask=None):
        # cr_loss and fcr_loss require the original smoky image as a negative reference
        loss = dict()
        if 'l1'          in self.loss_name: loss['l1']          = self.l1(pred, target)
        if 'smoothl1'    in self.loss_name: loss['smoothl1']    = self.smoothl1(pred, target)
        if 'vgg19_loss'  in self.loss_name: loss['vgg19_loss']  = self.percepetual_loss(pred, target)
        if 'vgg16_loss'  in self.loss_name: loss['vgg16_loss']  = self.vgg16(pred, target)
        if 'ms-ssim_loss'in self.loss_name: loss['ms-ssim_loss']= self.ms_ssim_loss(pred, target)
        if 'cr_loss'     in self.loss_name: loss['cr_loss']     = self.cr_loss(pred, target, smoky)
        if 'fcr_loss'    in self.loss_name: loss['fcr_loss']    = self.fcr_loss(pred, target, smoky)
        if 'fft_loss'    in self.loss_name: loss['fft_loss']    = self.fft_loss(pred, target)
        if 'ssim_loss'   in self.loss_name: loss['ssim_loss']   = self.ssim_loss(pred, target)
        return loss


class FFT_Loss(nn.Module):
    """
    Frequency-domain L1 loss.
    Computes rfft2 on pred/target, stacks real and imaginary parts as the last
    dimension, then applies L1 — penalising errors in both amplitude and phase
    simultaneously without explicitly decomposing them.
    """
    def __init__(self, reduction='mean', weights=None):
        super().__init__()
        self.reduction = reduction
        self.weights   = weights
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        if isinstance(pred, list) and isinstance(target, list):
            weights = self.weights if self.weights is not None else [1.0] * len(pred)
            assert len(pred) == len(target) == len(weights)
            return sum(w * self._fft_loss(p, t) for p, t, w in zip(pred, target, weights))
        return self._fft_loss(pred, target)

    def _fft_loss(self, pred, target):
        pred_fft   = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        # Stack real/imag into a trailing dim so standard L1 covers both components
        pred_fft   = torch.stack([pred_fft.real,   pred_fft.imag],   dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.criterion(pred_fft, target_fft)


class SSIM_Loss(nn.Module):
    """
    SSIM-based loss: loss = 1 - SSIM(pred, target).
    Supports multi-scale predictions via weighted sum over a list of outputs.
    """
    def __init__(self, channels=3, weights=None):
        super().__init__()
        self.ssim    = SSIM(data_range=1., size_average=True, channel=channels)
        self.weights = weights

    def forward(self, pred, target):
        if isinstance(pred, list) and isinstance(target, list):
            weights = self.weights if self.weights is not None else [1.0] * len(pred)
            assert len(pred) == len(target) == len(weights)
            return sum(w * (1 - self.ssim(p, t)) for p, t, w in zip(pred, target, weights))
        return 1 - self.ssim(pred, target)


class MS_SSIM_Loss(nn.Module):
    """Multi-Scale SSIM loss: loss = 1 - MS-SSIM(pred, target)."""
    def __init__(self):
        super().__init__()
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True)

    def forward(self, x, y):
        return 1 - self.ms_ssim(x, y)


class ContrastLoss(nn.Module):
    """
    VGG-based contrastive perceptual loss.

    For each VGG layer i:
        contrastive_i = d(pred, clear) / (d(pred, smoky) + eps)

    Minimising this ratio simultaneously pulls pred toward clear features
    and pushes it away from smoky features in VGG feature space.
    `ablation=True` disables the repulsion term, reducing this to plain perceptual loss.
    """
    def __init__(self, ablation=False):
        super().__init__()
        self.vgg     = Vgg19()
        self.l1      = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  # shallow → deep
        self.ab      = ablation

    def forward(self, a, p, n):
        # a: pred_clear  |  p: real_clear (positive)  |  n: smoky (negative)
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)  # ratio: small = pred is closer to clear than to smoky
            else:
                contrastive = d_ap  # ablation: no negative push
            loss += self.weights[i] * contrastive
        return loss


def sample_with_j(k, n, j):
    """
    Sample `n` unique indices from [0, k) such that index `j` is always included.
    Used in FCR to guarantee the anchor sample appears in every negative set.
    """
    if n >= k:
        raise ValueError("n must be less than k.")
    if j < 0 or j >= k:
        raise ValueError("j must be in range [0, k).")

    remaining = [x for x in range(k) if x != j]
    return [j] + random.sample(remaining, n - 1)


class FCR(nn.Module):
    """
    Frequency-domain Contrastive Regularisation (FCR).

    Operates like ContrastLoss but in the FFT domain instead of VGG feature space.
    For each sample i in the batch:
        - positive: fft distance between pred[i] and clear[i]  (d_ap)
        - negatives: fft distance between pred[i] and smoky[j] for j sampled
          via `sample_with_j` (guarantees diversity while always including i itself)

    Loss = mean over batch of (d_ap / (d_an + eps))
    Encourages the frequency spectrum of pred to match clear rather than smoky.
    """
    def __init__(self, ablation=False):
        super().__init__()
        self.l1          = nn.L1Loss()
        self.multi_n_num = 2  # number of negative samples per anchor

    def forward(self, a, p, n):
        # Transform entire batch to frequency domain at once
        a_fft = torch.fft.fft2(a)
        p_fft = torch.fft.fft2(p)
        n_fft = torch.fft.fft2(n)

        contrastive = 0
        for i in range(a_fft.shape[0]):
            d_ap = self.l1(a_fft[i], p_fft[i])
            # Sample `multi_n_num` negatives, always including index i to avoid trivial avoidance
            for j in sample_with_j(a_fft.shape[0], self.multi_n_num, i):
                d_an = self.l1(a_fft[i], n_fft[j])
                contrastive += d_ap / (d_an + 1e-7)

        # Normalise by total number of (anchor, negative) pairs
        contrastive = contrastive / (self.multi_n_num * a_fft.shape[0])
        return contrastive


class MultiL1Loss(nn.Module):
    """
    L1 loss with optional per-output weighting for multi-scale predictions.
    If pred/target are lists, computes weighted sum: loss = sum(w_i * L1(pred_i, target_i)).
    """
    def __init__(self, reduction='mean', weights=None):
        super().__init__()
        self.criterion = nn.L1Loss(reduction=reduction)
        self.weights   = weights

    def forward(self, pred, target):
        if isinstance(pred, list) and isinstance(target, list):
            weights = self.weights if self.weights is not None else [1.0] * len(pred)
            assert len(pred) == len(target) == len(weights)
            return sum(w * self.criterion(p, t) for p, t, w in zip(pred, target, weights))
        return self.criterion(pred, target)


class MultiSmoothL1Loss(nn.Module):
    """
    SmoothL1 (Huber) loss with optional per-output weighting for multi-scale predictions.
    Behaves like L2 near zero (beta threshold) and L1 further out — more robust to outliers.
    """
    def __init__(self, reduction='mean', beta=1.0, weights=None):
        super().__init__()
        self.criterion = nn.SmoothL1Loss(reduction=reduction, beta=beta)
        self.weights   = weights

    def forward(self, pred, target):
        if isinstance(pred, list) and isinstance(target, list):
            weights = self.weights if self.weights is not None else [1.0] * len(pred)
            assert len(pred) == len(target) == len(weights)
            return sum(w * self.criterion(p, t) for p, t, w in zip(pred, target, weights))
        return self.criterion(pred, target)


class TVLoss(nn.Module):
    """
    Total Variation loss — penalises high-frequency spatial noise.
    Computes mean absolute differences between horizontally and vertically adjacent pixels,
    normalised by the number of valid pixel pairs and batch size.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.size(0)
        h_x, w_x  = x.size(2), x.size(3)

        count_h = (h_x - 1) * w_x   # number of vertical adjacent pairs
        count_w = h_x * (w_x - 1)   # number of horizontal adjacent pairs

        h_tv = torch.abs(x[:, :, 1:, :]  - x[:, :, :h_x-1, :]).sum()
        w_tv = torch.abs(x[:, :, :, 1:]  - x[:, :, :, :w_x-1]).sum()
        return (h_tv / count_h + w_tv / count_w) / batch_size


if __name__ == '__main__':
    device = 'cuda:3'
    x1 = torch.rand((4, 3, 224, 224)).to(device)  # pred_clear
    x2 = torch.rand((4, 3, 224, 224)).to(device)  # real_clear
    x3 = torch.rand((4, 3, 224, 224)).to(device)  # smoky (negative reference)
    deh_loss = Desmoke_Loss(loss_name=['cr_loss']).to(device)
    print(deh_loss(x1, x2, x3))