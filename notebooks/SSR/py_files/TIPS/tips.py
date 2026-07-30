"""
Translation Invariant Polyphase Sampling (TIPS) pooling layer.

Reimplementation of Saha & Gokhale, "Improving Shift Invariance in
Convolutional Neural Networks with Translation Invariant Polyphase
Sampling" (arXiv:2404.07410), official code at
https://github.com/sourajitcs/tips.

Trimmed to stride=2 only (matches ResNet18 downsampling points) and
made self-contained -- no dependency on the original repo's package
layout. Kept close to their forward-pass logic since that's what's
empirically validated in the paper; naming loosely follows the paper's
notation (tau, psi, x_t) for readability against Eq. 1-5.

Key difference vs. LPD/APS you've already got: TIPS does NOT rely on
circular padding to achieve shift invariance. Padding here defaults to
'reflect' (paper's choice) precisely because the invariance guarantee
comes from the learned undo-regularization (L_undo) on *standard*
shift, not from the polyphase decomposition being exact under
circular wraparound. This is the thing to actually test against your
circular-pad LPD result.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_pad_layer(pad_type: str):
    if pad_type in ("refl", "reflect"):
        return nn.ReflectionPad2d
    if pad_type in ("repl", "replicate"):
        return nn.ReplicationPad2d
    if pad_type == "zero":
        return nn.ZeroPad2d
    raise ValueError(f"Unrecognized pad_type: {pad_type}")


class RandomFeatShift(nn.Module):
    """
    Randomly shifts a feature map by up to (max_shift_h, max_shift_w)
    pixels, either as a 'standard' shift (out-of-bounds pixels zeroed,
    lossy -- what happens with real-world sensor/crop misalignment) or
    a 'circular' shift (wraparound, lossless). Used only to build the
    L_undo training target x_t; not part of the forward path at
    inference.
    """

    def __init__(self, max_shift_h: int = 2, max_shift_w: int = 2,
                 transform_type: str = "standard"):
        super().__init__()
        self.max_shift_h = max_shift_h
        self.max_shift_w = max_shift_w
        self.transform_type = transform_type

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        shift_h = int(np.random.randint(-self.max_shift_h, self.max_shift_h + 1))
        shift_w = int(np.random.randint(-self.max_shift_w, self.max_shift_w + 1))

        if self.transform_type == "circular":
            return torch.roll(feat, shifts=(shift_h, shift_w), dims=(-2, -1))

        if self.transform_type == "standard":
            out = torch.roll(feat, shifts=shift_h, dims=-2)
            if shift_h > 0:
                out[:, :, :shift_h, :] = 0
            elif shift_h < 0:
                out[:, :, shift_h:, :] = 0
            out = torch.roll(out, shifts=shift_w, dims=-1)
            if shift_w > 0:
                out[:, :, :, :shift_w] = 0
            elif shift_w < 0:
                out[:, :, :, shift_w:] = 0
            return out

        raise ValueError(f"Unknown transform_type: {self.transform_type}")


class TIPS(nn.Module):
    """
    Learnable stride-2 polyphase-mixing downsampler.

    forward(x) -> (x_hat, aux)
        x_hat: downsampled feature map, same semantics as a stride-2
               pool/conv -- drop this in wherever LPD's downsampling
               module currently sits.
        aux:   dict with 'tau' (mixing coeffs, for L_FM) and, only
               when self.training, 'psi_x'/'x_t' (for L_undo). aux
               values are None outside training if compute_undo=False.

    Args:
        in_channels: channels of the input feature map.
        pad_type: padding used before polyphase decomposition.
            'reflect' (paper default) -- does NOT require circular
            shift/wraparound semantics, unlike LPD/APS.
        kernel: kernel size of the depthwise conv in the f_theta branch.
        max_shift_h/w: max random standard-shift (in feature-map
            pixels) used to build the L_undo target during training.
        compute_undo: if False, skip building x_t/psi_x entirely
            (e.g. for a quick forward-only sanity check or if you
            want to ablate L_undo without touching the training loop).
    """

    def __init__(
        self,
        in_channels: int,
        pad_type: str = "reflect",
        kernel: int = 3,
        max_shift_h: int = 2,
        max_shift_w: int = 2,
        transform_type: str = "standard",
        compute_undo: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stride = 2
        self.num_poly = 4  # stride^2
        self.compute_undo = compute_undo

        PadLayer = _get_pad_layer(pad_type)
        # four parity-dependent pad configs so H,W become even before
        # the stride-2 slice, exactly matching the official code.
        self.pad_ee = PadLayer((0, 0, 0, 0))
        self.pad_eo = PadLayer((1, 0, 0, 0))
        self.pad_oe = PadLayer((0, 0, 1, 0))
        self.pad_oo = PadLayer((1, 0, 1, 0))

        self.shift = RandomFeatShift(max_shift_h, max_shift_w, transform_type)

        # f_theta: depthwise 3x3 conv -> ReLU (= psi(x)) -> GAP to
        # (stride,stride) -> depthwise 1x1 -> softmax => tau
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel, groups=in_channels,
            padding=(kernel - 1) // 2, stride=1, bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((self.stride, self.stride))
        self.mix_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, groups=in_channels,
            stride=1, bias=False,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        N, C, H, W = x.shape
        if H % 2 == 0 and W % 2 == 0:
            x = self.pad_ee(x)
        elif H % 2 == 0 and W % 2 != 0:
            x = self.pad_eo(x)
        elif H % 2 != 0 and W % 2 == 0:
            x = self.pad_oe(x)
        else:
            x = self.pad_oo(x)
        N, C, H, W = x.shape

        psi_x = self.relu(self.dw_conv(x))          # psi(X), Eq. 4
        tau = self.mix_conv(self.gap(psi_x))
        tau = self.softmax(tau.view(N, C, -1))       # (N, C, 4)

        x_t = None
        if self.compute_undo and self.training:
            x_t = self.shift(x).detach()             # frozen target, Eq. 4

        # polyphase decomposition, Eq. 1
        p0 = x[:, :, 0::2, 0::2]
        p1 = x[:, :, 1::2, 0::2]
        p2 = x[:, :, 0::2, 1::2]
        p3 = x[:, :, 1::2, 1::2]
        stacks = torch.stack([p0, p1, p2, p3], dim=2)  # (N,C,4,H/2,W/2)
        h2, w2 = H // 2, W // 2
        stacks = stacks.view(N, C, self.num_poly, h2 * w2)

        x_hat = (tau.unsqueeze(-1) * stacks).sum(dim=2).view(N, C, h2, w2)

        aux = {"tau": tau, "psi_x": psi_x, "x_t": x_t}
        return x_hat, aux


def tips_fm_loss(tau: torch.Tensor) -> torch.Tensor:
    """
    L_FM (Eq. 3): discourages both skewed tau (one-hot, e.g. [1,0,0,0])
    and uniform tau (avg-pool-equivalent, [.25]*4). Simplifies to
    (1 - s^2) * ||tau||_2 for s=2 -> -3 * ||tau||_2, i.e. minimizing
    this term *maximizes* ||tau||_2 away from the uniform point while
    the softmax normalization keeps it away from one-hot collapse
    only insofar as gradients push back on collapse elsewhere in the
    loss (task loss + L_undo). Call once per TIPS layer, average
    across layers.
    """
    s2 = tau.shape[-1]  # = stride^2 = 4
    return (1 - s2) * torch.norm(tau, p=2, dim=-1).mean()


def tips_undo_loss(psi_x: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    """L_undo (Eq. 4): MSE between psi(X) and the standard-shifted target."""
    return F.mse_loss(psi_x, x_t)