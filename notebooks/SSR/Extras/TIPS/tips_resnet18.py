"""
TIPSResNet18: ResNet18 with Translation Invariant Polyphase Sampling
(TIPS) downsampling in place of LPD's PolyphaseInvariantDown2D, at the
stem and at the first block of each of layer1-4.

Wiring follows the official TIPS resnet.py (sourajitcs/tips) exactly:
  - main branch: conv1(stride=1) -> bn1 -> relu -> TIPS(downsample) -> conv2 -> bn2
    ("conv-then-pool", same as antialiased-cnns/BlurPool style)
  - shortcut branch: TIPS(x) -> 1x1 conv -> bn
    ("pool-then-conv", i.e. TIPS is applied directly to the raw input)

This mirrors your lpd_resnet18.py conventions:
  - self.core wraps the *entire* backbone including the classifier head,
    so `model.core.children()` still works for feature extraction
    (conv1, bn1, relu, [stem_pool], layer1..4, avgpool, fc) exactly as
    in your UMAP/t-SNE extraction code.
  - fc is nn.Sequential(Dropout(p), Linear(512, num_classes)) -> raw
    logits, for CrossEntropyLoss (matching LPDResNet18, not
    LPDResNet18Match's log_softmax+NLLLoss variant).
  - padding_mode default is 'reflect', not 'circular' -- this is the
    actual thing to test against your LPD circular-pad result, since
    TIPS's invariance claim doesn't depend on circular wraparound.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tips import TIPS, tips_fm_loss, tips_undo_loss


class TIPSBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        pad_type: str = "reflect",
        max_shift_h: int = 2,
        max_shift_w: int = 2,
        transform_type: str = "standard",
        compute_undo: bool = True,
    ):
        super().__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: Optional[TIPS] = None
        if stride != 1:
            self.downsample = TIPS(
                in_channels=planes, pad_type=pad_type,
                max_shift_h=max_shift_h, max_shift_w=max_shift_w,
                transform_type=transform_type, compute_undo=compute_undo,
            )

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut_tips: Optional[TIPS] = None
        self.shortcut_conv = None
        self.shortcut_bn = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_tips = TIPS(
                in_channels=in_planes, pad_type=pad_type,
                max_shift_h=max_shift_h, max_shift_w=max_shift_w,
                transform_type=transform_type, compute_undo=compute_undo,
            ) if stride != 1 else None
            # Spatial downsampling (when needed) already happened via
            # shortcut_tips above, so this 1x1 conv is always stride=1.
            self.shortcut_conv = nn.Conv2d(
                in_planes, self.expansion * planes, kernel_size=1,
                stride=1, bias=False,
            )
            self.shortcut_bn = nn.BatchNorm2d(self.expansion * planes)

    def forward(self, x: torch.Tensor):
        aux: List[dict] = []

        out = self.relu(self.bn1(self.conv1(x)))
        if self.downsample is not None:
            out, a = self.downsample(out)
            aux.append(a)
        out = self.bn2(self.conv2(out))

        if self.shortcut_conv is not None:
            sc = x
            if self.shortcut_tips is not None:
                sc, a = self.shortcut_tips(sc)
                aux.append(a)
            sc = self.shortcut_bn(self.shortcut_conv(sc))
        else:
            sc = x

        out = self.relu(out + sc)
        return out, aux


class _TIPSResNetCore(nn.Module):
    """Full backbone incl. classifier -- this becomes TIPSResNet18.core."""

    def __init__(
        self,
        num_blocks=(2, 2, 2, 2),
        num_classes: int = 10,
        in_channels: int = 3,
        dropout_p: float = 0.4,
        pad_type: str = "reflect",
        stem_downsample: bool = True,
        max_shift_h: int = 2,
        max_shift_w: int = 2,
        transform_type: str = "standard",
        compute_undo: bool = True,
    ):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.stem_pool: Optional[TIPS] = None
        if stem_downsample:
            self.stem_pool = TIPS(
                in_channels=64, pad_type=pad_type,
                max_shift_h=max_shift_h, max_shift_w=max_shift_w,
                transform_type=transform_type, compute_undo=compute_undo,
            )

        block_kwargs = dict(
            pad_type=pad_type, max_shift_h=max_shift_h, max_shift_w=max_shift_w,
            transform_type=transform_type, compute_undo=compute_undo,
        )
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, **block_kwargs)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, **block_kwargs)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, **block_kwargs)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, **block_kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512 * TIPSBasicBlock.expansion, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride, **block_kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(TIPSBasicBlock(self.in_planes, planes, stride=s, **block_kwargs))
            self.in_planes = planes * TIPSBasicBlock.expansion
        return nn.ModuleList(layers)

    def _run_layer(self, layer: nn.ModuleList, x, aux_all: List[dict]):
        for block in layer:
            x, aux = block(x)
            aux_all.extend(aux)
        return x

    def forward(self, x: torch.Tensor):
        aux_all: List[dict] = []

        out = self.relu(self.bn1(self.conv1(x)))
        if self.stem_pool is not None:
            out, a = self.stem_pool(out)
            aux_all.append(a)

        out = self._run_layer(self.layer1, out, aux_all)
        out = self._run_layer(self.layer2, out, aux_all)
        out = self._run_layer(self.layer3, out, aux_all)
        out = self._run_layer(self.layer4, out, aux_all)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out, aux_all


class TIPSResNet18(nn.Module):
    """
    Drop-in counterpart to your LPDResNet18.

    model(x) -> (logits, aux)
        aux is a list of {'tau', 'psi_x', 'x_t'} dicts, one per TIPS
        layer that fired: 7 for the default config -- stem_pool (1),
        plus main-branch + shortcut-branch TIPS at the first block of
        each of layer2/3/4 (2*3=6). layer1 has stride=1 so no
        downsampling block fires there. Feed the aux list straight
        into `tips_regularization_loss` below.

    In eval() / no_grad(), psi_x/x_t come back as None automatically
    (TIPS only builds the undo-shift target when self.training), so
    aux still contains 'tau' for inspection but costs nothing extra.
    """

    def __init__(
        self,
        num_classes: int,
        dropout_p: float = 0.4,
        pad_type: str = "reflect",         # 'reflect' (paper default) -- NOT circular
        stem_downsample: bool = True,       # mirrors your apply_maxpool=True convention
        max_shift_h: int = 2,
        max_shift_w: int = 2,
        transform_type: str = "standard",   # 'standard' shift for L_undo (the paper's main claim)
        compute_undo: bool = True,
    ):
        super().__init__()
        self.core = _TIPSResNetCore(
            num_blocks=(2, 2, 2, 2), num_classes=num_classes, in_channels=3,
            dropout_p=dropout_p, pad_type=pad_type, stem_downsample=stem_downsample,
            max_shift_h=max_shift_h, max_shift_w=max_shift_w,
            transform_type=transform_type, compute_undo=compute_undo,
        )

    def forward(self, x: torch.Tensor):
        return self.core(x)


def tips_regularization_loss(
    aux: List[dict],
    epoch: int,
    wake_up_epoch: int,
    alpha: float = 0.35,
) -> torch.Tensor:
    """
    Combine L_FM and L_undo from a forward pass's aux list, per Eq. 5:
        L_reg = L_FM + 1(epoch >= wake_up_epoch) * alpha * L_undo
    (L_undo only switched on after `wake_up_epoch` == epsilon*N epochs,
    epsilon=0.4 in the paper -> wake_up_epoch = int(0.4 * total_epochs)).

    Add this to (1-alpha)*task_loss yourself in your training loop:
        loss = (1 - alpha) * criterion(logits, y) + tips_regularization_loss(aux, epoch, wake_up_epoch, alpha)
    """
    device = aux[0]["tau"].device
    fm_terms = [tips_fm_loss(a["tau"]) for a in aux]
    l_fm = torch.stack(fm_terms).mean()

    if epoch >= wake_up_epoch and aux[0]["x_t"] is not None:
        undo_terms = [tips_undo_loss(a["psi_x"], a["x_t"]) for a in aux]
        l_undo = torch.stack(undo_terms).mean()
    else:
        l_undo = torch.zeros((), device=device)

    return l_fm + alpha * l_undo


if __name__ == "__main__":
    # Sanity check: shapes + standard-shift consistency at random init.
    # (Unlike LPD, TIPS does NOT guarantee exact invariance at random
    # init -- invariance is *learned* via L_undo over training, so this
    # is just a shape/forward check, not an invariance proof like your
    # LPD norm-check.)
    torch.manual_seed(0)
    model = TIPSResNet18(num_classes=10, pad_type="reflect")
    model.train()
    x = torch.randn(4, 3, 128, 128)
    logits, aux = model(x)
    print("logits:", logits.shape)
    print("num TIPS layers fired:", len(aux))
    for i, a in enumerate(aux):
        print(f"  layer {i}: tau {a['tau'].shape}, psi_x {a['psi_x'].shape}, "
              f"x_t {'None' if a['x_t'] is None else a['x_t'].shape}")

    loss_reg = tips_regularization_loss(aux, epoch=50, wake_up_epoch=40, alpha=0.35)
    print("reg loss (post-wakeup):", loss_reg.item())

    model.eval()
    with torch.no_grad():
        logits_eval, aux_eval = model(x)
    print("eval aux[0]['x_t'] is None:", aux_eval[0]["x_t"] is None)