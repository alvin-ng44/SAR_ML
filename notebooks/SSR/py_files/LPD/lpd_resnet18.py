"""
Plain PyTorch LPD-ResNet18, adapted from learnable_polyphase_sampling's
resnet_custom.py (ResNet18Custom), with the PyTorch Lightning scaffolding
stripped out and a dropout classification head matching the existing
SAR_ML pipeline convention.

DEPENDENCIES:
  - Only the `layers/` folder from learn_poly_sampling needs to be
    copied into your project (layers/__init__.py, polydown.py, polyup.py,
    lowpass_filter.py, lps_logit_layers.py, lps_utils.py). These are
    self-contained (torch only).
  - The BasicBlockCustom / cpad / replace_conv / replace_pool helpers
    from their resnet_custom.py are inlined below instead of imported,
    since the original file has a hard top-level dependency on
    pytorch_lightning / torchmetrics (via .core) that isn't needed here.
  - The ResNet is built via torchvision.models.resnet.ResNet directly
    instead of their private `_resnet()` helper, since that helper's
    signature (pretrained=... vs weights=...) has changed across
    torchvision versions and isn't worth depending on for four lines
    of convenience.
"""

from typing import Optional, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, conv3x3

from layers.polydown import set_pool, PolyphaseInvariantDown2D, LPS
from layers.lps_logit_layers import LPSLogitLayersV2


# --- Inlined from their resnet_custom.py (no Lightning dependency) ---

class cpad(nn.Module):
    """Circular padding module, e.g. for the maxpool-equivalent stage."""
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        return F.pad(x, pad=self.pad, mode="circular")

    def extra_repr(self):
        return f"pad={self.pad}"


def replace_conv(in_ch, out_ch, kernel_size, padding, padding_mode,
                  init, bias=False, stride=1):
    c = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                  kernel_size=kernel_size, padding=padding,
                  padding_mode=padding_mode, bias=bias, stride=stride)
    if init:
        nn.init.kaiming_normal_(c.weight, mode="fan_out", nonlinearity="relu")
    return c


def replace_pool(p, in_ch, out_ch, kernel_size, padding, padding_mode,
                  init, bn, swap_conv_pool=False):
    c = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding,
                  padding_mode=padding_mode, bias=False)
    if init:
        nn.init.kaiming_normal_(c.weight, mode="fan_out", nonlinearity="relu")

    if bn:
        b = nn.BatchNorm2d(out_ch)
        if init:
            nn.init.constant_(b.weight, 1)
            nn.init.constant_(b.bias, 0)
        s = nn.Sequential(c, b, p) if swap_conv_pool else nn.Sequential(p, c, b)
    else:
        s = nn.Sequential(c, p) if swap_conv_pool else nn.Sequential(p, c)
    return s


class BasicBlockCustom(nn.Module):
    """ResNet18 BasicBlock with prob-sharing ("fixed") shortcut, so the
    shortcut's LPD reuses the main branch's polyphase selection."""
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)   # replaced with an LPD Sequential when stride>1
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # Set externally after construction:
        self.ret_prob = False
        self.swap_conv_pool = False
        self.forward_pool_method = "LPS"

    def forward(self, x: Tensor, global_ret_prob=False):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))

        if self.stride > 1 and self.ret_prob:
            if self.swap_conv_pool:
                out = self.conv2[0](out)
                out, _p = self.conv2[1](x=out, ret_prob=True)
            else:
                out, _p = self.conv2[0](x=out, ret_prob=True)
                out = self.conv2[1](out)
            out = self.bn2(out)
            if self.downsample is not None:
                p = _p[0] if (self.forward_pool_method == "LPS" and self.training) else _p
                identity = self.downsample[0](x=x, prob=p)
                identity = self.downsample[1](identity)
                identity = self.downsample[2](identity)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)

        out = self.relu(out + identity)
        if global_ret_prob:
            return out, _p
        return out


def resnet18_fs(num_classes: int = 1000) -> ResNet:
    """ResNet18 with the prob-sharing BasicBlockCustom, built directly
    via torchvision's ResNet class (avoids relying on their private
    _resnet() helper, whose signature varies across torchvision versions)."""
    return ResNet(BasicBlockCustom, [2, 2, 2, 2], num_classes=num_classes)


class LPDResNet18(nn.Module):
    """
    Shift-invariant ResNet18 for SAR ATR (128x128 chips), using
    Learnable Polyphase Downsampling (LPD) in place of standard
    strided conv / maxpool downsampling.

    Mirrors ResNet18Custom's architecture wiring exactly, but:
      - drops all pytorch_lightning dependencies
      - returns raw logits (for CrossEntropyLoss, matching your
        existing training pipeline) instead of log_softmax
        (their code assumed NLLLoss)
      - fc head is nn.Sequential(Dropout, Linear) to match your
        established MultiTaskResNet18 / single-task convention
    """

    def __init__(
        self,
        num_classes: int,
        dropout_p: float = 0.4,
        padding_mode: str = "circular",
        pooling_layer=None,
        conv1_stride: bool = False,      # False -> stride=1 stem, recommended for 128x128
        apply_maxpool: bool = True,
        maxpool_zpad: bool = False,       # False -> circular pad in maxpool-equivalent
        swap_conv_pool: bool = False,     # False -> pool-then-conv ordering
        maxpool_no_antialias: bool = True,
        logits_channels: dict | None = None,
    ):
        super().__init__()

        if pooling_layer is None:
            from functools import partial
            pooling_layer = partial(
                PolyphaseInvariantDown2D,
                component_selection=LPS,
                get_logits=LPSLogitLayersV2,
            )

        self.conv1_stride = conv1_stride
        self.apply_maxpool = apply_maxpool
        self.maxpool_zpad = maxpool_zpad
        self.swap_conv_pool = swap_conv_pool
        self.maxpool_no_antialias = maxpool_no_antialias

        if logits_channels:
            maxpool_h_ch = logits_channels["maxpool"]
            layer2_h_ch = logits_channels["layer2"]
            layer3_h_ch = logits_channels["layer3"]
            layer4_h_ch = logits_channels["layer4"]
        else:
            # This matches the paper's ACTUAL published ImageNet training
            # config (learn_poly_sampling/configs/logits_channels/
            # resnet18_imagenet.json), confirmed directly from the repo --
            # not a guess. ResNet18Custom's own fallback default (used only
            # when logits_channels=None, e.g. in their unit test) is
            # (64, 128, 256, 512) -- full input channels, no bottleneck --
            # but that's a unit-test-only fallback, not what they actually
            # train with. Using their real training config here instead.
            maxpool_h_ch, layer2_h_ch, layer3_h_ch, layer4_h_ch = 8, 16, 32, 64

        # --- Core ResNet18 with fixed (prob-sharing) shortcut ---
        self.core = resnet18_fs()

        # Uniform circular padding on every conv in the model
        for layer in self.core.modules():
            if isinstance(layer, nn.Conv2d):
                layer.padding_mode = padding_mode

        # Pass block-level flags needed by BasicBlockCustom.forward
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer_seq = getattr(self.core, layer_name)
            for block in layer_seq:
                # ret_prob=True is required (not optional) whenever a block
                # has a downsample: it's what makes forward() thread the
                # main branch's polyphase choice into the shortcut's LPD
                # call via prob=... Without it, the shortcut falls back to
                # calling itself as a plain nn.Sequential(x), which never
                # passes prob and crashes since the shortcut LPD has no
                # get_logits of its own (use_get_logits=False by design).
                block.ret_prob = True
                block.swap_conv_pool = self.swap_conv_pool
                block.forward_pool_method = "LPS"

        # --- Stem: conv1 (7x7, stride configurable) ---
        stride = 2 if self.conv1_stride else 1
        self.core.conv1 = replace_conv(
            in_ch=3, out_ch=64, kernel_size=7, padding=3,
            padding_mode=padding_mode, stride=stride, init=True,
        )

        # --- Stem: maxpool-equivalent (LPD-based) ---
        if self.apply_maxpool:
            _maxpool = []
            if not self.conv1_stride:
                # conv1 didn't downsample -> this stage does the first stride-2 reduction
                _maxpool.append(set_pool(
                    pooling_layer=pooling_layer,
                    p_ch=64, h_ch=maxpool_h_ch,
                    no_antialias=self.maxpool_no_antialias,
                ))
            if self.maxpool_zpad:
                _maxpool.append(nn.ZeroPad2d((0, 1, 0, 1)))
            else:
                _maxpool.append(cpad(pad=[0, 1, 0, 1]))
            _maxpool.append(nn.MaxPool2d(kernel_size=2, stride=1))
            _maxpool.append(set_pool(
                pooling_layer=pooling_layer,
                p_ch=64, h_ch=maxpool_h_ch,
            ))
            self.core.maxpool = nn.Sequential(*_maxpool)
        else:
            self.core.maxpool = nn.Sequential()

        # --- layer2/3/4: replace stride-2 downsampling with LPD ---
        p2_1 = set_pool(pooling_layer=pooling_layer, p_ch=128, h_ch=layer2_h_ch)
        p3_1 = set_pool(pooling_layer=pooling_layer, p_ch=256, h_ch=layer3_h_ch)
        p4_1 = set_pool(pooling_layer=pooling_layer, p_ch=512, h_ch=layer4_h_ch)

        self.core.layer2[0].conv1 = replace_conv(
            in_ch=64, out_ch=128, kernel_size=3, padding=1,
            padding_mode=padding_mode, init=True,
        )
        self.core.layer2[0].conv2 = replace_pool(
            p=p2_1, in_ch=128, out_ch=128, kernel_size=3, padding=1,
            padding_mode=padding_mode, swap_conv_pool=self.swap_conv_pool,
            init=True, bn=False,
        )
        self.core.layer3[0].conv1 = replace_conv(
            in_ch=128, out_ch=256, kernel_size=3, padding=1,
            padding_mode=padding_mode, init=True,
        )
        self.core.layer3[0].conv2 = replace_pool(
            p=p3_1, in_ch=256, out_ch=256, kernel_size=3, padding=1,
            padding_mode=padding_mode, swap_conv_pool=self.swap_conv_pool,
            init=True, bn=False,
        )
        self.core.layer4[0].conv1 = replace_conv(
            in_ch=256, out_ch=512, kernel_size=3, padding=1,
            padding_mode=padding_mode, init=True,
        )
        self.core.layer4[0].conv2 = replace_pool(
            p=p4_1, in_ch=512, out_ch=512, kernel_size=3, padding=1,
            padding_mode=padding_mode, swap_conv_pool=self.swap_conv_pool,
            init=True, bn=False,
        )

        # --- shortcut branches: reuse main-branch polyphase selection (no own logits) ---
        p2_2 = set_pool(pooling_layer=pooling_layer, p_ch=64, use_get_logits=False)
        p3_2 = set_pool(pooling_layer=pooling_layer, p_ch=128, use_get_logits=False)
        p4_2 = set_pool(pooling_layer=pooling_layer, p_ch=256, use_get_logits=False)

        self.core.layer2[0].downsample = replace_pool(
            p=p2_2, in_ch=64, out_ch=128, kernel_size=1, padding=0,
            padding_mode=padding_mode, init=True, bn=True,
        )
        self.core.layer3[0].downsample = replace_pool(
            p=p3_2, in_ch=128, out_ch=256, kernel_size=1, padding=0,
            padding_mode=padding_mode, init=True, bn=True,
        )
        self.core.layer4[0].downsample = replace_pool(
            p=p4_2, in_ch=256, out_ch=512, kernel_size=1, padding=0,
            padding_mode=padding_mode, init=True, bn=True,
        )

        # --- head: match your existing dropout convention ---
        self.core.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # Raw logits -> use with nn.CrossEntropyLoss, matching your
        # existing training loop (their original returned log_softmax
        # for NLLLoss instead).
        return self.core(x)


if __name__ == "__main__":
    # Sanity check: circular shift invariance test, mirroring the
    # paper's Appendix A3.2 example.
    torch.manual_seed(0)
    model = LPDResNet18(num_classes=10).eval().double()

    x = torch.randn(2, 3, 128, 128).double()
    with torch.no_grad():
        y_orig = model(x)
        x_roll = torch.roll(x, shifts=(5, 5), dims=(-1, -2))
        y_roll = model(x_roll)

    print("y_orig:", y_orig)
    print("y_roll:", y_roll)
    print("Norm(y_orig - y_roll):", torch.norm(y_orig - y_roll).item())