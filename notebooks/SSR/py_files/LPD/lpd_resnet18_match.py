"""
LPDResNet18Match: a thin variant of LPDResNet18 for apples-to-apples
comparison against the original learnable_polyphase_sampling repo's
ResNet18Custom.

Differences from LPDResNet18 (your main training model):
  - fc head is plain nn.Linear(512, num_classes), no Dropout -- matches
    ResNet18Custom.core.fc = nn.Linear(512, num_classes)
  - forward() applies F.log_softmax(..., dim=1) and is meant to be paired
    with nn.NLLLoss(), not nn.CrossEntropyLoss() -- matches
    ResNet18Custom.forward()

Everything else (conv1_stride, swap_conv_pool, padding_mode, LPD wiring,
prob-threading through shortcuts) is identical to LPDResNet18, since
those are the actual architectural properties under test -- only the
head/loss convention is changed here, since that's a cosmetic difference
that would otherwise confound the comparison.

Usage for the side-by-side test:

    model_mine  = LPDResNet18Match(num_classes=N, conv1_stride=True, swap_conv_pool=True)
    # ...construct their_model = ResNet18Custom(..., extras_model={...conv1_stride: True, swap_conv_pool: True...})
    # then train both with the same seed/data/optimizer/scheduler, using
    # criterion = nn.NLLLoss() for both.
"""

import torch
from torch import nn
import torch.nn.functional as F

from lpd_resnet18 import LPDResNet18


class LPDResNet18Match(LPDResNet18):
    def __init__(self, num_classes: int, **kwargs):
        # dropout_p is forced to 0.0 regardless of what's passed, to match
        # ResNet18Custom's plain nn.Linear head (no dropout at all).
        kwargs.pop("dropout_p", None)
        super().__init__(num_classes=num_classes, dropout_p=0.0, **kwargs)

        # Replace the Sequential(Dropout, Linear) head with a plain Linear,
        # matching self.core.fc = nn.Linear(512, num_classes) exactly.
        self.core.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.core(x)
        return F.log_softmax(out, dim=1)


if __name__ == "__main__":
    # Same circular-shift sanity check as lpd_resnet18.py, just confirming
    # the head/loss-convention change didn't break shift invariance.
    torch.manual_seed(0)
    model = LPDResNet18Match(num_classes=10, conv1_stride=False, swap_conv_pool=True).eval().double()

    x = torch.randn(2, 3, 128, 128).double()
    with torch.no_grad():
        y_orig = model(x)
        x_roll = torch.roll(x, shifts= 5, dims=-1)
        y_roll = model(x_roll)

    print("y_orig:", y_orig)
    print("y_roll:", y_roll)
    print("Norm(y_orig - y_roll):", torch.norm(y_orig - y_roll).item())