"""Utility helpers for QWGAN components."""
from __future__ import annotations

import torch


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize convolutional weights with a normal distribution.

    Mirrors the lightweight initialization used in the reference
    QWGAN implementation and keeps other module types untouched.
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Return symmetric padding for a dilated convolution kernel."""
    return int((kernel_size * dilation - dilation) / 2)


__all__ = ["init_weights", "get_padding"]
