"""Residual prediction-head fusion for the UHI ConvLSTM (optimization Phase 4).

This module does **not** modify the frozen reference model in
``models/convlstm.py``. It reuses the exact same ``ConvLSTM`` backbone and adds
a residual fusion: the head predicts a normalized LST *residual* relative to
the final observed input LST channel, and the forward pass adds that anchor
back.

Compatibility of the quantities (verified against the frozen contract):
    * the last input frame is ``X[:, -1]`` and the LST channel is index 0;
    * the input LST channel and the target both use the **same** LST train
      statistics for z-score normalization (``dataset.py`` lines 210-211 and
      204-211; ``normalization_stats.json`` LST mean/std);
    * therefore ``target_norm - X[:, -1, 0:1]`` is a meaningful normalized
      residual and adding the anchor back reproduces normalized LST.

The public input/output contract is unchanged: ``[B, 12, 18, 128, 128]`` in,
``[B, 1, 128, 128]`` out, using only ONNX-friendly standard ops.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.convlstm import ConvLSTM

__all__ = ["ResidualConvLSTM"]


class ResidualConvLSTM(nn.Module):
    """ConvLSTM backbone whose head predicts a residual over the last LST frame.

    Parameters mirror ``ConvLSTM`` so the ONNX contract and training code are
    shared. The anchor is the last input timestep's LST channel (index 0).
    """

    def __init__(
        self,
        input_channels: int = 18,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        output_channels: int = 1,
        lst_channel_index: int = 0,
    ):
        super().__init__()
        if output_channels != 1:
            raise ValueError("ResidualConvLSTM supports a single output channel")
        self.lst_channel_index = int(lst_channel_index)
        self.backbone = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            output_channels=output_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], received {tuple(x.shape)}")
        residual = self.backbone(x)
        anchor = x[:, -1, self.lst_channel_index : self.lst_channel_index + 1]
        return residual + anchor