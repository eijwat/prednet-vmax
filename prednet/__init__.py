"""PredNet in PyTorch.

Kenta Tanaka, Manabu Kosaka & Eiji Watanabe, 2021. Refactored 2026.

Package layout
--------------
prednet.layers : ConvLSTMCell, SatLU, AddSin (generic building blocks)
prednet.model  : PredNet (the network)
"""

from .model import PredNet

__all__ = ["PredNet"]
