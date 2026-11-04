"""Small building blocks used by the PredNet model.

This module holds the generic layers; the PredNet network itself, which stacks
these layers into a hierarchy, lives in ``prednet/model.py``.

Contents
--------
SatLU
    Saturating linear unit. Clamps values into [lower, upper]. Used at the end
    of the lowest layer so that predicted pixels stay in the valid range [0, 1].
AddSin
    Adds a time-varying sine offset ``amp * sin(omg * t)`` to its input. With
    the default ``amp=0`` it does nothing; it exists for experiments that
    inject an oscillation into the network.
ConvLSTMCell
    One step of a convolutional LSTM with peephole connections. This is the
    recurrent unit ("R" units) of every PredNet layer.

Note on compatibility
---------------------
Attribute and parameter names (``weight_ih``, ``bias_ch``, ``wc_blank``, ...)
must stay exactly as they are: they are the keys stored in existing ``.pth``
checkpoint files.
"""

import math

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class SatLU(nn.Module):
    """Saturating linear unit: ``min(max(x, lower), upper)``."""

    def __init__(self, lower=0, upper=1, inplace=False):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        # hardtanh is exactly a clamp between min_val and max_val.
        return F.hardtanh(input, self.lower, self.upper, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return (self.__class__.__name__ + ' ('
                + 'min_val=' + str(self.lower)
                + ', max_val=' + str(self.upper)
                + inplace_str + ')')


class AddSin(nn.Module):
    """Add ``amp * sin(omg * t)`` to the input, where t counts forward calls.

    The counter ``_time`` is advanced once per call and is never reset, so the
    phase keeps running across frames, sequences and batches. This matches the
    original implementation and is kept on purpose.
    """

    def __init__(self, amp=0.0, omg=1.0):
        super().__init__()
        self._time = 0
        self._amp = amp
        self._omg = omg

    def forward(self, input):
        output = input + self._amp * math.sin(self._omg * self._time)
        self._time += 1
        return output


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell with peephole connections.

    Based on https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81
    Local additions: the optional sine offset (``amp``, ``omg``) that is added
    to the three pre-activation terms, as in ``AddSin``.

    Shapes
    ------
    input : (batch, in_channels, H, W)
    hx    : tuple (h, c), each (batch, out_channels, H, W)
    returns ``h_1, (h_1, c_1)``

    Gates
    -----
    The 4 * out_channels pre-activation channels are split in this order:
    i (input gate), f (forget gate), g (candidate cell), o (output gate).
    The peephole term from the cell state ``c`` is computed by a convolution
    with only 3 * out_channels outputs: it feeds i, f and o, while g receives
    zeros (``wc_blank``).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, amp=0.0, omg=1.0):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        # Sine offset state (see AddSin). _time advances once per forward call.
        self._time = 0
        self._amp = amp
        self._omg = omg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # "Same" padding for the hidden-to-hidden and cell-to-hidden convs.
        self.padding_h = tuple(
            k // 2 for k, s, p, d in zip(kernel_size, stride, padding, dilation))
        self.dilation = dilation
        self.groups = groups
        # Input-to-hidden, hidden-to-hidden and cell-to-hidden (peephole) weights.
        self.weight_ih = Parameter(torch.Tensor(
            4 * out_channels, in_channels // groups, *kernel_size))
        self.weight_hh = Parameter(torch.Tensor(
            4 * out_channels, out_channels // groups, *kernel_size))
        self.weight_ch = Parameter(torch.Tensor(
            3 * out_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * out_channels))
            self.bias_hh = Parameter(torch.Tensor(4 * out_channels))
            self.bias_ch = Parameter(torch.Tensor(3 * out_channels))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('bias_ch', None)
        # A single zero, broadcast to fill the missing peephole slot of gate g.
        # Registered as a buffer, so it is saved in checkpoints.
        self.register_buffer('wc_blank', torch.zeros(1, 1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        """Uniform init in [-stdv, stdv] with stdv = 1 / sqrt(4 * C_in * kH * kW)."""
        n = 4 * self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        # The order of these calls fixes the random-number sequence; keep it.
        self.weight_ih.data.uniform_(-stdv, stdv)
        self.weight_hh.data.uniform_(-stdv, stdv)
        self.weight_ch.data.uniform_(-stdv, stdv)
        if self.bias_ih is not None:
            self.bias_ih.data.uniform_(-stdv, stdv)
            self.bias_hh.data.uniform_(-stdv, stdv)
            self.bias_ch.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h_0, c_0 = hx
        offset = self._amp * math.sin(self._omg * self._time)

        wx = F.conv2d(input, self.weight_ih, self.bias_ih,
                      self.stride, self.padding, self.dilation, self.groups)
        wx += offset

        wh = F.conv2d(h_0, self.weight_hh, self.bias_hh, self.stride,
                      self.padding_h, self.dilation, self.groups)
        wh += offset

        # Peephole term. The original gist notes that the paper uses a
        # Hadamard product here; this implementation uses a convolution.
        wc = F.conv2d(c_0, self.weight_ch, self.bias_ch, self.stride,
                      self.padding_h, self.dilation, self.groups)
        wc += offset

        # Re-arrange the 3 peephole blocks into 4 gate slots: [i, f, 0, o].
        blank = self.wc_blank.expand(
            wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3))
        wxhc = wx + wh + torch.cat(
            (wc[:, :2 * self.out_channels], blank, wc[:, 2 * self.out_channels:]), 1)

        oc = self.out_channels
        i = torch.sigmoid(wxhc[:, :oc])
        f = torch.sigmoid(wxhc[:, oc:2 * oc])
        g = torch.tanh(wxhc[:, 2 * oc:3 * oc])
        o = torch.sigmoid(wxhc[:, 3 * oc:])

        c_1 = f * c_0 + i * g
        h_1 = o * torch.tanh(c_1)
        self._time += 1
        return h_1, (h_1, c_1)
