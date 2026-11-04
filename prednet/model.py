"""The PredNet network (Lotter, Kreiman & Cox, 2016) in PyTorch.

PredNet is a stack of layers l = 0 .. n_layers-1. Every layer has four kinds
of units:

    A    : the target of the layer. A_0 is the input frame; A_{l+1} is computed
           from the error of the layer below (``update_A``).
    Ahat : the layer's prediction of A, computed from R (``conv``).
    E    : the prediction error, [relu(Ahat - A), relu(A - Ahat)] stacked on
           the channel axis, so it has 2 * a_channels[l] channels.
    R    : the recurrent representation, a ConvLSTM state (``cell``). It reads
           the layer's own error E and the upsampled R of the layer above.

For each time step t the network sees frame t and is scored on how well its
lowest-layer prediction Ahat_0 matches frame t+1.

Original papers and code
------------------------
https://coxlab.github.io/prednet/           (original PredNet)
https://github.com/quadjr/PredNet           (Chainer port)
https://github.com/leido/pytorch-prednet    (PyTorch port this one grew from)
https://doi.org/10.3389/fpsyg.2018.00345    (illusory motion study)

Note on compatibility
---------------------
Sub-module names (``cell{l}``, ``conv{l}``, ``update_A{l}``) and the order of
modules inside each ``nn.Sequential`` are the keys of saved ``.pth`` files.
The order in which modules are created also fixes the random initial
weights. Change neither unless you accept breaking old checkpoints.
"""

from collections import defaultdict
from typing import Literal

import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import AddSin, ConvLSTMCell, SatLU

ROUND_MODES = ("down_up_down", "up_down_up")
DIFF_MODES = ("pos_neg", "pos", "neg")


class PredNet(nn.Module):
    """PredNet with ``len(A_channels)`` layers.

    Parameters
    ----------
    A_channels : list of int
        Channels of A (and Ahat) per layer. ``A_channels[0]`` is the number of
        image channels, e.g. [3, 48, 96, 192] for RGB input.
    R_channels : list of int, optional
        Channels of the ConvLSTM state R per layer. Defaults to A_channels.
    output_mode : 'error' or 'prediction'
        'error' returns (prediction, loss_per_sample, eval_index);
        'prediction' returns only the prediction of the next frame.
    round_mode : 'down_up_down' or 'up_down_up'
        Order of the two passes inside one time step.
        'down_up_down': first the bottom-up error pass, then the top-down
        prediction pass (default). 'up_down_up': the reverse.
    diff_mode : 'pos_neg', 'pos' or 'neg'
        Which half of the error E is used. The unused half is set to zero,
        so the channel count of E does not change.
    device : torch.device
        Kept for backward compatibility. State tensors are now created on the
        same device as the input.
    amp, omg : float
        Amplitude and angular velocity of the optional sine offset added inside
        every ConvLSTMCell, conv and update_A block. amp=0 disables it.

    Input shape: (batch, time, channels, height, width). Height and width must
    be divisible by 2 ** (n_layers - 1), because each layer halves them.
    """

    def __init__(self, A_channels, R_channels=None,
                 output_mode='error',
                 round_mode="down_up_down",
                 diff_mode: Literal["pos_neg", "pos", "neg"] = "pos_neg",
                 device=torch.device("cpu"),
                 amp=0.0,
                 omg=1.0,
                 ):
        super().__init__()
        if output_mode not in ('prediction', 'error'):
            raise ValueError('Invalid output_mode: ' + str(output_mode))
        if round_mode not in ROUND_MODES:
            raise ValueError('Invalid round_mode: ' + str(round_mode))
        if diff_mode not in DIFF_MODES:
            raise ValueError('Invalid diff_mode: ' + str(diff_mode))
        self.device = device
        if R_channels is None:
            R_channels = A_channels
        # A trailing 0 lets the top layer use r_channels[l + 1] = 0, i.e. it
        # has no layer above it to read from.
        self.r_channels = list(R_channels) + [0]
        self.a_channels = list(A_channels)
        self.n_layers = len(R_channels)
        self.output_mode = output_mode
        self.round_mode = round_mode
        self.diff_mode = diff_mode
        # Per-pixel squared error; averaged over C, H, W later.
        self.loss = nn.MSELoss(reduction="none")
        # Intermediate activations collected by forward hooks, for TensorBoard.
        # Cleared at the start of every forward call.
        self.outputs = defaultdict(list)

        # R units: ConvLSTM per layer. Input = own error E (2 * A channels)
        # concatenated with the upsampled R of the layer above.
        for i in range(self.n_layers):
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i + 1],
                                self.r_channels[i], (3, 3), amp=amp, omg=omg)
            setattr(self, 'cell{}'.format(i), cell)
            self._register_hooks(cell, "ConvLSTMCell_layer{}".format(i))

        # Ahat units: R -> prediction of A. The bottom layer is clamped to
        # [0, 1] so it is a valid image.
        for i in range(self.n_layers):
            conv = nn.Sequential(
                nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1),
                AddSin(amp, omg),
                nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)
            self._register_hooks(conv, "Conv_sequential_layer{}".format(i))

        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # A units of the next layer: error E_l -> A_{l+1}, at half resolution.
        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(
                nn.Conv2d(2 * self.a_channels[l], self.a_channels[l + 1], (3, 3), padding=1),
                AddSin(amp, omg),
                self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)
            self._register_hooks(update_A, "UpdateA_layer{}".format(l))

        self.reset_parameters()

    # Kept under the old name too, in case outside scripts call it.
    def register_hooks(self, mod, name):
        self._register_hooks(mod, name)

    def _register_hooks(self, mod, name):
        """Record the first sample of ``mod``'s output into ``self.outputs[name]``."""
        def hook_fn(m, input, output):
            if isinstance(output[0], torch.Tensor):
                self.outputs[name].append(output[0].detach().cpu())
            elif isinstance(output[0], tuple):
                self.outputs[name].append([o.detach().cpu() for o in output[0]])
        mod.register_forward_hook(hook_fn)

    def reset_parameters(self):
        """Re-initialise the ConvLSTM cells (convs keep PyTorch defaults)."""
        for l in range(self.n_layers):
            getattr(self, 'cell{}'.format(l)).reset_parameters()

    def _error(self, Ahat, A):
        """Prediction error E = [relu(Ahat - A), relu(A - Ahat)] per diff_mode."""
        pos = F.relu(Ahat - A) if self.diff_mode in ["pos_neg", "pos"] else torch.zeros_like(Ahat)
        neg = F.relu(A - Ahat) if self.diff_mode in ["pos_neg", "neg"] else torch.zeros_like(Ahat)
        return torch.cat([pos, neg], 1)

    def down_to_up(self, A, Ahat_seq, E_seq):
        """Bottom-up pass: compute E for every layer and feed it upward as A."""
        for l in range(self.n_layers):
            E_seq[l] = self._error(Ahat_seq[l], A)
            if l < self.n_layers - 1:
                A = getattr(self, 'update_A{}'.format(l))(E_seq[l])

    def up_to_down(self, t, Ahat_seq, E_seq, R_seq, H_seq, total_loss, gt):
        """Top-down pass: update R and Ahat from the top layer to the bottom.

        Adds the bottom-layer MSE against ``gt`` (the next frame) to
        ``total_loss`` in place, and returns the frame prediction Ahat_0.
        """
        for l in reversed(range(self.n_layers)):
            cell = getattr(self, 'cell{}'.format(l))
            E = E_seq[l]
            R = R_seq[l]
            # At t = 0 there is no LSTM state yet; start from (R, R) = zeros.
            hx = (R, R) if t == 0 else H_seq[l]
            if l == self.n_layers - 1:
                R, hx = cell(E, hx)
            else:
                R, hx = cell(torch.cat((E, self.upsample(R_seq[l + 1])), 1), hx)
            R_seq[l] = R
            H_seq[l] = hx
            Ahat_seq[l] = getattr(self, 'conv{}'.format(l))(R_seq[l])
            if l == 0:
                frame_prediction = Ahat_seq[l]
                total_loss += self.loss(frame_prediction, gt).mean(axis=(1, 2, 3))
        return frame_prediction

    def forward(self, input):
        """Run the network over a sequence.

        input : (batch, T, C, H, W), T >= 2.

        Frames 0 .. T-2 are fed in; after the last step the prediction refers
        to frame T-1. Returns, in 'error' mode:

        frame_prediction : (batch, C, H, W) prediction of frame T-1
        total_loss       : (batch,) sum over steps of the per-step MSE
                           between Ahat_0 and the next frame
        eval_index       : list (one per layer) of (batch,) MSE between each
                           layer's final Ahat and its A computed from frame T-1
        """
        self.outputs = defaultdict(list)
        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers
        Ahat_seq = [None] * self.n_layers

        batch_size = input.size(0)
        h, w = input.size(-2), input.size(-1)
        device = input.device

        # Zero initial states; each layer has half the resolution of the one below.
        for l in range(self.n_layers):
            E_seq[l] = torch.zeros(batch_size, 2 * self.a_channels[l], h, w, device=device)
            R_seq[l] = torch.zeros(batch_size, self.r_channels[l], h, w, device=device)
            Ahat_seq[l] = torch.zeros(batch_size, self.a_channels[l], h, w, device=device)
            h = h // 2
            w = w // 2

        time_steps = input.size(1)
        total_loss = torch.zeros(batch_size, device=device)
        for t in range(time_steps - 1):
            A = input[:, t]
            gt = input[:, t + 1]
            if self.round_mode == "down_up_down":
                self.down_to_up(A, Ahat_seq, E_seq)
                frame_prediction = self.up_to_down(t, Ahat_seq, E_seq, R_seq, H_seq, total_loss, gt)
            else:
                frame_prediction = self.up_to_down(t, Ahat_seq, E_seq, R_seq, H_seq, total_loss, gt)
                self.down_to_up(A, Ahat_seq, E_seq)

        # Evaluation: compare each layer's last prediction with the A values
        # that the final frame produces, going bottom-up as in down_to_up.
        eval_index = [None] * self.n_layers
        A = input[:, time_steps - 1]
        for l in range(self.n_layers):
            eval_index[l] = self.loss(Ahat_seq[l], A).mean(axis=(1, 2, 3))
            E_seq[l] = self._error(Ahat_seq[l], A)
            if l < self.n_layers - 1:
                A = getattr(self, 'update_A{}'.format(l))(E_seq[l])

        if self.output_mode == 'error':
            return frame_prediction, total_loss, eval_index
        return frame_prediction
