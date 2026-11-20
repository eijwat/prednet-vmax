# Third-party notices

This project builds on the PredNet work of others. The sources below were
used as references or starting points.

| Source | License | How it is used here |
|---|---|---|
| [coxlab/prednet](https://github.com/coxlab/prednet) (Lotter, Kreiman & Cox, 2016) | MIT | The original PredNet (Keras). The model follows its design. |
| [quadjr/PredNet](https://github.com/quadjr/PredNet) | Apache-2.0 | Chainer port; used as a reference for the implementation. |
| [leido/pytorch-prednet](https://github.com/leido/pytorch-prednet) | none stated | PyTorch port from which this code grew (see `prednet/model.py`). |
| [Kaixhin's LSTM gist](https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81) | none stated | The ConvLSTM cell in `prednet/layers.py` is based on it. |

## Optical-flow models (removed)

Earlier versions of this repository (before it was made public) carried copies of
[PWC-Net by sniklaus](https://github.com/sniklaus/pytorch-pwc) (GPL-3.0) and
[RAFT](https://github.com/princeton-vl/RAFT) (BSD-3-Clause). They are not
part of this repository. `CorrWise` with `flow_method: FBFlow` (OpenCV) works
without them; to use PWC-Net or RAFT, obtain them from their own repositories
and follow their licenses.

## Citation

If you use this code, please cite the original PredNet paper
(Lotter, Kreiman & Cox, 2016, https://arxiv.org/abs/1605.08104) and the work
that this code was written for: Ueda, Sinapayen & Watanabe, *Predictive networks
generate motion-induced color illusions*, Scientific Reports 16, 27582 (2026),
https://doi.org/10.1038/s41598-026-57953-w.
