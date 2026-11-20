"""Check that prednet.model.PredNet is numerically identical to legacy/prednet.py.

Run from the project root:

    python3 tests/test_model_equivalence.py

(The functions are also pytest-compatible, if pytest is installed later.)

What is compared, for several option combinations:
  * the random initial weights for the same seed (same creation order),
  * loading a real checkpoint saved by the original code,
  * forward outputs (prediction, loss, eval_index) on real frames,
  * gradients after one backward pass,
  * the sine-offset counters, by running two forward calls in a row.
All comparisons are exact (torch.equal), not approximate.
"""

import os
import sys

import importlib.util

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# legacy/prednet.py does "from convlstmcell import ...", so legacy/ must be
# importable, but AFTER the root so that "prednet" means the new package.
sys.path.append(os.path.join(ROOT, "legacy"))

from prednet.model import PredNet  # noqa: E402  new package

# The legacy module has the same name as the new package, so load it by path.
_spec = importlib.util.spec_from_file_location(
    "legacy_prednet", os.path.join(ROOT, "legacy", "prednet.py"))
legacy_prednet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy_prednet)

DATA = os.environ.get("PREDNET_DATA", os.path.expanduser("~/prednet_data"))
CHECKPOINT = os.path.join(DATA, "weights", "init", "init4.pth")
FRAMES = os.path.join(DATA, "frames", "sample")
CHANNELS = [3, 48, 96, 192]


def load_frames(n=6, size=(160, 120)):
    """(1, n, 3, H, W) float tensor of real frames in [0, 1]."""
    imgs = []
    for i in range(n):
        img = Image.open(os.path.join(FRAMES, "frame_%05d.jpg" % i)).resize(size)
        imgs.append(np.asarray(img).transpose(2, 0, 1).astype(np.float32) / 255)
    return torch.from_numpy(np.stack(imgs))[None]


def build_pair(seed=0, **kw):
    """Build legacy and new models from the same seed."""
    torch.manual_seed(seed)
    old = legacy_prednet.PredNet(CHANNELS, **kw)
    torch.manual_seed(seed)
    new = PredNet(CHANNELS, **kw)
    return old, new


def assert_same_state(old, new):
    a, b = old.state_dict(), new.state_dict()
    assert list(a) == list(b), "state_dict keys differ"
    for k in a:
        assert torch.equal(a[k], b[k]), "weight differs: " + k


def assert_same_run(old, new, x):
    """Forward + backward on both models; everything must match exactly."""
    out_old = old(x)
    out_new = new(x)
    assert torch.equal(out_old[0], out_new[0]), "prediction differs"
    assert torch.equal(out_old[1], out_new[1]), "loss differs"
    for l, (eo, en) in enumerate(zip(out_old[2], out_new[2])):
        assert torch.equal(eo, en), "eval_index differs at layer %d" % l
    out_old[1].mean().backward()
    out_new[1].mean().backward()
    for (k, po), (_, pn) in zip(old.named_parameters(), new.named_parameters()):
        assert torch.equal(po.grad, pn.grad), "gradient differs: " + k
    assert list(old.outputs) == list(new.outputs), "hook names differ"


def test_random_init_identical():
    old, new = build_pair(seed=123)
    assert_same_state(old, new)


def test_checkpoint_loads_and_matches():
    sd = torch.load(CHECKPOINT, map_location="cpu")
    old, new = build_pair()
    old.load_state_dict(sd)
    new.load_state_dict(sd)
    assert_same_run(old, new, load_frames())


def test_option_combinations():
    x = load_frames()
    # The legacy code only checks round_mode == "down_up_down"; any other
    # string meant up_down_up, so "up_down_up" is used for both here.
    for kw in [dict(),
               dict(round_mode="up_down_up"),
               dict(diff_mode="pos"),
               dict(diff_mode="neg"),
               dict(amp=0.05, omg=0.3)]:
        old, new = build_pair(seed=7, **kw)
        assert_same_state(old, new)
        # Two calls in a row also checks that the sine counters advance alike.
        assert_same_run(old, new, x)
        old.zero_grad()
        new.zero_grad()
        assert_same_run(old, new, x)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("all %d tests passed" % len(tests))
