"""End-to-end check of training and prediction against the baseline.

The baseline (/home/deepimac/Claude/prednet_data/baseline, see BASELINE.md) holds the
outputs of the original main.py on the 21-frame sample. This test runs the
new code with configs/sample.yaml and requires:

  * the same loss values in log.csv as in baseline/log_t.txt,
  * checkpoints with exactly the same tensors as baseline/models/*.pth,
  * prediction images byte-identical to baseline/result/,
  * the same eval_index values as baseline/log_p.txt,
  * CorrWise giving the same loss as legacy/corr_wise.py.

Takes about 1.5 minutes on the imac CPU. Run from the project root:

    python3 tests/test_train_predict.py
"""

import filecmp
import importlib.util
import os
import shutil
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from prednet.config import load_config  # noqa: E402
from prednet.predict import predict_run  # noqa: E402
from prednet.train import train  # noqa: E402

BASE = "/home/deepimac/Claude/prednet_data/baseline"


def read_csv(path, header):
    rows = [line.strip() for line in open(path) if line.strip()]
    rows = rows[1:] if header else rows
    return [[float(x) for x in r.replace(" ", "").split(",")] for r in rows]


def test_train_and_predict_match_baseline():
    tmp = tempfile.mkdtemp(prefix="prednet_test_")
    try:
        cfg = load_config(os.path.join(ROOT, "configs", "sample.yaml"),
                          ["output.runs_dir=" + tmp])
        run = train(cfg)

        assert read_csv(os.path.join(run, "log.csv"), True) == \
            read_csv(os.path.join(BASE, "log_t.txt"), False)
        for name in ("0.pth", "20.pth"):
            a = torch.load(os.path.join(run, "checkpoints", name))
            b = torch.load(os.path.join(BASE, "models", name))
            assert list(a) == list(b)
            for k in a:
                assert torch.equal(a[k], b[k]), name + " " + k

        out = predict_run(cfg, run, checkpoint="20")
        images = sorted(f for f in os.listdir(os.path.join(BASE, "result")))
        assert len(images) == 50
        assert sorted(f for f in os.listdir(out) if f.endswith(".jpg")) == images
        match, mismatch, errors = filecmp.cmpfiles(
            os.path.join(BASE, "result"), out, images, shallow=False)
        assert not mismatch and not errors, (mismatch, errors)
        assert read_csv(os.path.join(out, "log.csv"), True) == \
            read_csv(os.path.join(BASE, "log_p.txt"), False)
    finally:
        shutil.rmtree(tmp)


def test_corr_wise_matches_legacy():
    sys.path.append(os.path.join(ROOT, "legacy"))
    spec = importlib.util.spec_from_file_location(
        "legacy_corr_wise", os.path.join(ROOT, "legacy", "corr_wise.py"))
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    from prednet.train import build_loss
    torch.manual_seed(0)
    pred = torch.rand(1, 3, 120, 160)
    target = torch.rand(1, 3, 120, 160)
    old = legacy.CorrWise(torch.nn.L1Loss(), flow_method="FBFlow", return_warped=False,
                          reduction_clip=False, flow_cycle_loss=True, scale_clip=True,
                          device="cpu")
    new = build_loss("corr_wise", "cpu")
    assert torch.equal(old(pred, target), new(pred, target))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("all %d tests passed" % len(tests))
