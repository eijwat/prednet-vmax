"""Small end-to-end batch: 2 repeats on the sample frames, 2 stimuli.

Checks the folder layout, the summary files, that a stimulus gives the
same images as the old list file it came from, and that --resume skips
finished work. About 4 minutes on the imac CPU. Run from the project root:

    python3 tests/test_batch.py
"""

import filecmp
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from prednet.batch import resume_batch, run_batch  # noqa: E402
from prednet.config import load_config  # noqa: E402
from prednet.predict import predict  # noqa: E402

DATA = os.environ.get("PREDNET_DATA", os.path.expanduser("~/prednet_data"))


def test_small_batch_and_resume():
    tmp = tempfile.mkdtemp(prefix="prednet_batch_test_")
    try:
        cfg = load_config(os.path.join(ROOT, "configs", "sample.yaml"), [
            "name=minibatch",
            "output.runs_dir=" + tmp,
            "batch.repeats=2",
            "batch.stimuli=[bht, bht_static]",
            "batch.checkpoints=[20]",
            "batch.collect=[test_19y_0.jpg]",
        ])
        bdir = run_batch(cfg)

        for r in (0, 1):
            run = os.path.join(bdir, "minibatch", str(r))
            assert os.path.exists(os.path.join(run, "DONE"))
            for stim in ("bht", "bht_static"):
                assert os.path.exists(os.path.join(run, "predictions", "20_" + stim, "DONE"))
        summary = sorted(os.listdir(os.path.join(bdir, "summary")))
        assert summary == sorted("minibatch_%d_%s_20_test_19y_0.jpg" % (r, s)
                                 for r in (0, 1) for s in ("bht", "bht_static"))

        # Same init weights, no shuffle, CPU: both repeats are identical.
        a = os.path.join(bdir, "minibatch", "0", "predictions", "20_bht")
        b = os.path.join(bdir, "minibatch", "1", "predictions", "20_bht")
        imgs = sorted(f for f in os.listdir(a) if f.endswith(".jpg"))
        assert not filecmp.cmpfiles(a, b, imgs, shallow=False)[1]

        # The stimulus "bht" equals the old list legacy_lists/bht_test_list.txt.
        run_cfg = load_config(os.path.join(bdir, "minibatch", "0", "config.yaml"),
                              ["data.root=" + DATA])
        old = os.path.join(tmp, "old_list")
        predict(run_cfg, os.path.join(bdir, "minibatch", "0", "checkpoints", "20.pth"),
                os.path.join(DATA, "test_images", "legacy_lists", "bht_test_list.txt"), old)
        assert not filecmp.cmpfiles(a, old, imgs, shallow=False)[1]

        # Resume: nothing is redone.
        stamp = os.path.getmtime(os.path.join(bdir, "minibatch", "0", "checkpoints", "20.pth"))
        resume_batch(bdir)
        assert os.path.getmtime(os.path.join(bdir, "minibatch", "0", "checkpoints", "20.pth")) == stamp
    finally:
        shutil.rmtree(tmp)


def test_batch_config_checks():
    from prednet.batch import check_batch
    base = os.path.join(ROOT, "configs", "benham_batch.yaml")
    check_batch(load_config(base))  # the shipped example is valid
    for override, error in [("batch.stimuli=[bht, no_such]", KeyError),
                            ("batch.checkpoints=[15000]", ValueError),
                            ("batch.datasets={x: /no/such/list.txt}", FileNotFoundError)]:
        try:
            check_batch(load_config(base, [override]))
        except error:
            continue
        raise AssertionError("not rejected: " + override)


if __name__ == "__main__":
    test_batch_config_checks()
    print("ok   test_batch_config_checks")
    test_small_batch_and_resume()
    print("ok   test_small_batch_and_resume")
