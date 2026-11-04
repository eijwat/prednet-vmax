"""Repeated training and testing (``python3 -m prednet batch CONFIG``).

Replaces the original auto_train_test.py and auto_benham.py (removed from
legacy/; see git commit 1e000af). One batch is described by an ordinary
config plus a ``batch`` section:

    name: benham
    model: {channels: [3, 48, 96, 192], init_weights: .../56_1945_5000.pth}
    train: {period: 40000, save_every: 10000, lr_rate: 0.9}
    batch:
      repeats: 20
      datasets:                       # name -> train list
        unity: /mnt/.../unity/train_list.txt
        sim000: /mnt/.../sim000/train_list.txt
      stimuli: [bht, bht_rev, bht_static, bhtr, bhtr_rev, bhtr_static]
      checkpoints: [10000, 20000, 30000, 40000]
      collect: [test_19y_0.jpg]

For every dataset and repeat it trains one run, then predicts every
stimulus with every listed checkpoint, then copies the ``collect`` files of
each prediction into ``summary/``:

    <runs_dir>/<date>_<time>_<name>/
        batch.yaml                    the full config used
        status.csv                    one line per finished step
        <dataset>/<repeat>/           an ordinary run folder (see train.py)
            predictions/<checkpoint>_<stimulus>/
        summary/<dataset>_<repeat>_<stimulus>_<checkpoint>_<file>

A batch can be stopped and resumed: ``prednet batch --resume BATCH_DIR``
skips every run and prediction that already finished (marked by a file
``DONE`` inside its folder) and redoes unfinished ones from the start.
"""

import os
import shutil

from . import config as config_mod
from .io import make_run_dir
from .predict import predict_run, prediction_dir, resolve_checkpoint
from .train import train

DONE = "DONE"


def _is_done(path):
    return os.path.exists(os.path.join(path, DONE))


def _mark_done(path):
    open(os.path.join(path, DONE), "w").close()


def datasets_of(cfg):
    """{name: train_list} for the batch; falls back to data.train_list."""
    ds = cfg["batch"]["datasets"]
    if ds:
        return dict(ds)
    return {cfg["name"]: cfg["data"]["train_list"]}


def plan(cfg):
    """All (dataset, repeat) pairs, in the order they are run."""
    return [(name, r) for name in datasets_of(cfg) for r in range(cfg["batch"]["repeats"])]


def run_batch(cfg, batch_dir=None):
    """Run (or resume, when ``batch_dir`` exists) a whole batch."""
    b = cfg["batch"]
    seeds = b["seeds"]
    if seeds is not None and len(seeds) != b["repeats"]:
        raise ValueError("batch.seeds needs one seed per repeat")
    if not b["stimuli"]:
        print("Note: batch.stimuli is empty, so only training is done.")

    if batch_dir is None:
        batch_dir = make_run_dir(cfg["output"]["runs_dir"], cfg["name"])
        config_mod.save_config(cfg, os.path.join(batch_dir, "batch.yaml"))
    summary = os.path.join(batch_dir, "summary")
    os.makedirs(summary, exist_ok=True)
    status = open(os.path.join(batch_dir, "status.csv"), "a")
    print("Batch folder:", batch_dir)

    datasets = datasets_of(cfg)
    steps = plan(cfg)
    for n, (name, r) in enumerate(steps):
        run_dir = os.path.join(batch_dir, name, str(r))
        label = "%s/%d" % (name, r)
        # --- training
        if _is_done(run_dir):
            print("[%d/%d] %s: training already done" % (n + 1, len(steps), label))
        else:
            print("[%d/%d] %s: training" % (n + 1, len(steps), label))
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)  # unfinished: start this run again
            os.makedirs(run_dir)
            run_cfg = config_mod.deep_merge(cfg, {
                "name": "%s_%s_%d" % (cfg["name"], name, r),
                "data": {"train_list": datasets[name]},
                "train": {"seed": seeds[r] if seeds is not None else cfg["train"]["seed"]},
            })
            del run_cfg["batch"]
            train(run_cfg, run_dir=run_dir)
            _mark_done(run_dir)
            status.write("%s,%d,train,done\n" % (name, r))
            status.flush()
        run_cfg = config_mod.load_config(os.path.join(run_dir, "config.yaml"))

        # --- testing
        checkpoints = b["checkpoints"] or [None]
        for stim in b["stimuli"]:
            for ck in checkpoints:
                weights = resolve_checkpoint(run_dir, ck)
                out = prediction_dir(run_dir, weights, stim)
                if not _is_done(out):
                    if os.path.exists(out):
                        shutil.rmtree(out)
                    predict_run(run_cfg, run_dir, checkpoint=ck, stimulus=stim)
                    _mark_done(out)
                    status.write("%s,%d,predict %s %s,done\n" % (name, r, stim, os.path.basename(weights)))
                    status.flush()
                ck_name = os.path.splitext(os.path.basename(weights))[0]
                for fname in b["collect"]:
                    src = os.path.join(out, fname)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(
                            summary, "%s_%d_%s_%s_%s" % (name, r, stim, ck_name, fname)))
                    else:
                        print("  collect: missing " + src)
    status.close()
    print("Batch finished:", batch_dir)
    return batch_dir


def resume_batch(batch_dir):
    cfg = config_mod.load_config(os.path.join(batch_dir, "batch.yaml"))
    return run_batch(cfg, batch_dir=batch_dir)
