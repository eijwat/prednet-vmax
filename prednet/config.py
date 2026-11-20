"""Experiment configuration: defaults, YAML loading and command-line overrides.

A config is a nested dict with four sections. Every key has a default
(``DEFAULTS`` below), so a config file only needs the values that differ.

    data:   which frames to use and how to read them
    model:  network shape, initial weights and model options
    train:  optimiser and schedule (used by ``prednet train``)
    predict: prediction settings (used by ``prednet predict``)
    batch:  repeated training + testing (used by ``prednet batch``)
    output: where run folders are created

Overrides from the command line use dotted keys and YAML values:

    --set train.period=40000 --set model.channels=[1,48,96,192]
"""

import copy
import os

import yaml

DEFAULT_RUNS_DIR = "/home/deepimac/Claude/prednet_runs"

DEFAULTS = {
    "name": "run",
    "data": {
        # A list file, an .h5/.hdf5 file, or a YAML list of them. With several
        # entries, training cycles through them in order (the legacy
        # "sequence list" behaviour).
        "train_list": None,
        # List used by "prednet predict" when no list is given on the
        # command line.
        "test_list": None,
        "size": [160, 120],          # width, height in pixels
        "color_space": "RGB",        # RGB, LAB, HSV, YCbCr, CMYK
        # Legacy rule: join list entries to this folder. Leave null to resolve
        # entries relative to each list file (see prednet/data.py).
        "root": None,
    },
    "model": {
        # channels[0] is the number of image channels: 3 = RGB, 1 = gray,
        # 4 = RGB + gray, 2 = audio spectrogram.
        "channels": [3, 48, 96, 192],
        "init_weights": None,        # .pth file to start from, or null
        "round_mode": "down_up_down",  # or "up_down_up"
        "diff_mode": "pos_neg",      # "pos_neg", "pos" or "neg"
        "amp": 0.0,                  # sine offset amplitude (0 = off)
        "omg": 1.0,                  # sine offset angular velocity
    },
    "train": {
        "bprop": 20,                 # frames per sample (back-prop length)
        "period": 1000000,           # stop after this many frames
        "save_every": 100000,        # save a checkpoint every N frames
        "lr": 0.001,
        "lr_rate": 1.0,              # StepLR gamma, applied every 100 steps
        "min_lr": 0.0001,
        "batchsize": 1,
        "shuffle": False,
        "num_workers": 0,
        "loss": "mse",               # "mse", "corr_wise" or "ensemble"
        "useamp": False,             # automatic mixed precision (GPU only)
        "save_images": False,        # write input/prediction images per step
        "tensorboard": False,
        "save_graph": False,         # write model.dot (needs torchviz)
        "seed": None,                # torch.manual_seed before building
        # Flush denormal floats to zero (CPU). Without it, CPU training slows
        # down 5-10x within a few steps as gradients/activations become
        # denormal. Ignored on GPU. Also applies to "prednet predict".
        "flush_denormal": True,
    },
    "predict": {
        "input_len": 20,             # frames given before predicting
        "ext": 10,                   # extra frames predicted from predictions
        "num_max_window": None,      # limit how many past frames are fed
        "batchsize": 1,
        "num_workers": 0,
        "useamp": False,
        "tensorboard": False,
        # Catalog of named test stimuli (see prednet/stimuli.py).
        "stimuli_catalog": "/home/deepimac/Claude/prednet_data/test_images/stimuli.yaml",
    },
    # Used only by "prednet batch" (see prednet/batch.py).
    "batch": {
        "repeats": 1,                # train each dataset this many times
        # name -> train list (or list of lists). Empty: one dataset, named
        # after the config, using data.train_list.
        "datasets": {},
        # One seed per repeat (torch.manual_seed), or null for none. Note: with
        # model.init_weights and shuffle off, CPU training is deterministic,
        # so repeats only differ if seeds/shuffle make them differ.
        "seeds": None,
        "stimuli": [],               # stimulus names to predict after training
        "checkpoints": [],           # frame numbers to test; empty = latest
        "collect": [],               # prediction files copied into summary/
    },
    "output": {
        "runs_dir": DEFAULT_RUNS_DIR,
    },
    "device": "",                    # "" = cuda if available, else cpu
}


def deep_merge(base, override):
    """Return a copy of ``base`` updated recursively with ``override``."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def check_keys(cfg, ref=DEFAULTS, prefix=""):
    """Raise on keys that are not in DEFAULTS, to catch typos early.

    A default of ``{}`` (e.g. batch.datasets) means free-form keys: its
    contents are not checked.
    """
    for k, v in cfg.items():
        if k not in ref:
            raise KeyError("Unknown config key: " + prefix + k)
        if isinstance(v, dict) and isinstance(ref[k], dict) and ref[k]:
            check_keys(v, ref[k], prefix + k + ".")


def apply_override(cfg, expr):
    """Apply one ``a.b.c=value`` override in place. value is parsed as YAML."""
    if "=" not in expr:
        raise ValueError("Override must look like key=value: " + expr)
    key, value = expr.split("=", 1)
    parts = key.strip().split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(value)


def load_config(path=None, overrides=()):
    """Defaults <- YAML file <- overrides, then validated."""
    user = {}
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
    for expr in overrides:
        apply_override(user, expr)
    check_keys(user)
    cfg = deep_merge(DEFAULTS, user)
    if path and "name" not in user:
        cfg["name"] = os.path.splitext(os.path.basename(path))[0]
    return cfg


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def train_lists(cfg):
    """data.train_list as a Python list."""
    lists = cfg["data"]["train_list"]
    if lists is None:
        raise ValueError("data.train_list is not set")
    return [lists] if isinstance(lists, str) else list(lists)
