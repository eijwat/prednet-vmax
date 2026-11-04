"""Command line: ``python3 -m prednet <command> ...`` (run from the project root).

Commands
--------
train CONFIG [--set key=value ...]
    Train with a YAML config. A new run folder is created for every call.

predict RUN_DIR [--checkpoint N|PATH] [--stimulus NAME ... | --list LIST]
    Predict with a checkpoint of a run (default: the latest one), using the
    settings saved in RUN_DIR/config.yaml. --stimulus takes names from the
    stimuli catalog and can be repeated.

predict --config CONFIG --weights PTH --list LIST --out DIR
    Predict with any weights file (e.g. old models that have no run folder).

batch CONFIG [--set key=value ...]  |  batch --resume BATCH_DIR
    Train several datasets / repeats and test them on stimuli (see batch.py).

stimuli [NAME ...]
    List the stimuli in the catalog, or show the frames of the given ones.

prepare VIDEO --out DIR [options]
    Cut a video into frames and write train_list.txt / test_list.txt.

config [CONFIG] [--set key=value ...]
    Print the full config (defaults filled in) without running anything.

Examples
--------
    python3 -m prednet train configs/sample.yaml --set train.period=40
    python3 -m prednet predict /mnt/buffalo/prednet_runs/2026-11-05_1430_sample
    python3 -m prednet predict RUN_DIR --stimulus bht --stimulus bht_rev
    python3 -m prednet batch configs/benham_batch.yaml
"""

import argparse
import os
import sys

import yaml

from . import config as config_mod


def _add_set(p):
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="override a config value, e.g. train.period=40000 (repeatable)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m prednet",
                                     description="PredNet in PyTorch")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a model")
    p.add_argument("config", help="YAML config file")
    _add_set(p)

    p = sub.add_parser("predict", help="predict frames with a trained model")
    p.add_argument("run_dir", nargs="?", help="run folder made by train")
    p.add_argument("--checkpoint", help="frame number in checkpoints/, or a .pth path")
    p.add_argument("--list", dest="test_list", help="list of frames to predict")
    p.add_argument("--stimulus", action="append", default=[],
                   help="stimulus name from the catalog (repeatable)")
    p.add_argument("--config", help="config file (when not using a run folder)")
    p.add_argument("--weights", help=".pth file (when not using a run folder)")
    p.add_argument("--out", help="output folder (when not using a run folder)")
    _add_set(p)

    p = sub.add_parser("batch", help="repeated training and testing")
    p.add_argument("config", nargs="?", help="YAML config with a batch section")
    p.add_argument("--resume", metavar="BATCH_DIR", help="continue a stopped batch")
    _add_set(p)

    p = sub.add_parser("stimuli", help="list or show test stimuli")
    p.add_argument("names", nargs="*")
    p.add_argument("--catalog", default=None, help="catalog file (default: predict.stimuli_catalog)")

    p = sub.add_parser("prepare", help="cut a video into frames")
    p.add_argument("video")
    p.add_argument("--out", required=True, help="folder for the frames")
    p.add_argument("--width", type=int, default=-1, help="resize width (keep aspect if no height)")
    p.add_argument("--height", type=int, default=-1, help="resize height")
    p.add_argument("--test-ratio", type=float, default=0.1,
                   help="fraction of frames at the end used for test_list.txt")
    p.add_argument("--prefix", default="frame", help="file name prefix")

    p = sub.add_parser("config", help="print the effective config")
    p.add_argument("config", nargs="?")
    _add_set(p)

    args = parser.parse_args(argv)

    if args.command == "train":
        from .train import train
        cfg = config_mod.load_config(args.config, args.overrides)
        train(cfg)

    elif args.command == "predict":
        from .predict import predict, predict_run
        if args.run_dir:
            cfg = config_mod.load_config(os.path.join(args.run_dir, "config.yaml"), args.overrides)
            if args.stimulus and args.test_list:
                parser.error("give --stimulus or --list, not both")
            for stim in args.stimulus or [None]:
                predict_run(cfg, args.run_dir, args.checkpoint, args.test_list, stimulus=stim)
        else:
            if not (args.config and args.weights and args.out) or \
                    bool(args.test_list) == bool(args.stimulus) or len(args.stimulus) > 1:
                parser.error("without RUN_DIR, give --config, --weights, --out and "
                             "either --list or one --stimulus")
            cfg = config_mod.load_config(args.config, args.overrides)
            source = args.test_list
            if args.stimulus:
                from .stimuli import stimulus_entries
                source = stimulus_entries(args.stimulus[0], cfg["predict"]["stimuli_catalog"])
            predict(cfg, args.weights, source, args.out)

    elif args.command == "batch":
        from .batch import resume_batch, run_batch
        if args.resume:
            if args.config or args.overrides:
                parser.error("--resume uses the saved batch.yaml; give no config or --set")
            resume_batch(args.resume)
        else:
            if not args.config:
                parser.error("give a config file or --resume BATCH_DIR")
            run_batch(config_mod.load_config(args.config, args.overrides))

    elif args.command == "stimuli":
        from . import stimuli as stim_mod
        path = args.catalog or config_mod.DEFAULTS["predict"]["stimuli_catalog"]
        catalog = stim_mod.load_catalog(path)
        if not args.names:
            for name, e in catalog.items():
                n = e.get("length", len(stim_mod.unroll(e["frames"])))
                print("%-22s %4d frames  %s" % (name, n, e.get("note", "")))
        for name in args.names:
            entries = stim_mod.expand(catalog, name, os.path.dirname(os.path.abspath(path)))
            print("%s (%d frames):" % (name, len(entries)))
            for e in entries:
                print("  " + e[0])

    elif args.command == "prepare":
        from .prepare import prepare
        prepare(args.video, args.out, width=args.width, height=args.height,
                test_ratio=args.test_ratio, prefix=args.prefix)

    elif args.command == "config":
        cfg = config_mod.load_config(args.config, args.overrides)
        yaml.safe_dump(cfg, sys.stdout, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    main()
