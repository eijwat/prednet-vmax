"""Command line: ``python3 -m prednet <command> ...`` (run from the project root).

Commands
--------
train CONFIG [--set key=value ...]
    Train with a YAML config. A new run folder is created for every call.

predict RUN_DIR [--checkpoint N|PATH] [--list LIST] [--set key=value ...]
    Predict with a checkpoint of a run (default: the latest one), using the
    settings saved in RUN_DIR/config.yaml.

predict --config CONFIG --weights PTH --list LIST --out DIR
    Predict with any weights file (e.g. old models that have no run folder).

prepare VIDEO --out DIR [options]
    Cut a video into frames and write train_list.txt / test_list.txt.

config [CONFIG] [--set key=value ...]
    Print the full config (defaults filled in) without running anything.

Examples
--------
    python3 -m prednet train configs/sample.yaml --set train.period=40
    python3 -m prednet predict /mnt/buffalo/prednet_runs/2026-11-05_1430_sample
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
    p.add_argument("--config", help="config file (when not using a run folder)")
    p.add_argument("--weights", help=".pth file (when not using a run folder)")
    p.add_argument("--out", help="output folder (when not using a run folder)")
    _add_set(p)

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
            predict_run(cfg, args.run_dir, args.checkpoint, args.test_list)
        else:
            if not (args.config and args.weights and args.test_list and args.out):
                parser.error("without RUN_DIR, give --config, --weights, --list and --out")
            cfg = config_mod.load_config(args.config, args.overrides)
            predict(cfg, args.weights, args.test_list, args.out)

    elif args.command == "prepare":
        from .prepare import prepare
        prepare(args.video, args.out, width=args.width, height=args.height,
                test_ratio=args.test_ratio, prefix=args.prefix)

    elif args.command == "config":
        cfg = config_mod.load_config(args.config, args.overrides)
        yaml.safe_dump(cfg, sys.stdout, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    main()
