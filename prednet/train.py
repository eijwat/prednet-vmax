"""Training (``python3 -m prednet train CONFIG``).

Everything a run produces goes into its own run folder:

    <runs_dir>/<date>_<time>_<name>/
        config.yaml        the full config actually used (defaults filled in)
        log.csv            frames,loss  -- one line per optimiser step
        checkpoints/       <frames>.pth every train.save_every frames
        train_images/      only with train.save_images
        tensorboard/       only with train.tensorboard
        model.dot          only with train.save_graph

The numerical procedure is the same as ``train()`` in the original main.py
(git 1e000af, also in /home/deepimac/Claude/prednet_data/baseline); with the same
initial weights and data it produces bit-identical checkpoints (checked by
tests/test_train_predict.py).
"""

import os

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as config_mod
from .data import make_dataset
from .io import (load_weights, make_run_dir, make_writer, save_weights,
                 write_image, write_outputs)
from .model import PredNet


def resolve_device(name):
    if name:
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(cfg, device):
    """PredNet from cfg["model"], on ``device``, with init weights if given."""
    m = cfg["model"]
    net = PredNet(
        m["channels"],
        round_mode=m["round_mode"],
        diff_mode=m["diff_mode"],
        device=device,
        amp=m["amp"],
        omg=m["omg"],
    ).to(device)
    return net


def build_loss(name, device):
    """The CorrWise loss, only when the chosen loss needs it.

    The plain "mse" loss is the error that PredNet itself returns, so no extra
    module is built for it (it needs OpenCV optical flow otherwise).
    """
    if name not in ("corr_wise", "ensemble"):
        if name != "mse":
            raise ValueError('train.loss must be "mse", "corr_wise" or "ensemble"')
        return None
    from .losses.corr_wise import CorrWise
    return CorrWise(nn.L1Loss(), flow_method="FBFlow", return_warped=False,
                    reduction_clip=False, flow_cycle_loss=True, scale_clip=True,
                    device=device)


def train(cfg, run_dir=None):
    """Train according to ``cfg``. Returns the run folder path."""
    d, t = cfg["data"], cfg["train"]
    device = resolve_device(cfg["device"])
    width, height = d["size"]
    channels = cfg["model"]["channels"]
    lists = config_mod.train_lists(cfg)

    if run_dir is None:
        run_dir = make_run_dir(cfg["output"]["runs_dir"], cfg["name"])
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    img_dir = os.path.join(run_dir, "train_images")
    if t["save_images"]:
        os.makedirs(img_dir, exist_ok=True)
    config_mod.save_config(cfg, os.path.join(run_dir, "config.yaml"))
    print("Run folder:", run_dir)

    if t["seed"] is not None:
        torch.manual_seed(t["seed"])
    net = build_model(cfg, device)
    loss_fn = build_loss(t["loss"], device)
    if cfg["model"]["init_weights"]:
        load_weights(net, cfg["model"]["init_weights"], device)

    optimizer = torch.optim.Adam(net.parameters(), lr=t["lr"])
    use_cuda = str(device).startswith("cuda")
    amp_device = "cuda" if use_cuda else "cpu"
    useamp = t["useamp"] and use_cuda
    scaler = torch.amp.GradScaler(amp_device, enabled=useamp)
    # The learning rate is multiplied by lr_rate every 100 steps, until it
    # reaches min_lr, where it is then held.
    lr_maker = lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=t["lr_rate"])
    writer = make_writer(os.path.join(run_dir, "tensorboard"), t["tensorboard"])

    def make_loader(path):
        ds, n = make_dataset(path, (width, height), t["bprop"], channels[0],
                             d["color_space"], d["root"])
        loader = DataLoader(ds, batch_size=t["batchsize"], shuffle=t["shuffle"],
                            num_workers=t["num_workers"])
        return ds, n, loader

    logf = open(os.path.join(run_dir, "log.csv"), "w")
    logf.write("frames,loss\n")

    # "count" is the number of frames seen so far; it drives saving and
    # stopping. Each sample holds bprop frames.
    count = 0
    seq = 0
    img_dataset, n_frames, data_loader = make_loader(lists[0])
    pred = None
    while count <= t["period"]:
        if seq > 0:
            # With several lists, a new dataset is built for each one in turn.
            img_dataset, n_frames, data_loader = make_loader(lists[seq])
        if n_frames == 0:
            print("Not found images.")
            break
        fn = 0
        bar = tqdm(data_loader, unit="batch", desc="list %d" % seq)
        for data in bar:
            with torch.amp.autocast(amp_device, enabled=useamp):
                data = data.to(device)
                pred, errors, _ = net(data)
                if t["loss"] == "corr_wise":
                    mean_error = loss_fn(pred, data[:, -1])
                elif t["loss"] == "ensemble":
                    mean_error = loss_fn(pred, data[:, -1]) + errors.mean()
                else:
                    mean_error = errors.mean()

            optimizer.zero_grad()
            scaler.scale(mean_error).backward()
            scaler.step(optimizer)
            scaler.update()
            if lr_maker.get_last_lr()[0] > t["min_lr"]:
                lr_maker.step()
            else:
                lr_maker.optimizer.param_groups[0]['lr'] = t["min_lr"]

            if t["save_images"]:
                for j in range(len(data)):
                    stem = os.path.join(img_dir, str(count) + '_' + str(fn / t["bprop"] + j))
                    write_image(data[j, -1].detach().cpu().numpy(), stem + 'x',
                                img_dataset.mode, d["color_space"])
                    write_image(pred[j].detach().cpu().numpy(), stem + 'y',
                                img_dataset.mode, d["color_space"])

            loss_value = mean_error.detach().cpu().numpy()
            bar.set_postfix(frames=count, loss=float(loss_value))
            logf.write(str(count) + ',' + str(loss_value) + '\n')
            logf.flush()
            if writer is not None:
                writer.add_scalar("loss", loss_value, count)

            # Save when count has just passed a multiple of save_every.
            if count % t["save_every"] < len(data) * t["bprop"]:
                save_weights(net, os.path.join(ckpt_dir, str(count) + ".pth"))
                if writer is not None:
                    for name, param in net.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), count)
                    write_outputs(writer, net.outputs, count)

            if count > t["period"]:
                break
            count += len(data) * t["bprop"]
            fn += len(data) * t["bprop"]
        seq = (seq + 1) % len(lists)
    logf.close()

    if writer is not None:
        dummy_input = torch.zeros((1, 2, channels[0], height, width)).to(device)
        net.output_mode = 'prediction'
        writer.add_graph(net, dummy_input)
        net.output_mode = 'error'
        writer.close()
    if t["save_graph"] and pred is not None:
        from torchviz import make_dot
        dot = make_dot(pred, params=dict(net.named_parameters()))
        with open(os.path.join(run_dir, "model.dot"), "w") as f:
            f.write(dot.source)
    print("Done. Run folder:", run_dir)
    return run_dir
