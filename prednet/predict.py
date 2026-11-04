"""Prediction (``python3 -m prednet predict RUN_DIR``).

Usually you point it at a run folder; the model settings are then read from
that run's config.yaml, so prediction always uses the same options as
training (the legacy code ignored round_mode, amp and omg here).

Output goes to ``<run>/predictions/<checkpoint>_<list>/``:

    test_<n>x.jpg     input frame n
    test_<n>y_0.jpg   prediction of the frame after frame n
    test_<n>y_<k>.jpg for the last frame n of each clip: the k-th frame
                      predicted further ahead from the network's own
                      predictions (k = 1 .. predict.ext)
    log.csv           frame, then the eval_index (MSE) of every layer

File names and numbers are the same as in the legacy ``result/`` folder.
"""

import glob
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import make_dataset
from .io import load_weights, make_writer, write_image, write_outputs
from .train import build_model, resolve_device


def latest_checkpoint(run_dir):
    """The checkpoint with the largest frame number in ``run_dir``."""
    paths = glob.glob(os.path.join(run_dir, "checkpoints", "*.pth"))
    if not paths:
        raise FileNotFoundError("No checkpoints in " + run_dir)
    return max(paths, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def resolve_checkpoint(run_dir, checkpoint):
    """None -> latest; a number -> checkpoints/<number>.pth; else a path."""
    if checkpoint is None:
        return latest_checkpoint(run_dir)
    if str(checkpoint).isdigit():
        return os.path.join(run_dir, "checkpoints", str(checkpoint) + ".pth")
    return checkpoint


def predict(cfg, weights, test_list, out_dir):
    """Predict every clip of ``test_list`` with ``weights``, into ``out_dir``.

    test_list : a list file, or entries from prednet.stimuli.
    """
    d, p = cfg["data"], cfg["predict"]
    device = resolve_device(cfg["device"])
    width, height = d["size"]
    ch = cfg["model"]["channels"][0]
    input_len = p["input_len"]
    os.makedirs(out_dir, exist_ok=True)
    print("Output folder:", out_dir)

    net = build_model(cfg, device)
    net.eval()
    load_weights(net, weights, device)

    use_cuda = str(device).startswith("cuda")
    amp_device = "cuda" if use_cuda else "cpu"
    useamp = p["useamp"] and use_cuda
    writer = make_writer(os.path.join(out_dir, "tensorboard"), p["tensorboard"])

    img_dataset, n_frames = make_dataset(test_list, (width, height), input_len, ch,
                                         d["color_space"], d["root"])
    if n_frames == 0:
        print("Not found images.")
        return out_dir
    data_loader = DataLoader(img_dataset, batch_size=p["batchsize"], shuffle=False,
                             num_workers=p["num_workers"])

    def out(name):
        return os.path.join(out_dir, name)

    logf = open(out("log.csv"), "w")
    logf.write("frame," + ",".join("layer%d" % l for l in range(net.n_layers)) + "\n")
    bs = p["batchsize"]
    for i, data in enumerate(tqdm(data_loader, unit="batch")):
        for j in range(len(data)):
            # Clip number within the whole list, and its first frame number.
            clip_start = (i * bs + j) * input_len
            # 1) One-step predictions: feed frames 0..k (at most
            #    num_max_window + 1 of them) and predict frame k + 1.
            for k in range(input_len):
                start_idx = max(0, k - p["num_max_window"]) if p["num_max_window"] is not None else 0
                x_batch = data[j, start_idx:k + 2].view(1, k + 2 - start_idx, ch, height, width)
                with torch.no_grad():
                    with torch.amp.autocast(amp_device, enabled=useamp):
                        pred, errors, eval_index = net(x_batch.to(device))
                n = str(k + clip_start)
                write_image(data[j, k].detach().cpu().numpy(), out('test_' + n + 'x'),
                            img_dataset.mode, d["color_space"])
                write_image(pred[0].detach().cpu().numpy(), out('test_' + n + 'y_0'),
                            img_dataset.mode, d["color_space"])
                if writer is not None:
                    write_outputs(writer, net.outputs, k, "test_%d_%d" % (i, j))
                logf.write(n + "," + ",".join(
                    str(eval_index[l][0].detach().cpu().numpy()) for l in range(net.n_layers)) + "\n")
            # 2) Extended prediction: keep appending the network's own
            #    predictions and predict further ahead. As in the legacy code,
            #    the true last frame (y_batch) is appended after them.
            exts = [pred[0].view(1, 1, ch, height, width)]
            y_batch = data[j, input_len:].view(1, 1, ch, height, width)
            last = str(clip_start + input_len - 1)
            for k in range(p["ext"]):
                with torch.no_grad():
                    pred_ext, _, _ = net(torch.cat([x_batch.to(device)] + exts + [y_batch.to(device)], axis=1))
                exts.append(pred_ext.unsqueeze(0))
                write_image(pred_ext[0].detach().cpu().numpy(), out('test_' + last + 'y_' + str(k + 1)),
                            img_dataset.mode, d["color_space"])
                if writer is not None:
                    write_outputs(writer, net.outputs, k, "test_ext_%d_%d" % (i, j))
    logf.close()
    if writer is not None:
        writer.close()
    print("Done. Output folder:", out_dir)
    return out_dir


def prediction_dir(run_dir, weights, label):
    ckpt_name = os.path.splitext(os.path.basename(weights))[0]
    return os.path.join(run_dir, "predictions", "%s_%s" % (ckpt_name, label))


def predict_run(cfg, run_dir, checkpoint=None, test_list=None, stimulus=None):
    """Predict with a checkpoint of a run; output inside that run folder.

    Give either a stimulus name (from the catalog), a list file, or neither
    (then data.test_list is used). Output folder:
    ``predictions/<checkpoint>_<stimulus>`` or
    ``predictions/<checkpoint>_<frame folder>_<list name>``.
    """
    weights = resolve_checkpoint(run_dir, checkpoint)
    if stimulus:
        from .stimuli import stimulus_entries
        source = stimulus_entries(stimulus, cfg["predict"]["stimuli_catalog"])
        label = stimulus
    else:
        source = test_list or cfg["data"]["test_list"]
        if not source:
            raise ValueError("No test data: pass --stimulus or --list, or set data.test_list")
        list_name = os.path.splitext(os.path.basename(source))[0]
        parent = os.path.basename(os.path.dirname(os.path.abspath(source)))
        label = "%s_%s" % (parent, list_name)
    return predict(cfg, weights, source, prediction_dir(run_dir, weights, label))
