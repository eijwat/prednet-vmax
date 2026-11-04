"""Files on disk: run folders, checkpoints, output images and TensorBoard."""

import datetime
import os

import numpy as np
import torch
from PIL import Image, ImageCms

_srgb_profile = ImageCms.createProfile("sRGB")
_lab_profile = ImageCms.createProfile("LAB")
_lab2rgb = ImageCms.buildTransformFromOpenProfiles(
    _lab_profile, _srgb_profile, "LAB", "RGB")


# ---------------------------------------------------------------- run folders

def make_run_dir(runs_dir, name):
    """Create ``<runs_dir>/<YYYY-MM-DD_HHMM>_<name>`` and return its path.

    If that folder already exists (two runs in the same minute), a suffix
    ``_2``, ``_3``, ... is added, so a run never writes into another run.
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    base = os.path.join(runs_dir, stamp + "_" + name)
    path, n = base, 1
    while True:
        try:
            os.makedirs(path)
            return path
        except FileExistsError:
            n += 1
            path = "%s_%d" % (base, n)


# ---------------------------------------------------------------- checkpoints

def load_weights(net, path, device):
    """Load a state_dict into ``net``.

    map_location makes checkpoints saved on a GPU load on a CPU-only machine
    (the original code failed there).
    """
    print("Load model from", path)
    net.load_state_dict(torch.load(path, map_location=device))


def save_weights(net, path):
    torch.save(net.state_dict(), path)


# ---------------------------------------------------------------- images

def write_image(image, path, mode="img", c_space="RGB"):
    """Write one (C, H, W) array in [0, 1].

    ``path`` is given without extension. mode "img" writes JPEG; other modes
    (audio spectrograms) write the raw array as ``.npy``.

    Channel handling for RGB: 1 channel -> grayscale JPEG; 4 channels ->
    ``<path>_color.jpg`` (first 3) and ``<path>_gray.jpg`` (4th); otherwise
    an RGB JPEG. LAB is converted back to sRGB; other colour spaces are
    converted with PIL.
    """
    if mode != "img":
        np.save(path + ".npy", image)
        return
    img = image * 255
    img = img.transpose(1, 2, 0)
    img = img.astype(np.uint8)
    h, w, ch = img.shape
    if c_space == "RGB":
        if ch == 1:
            Image.fromarray(img.reshape((h, w))).save(path + ".jpg")
        elif ch == 4:
            Image.fromarray(img[:, :, 3]).save(path + "_gray.jpg")
            Image.fromarray(img[:, :, :3]).save(path + "_color.jpg")
        else:
            Image.fromarray(img).save(path + ".jpg")
    elif c_space == "LAB":
        tmp = Image.fromarray(img, mode="LAB")
        ImageCms.applyTransform(tmp, _lab2rgb).save(path + ".jpg")
    else:
        Image.fromarray(img, mode=c_space).convert("RGB").save(path + ".jpg")


# ---------------------------------------------------------------- tensorboard

def make_writer(log_dir, enabled):
    """A SummaryWriter in ``log_dir``, or None when disabled."""
    if not enabled:
        return None
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir=log_dir)


def write_outputs(writer, outputs, count, prefix=""):
    """Log the hook-recorded activations (net.outputs) as image grids."""
    import torchvision.utils as vutils
    for k, v in outputs.items():
        for i, vv in enumerate(v):
            if isinstance(vv, torch.Tensor):
                if len(vv.shape) == 3:
                    vv = vv.unsqueeze(axis=1)
                elif len(vv.shape) == 4:
                    vv = vv.reshape((-1, vv.shape[2], vv.shape[3]))
                    vv = vv.unsqueeze(axis=1)
                x = vutils.make_grid(vv, normalize=True, scale_each=True)
                tag = k + "_time{}".format(i)
                writer.add_image(tag if prefix == "" else prefix + "/" + tag, x, count)
