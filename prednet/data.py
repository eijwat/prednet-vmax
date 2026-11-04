"""Reading frame lists, images and HDF5 archives for training and prediction.

Frame lists
-----------
A list file (e.g. ``train_list.txt``) has one sample per line. A line is
either a single path, or ``input_path,target_path`` separated by a comma
(the target is then used as the frame to predict instead of the next input).

Paths in a list are resolved by ``load_list``:

1. absolute paths are used as they are;
2. if ``root`` is given, paths are joined to it (the legacy ``--root`` rule);
3. otherwise they are taken relative to the folder that contains the list;
4. if that file does not exist, leading folders are dropped one at a time
   (``a/b/c.jpg`` -> ``b/c.jpg`` -> ``c.jpg``), again relative to the list's
   folder, and the first one that exists is used.

Rule 4 makes old lists work after the frames were moved. Lists written by
the original scripts hold paths relative to the old working folder:
``data_e/frame_00000000.jpg`` (list inside the frame folder) is found as
``frame_00000000.jpg``, and ``test_images/bht/0.jpg`` (list next to the
frame folder) is found as ``bht/0.jpg``. Backslashes count as folder
separators, for lists written on Windows.

Samples
-------
Both dataset classes return arrays of shape (input_len + 1, C, H, W): the
``input_len`` input frames followed by the target frame. Samples do not
overlap: sample k uses frames k*input_len .. (k+1)*input_len.
"""

import os
from io import BytesIO, IOBase

import h5py
import numpy as np
from PIL import Image, ImageCms
from torch.utils.data import Dataset

# File extensions read as images; anything else is read with np.load
# (spectrogram ".npy" data from generate_spectrum.py).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

_srgb_profile = ImageCms.createProfile("sRGB")
_lab_profile = ImageCms.createProfile("LAB")
_rgb2lab = ImageCms.buildTransformFromOpenProfiles(
    _srgb_profile, _lab_profile, "RGB", "LAB")


def is_image_path(path):
    return os.path.splitext(path)[-1].lower() in IMAGE_EXTENSIONS


def resolve_path(path, list_dir, root=None):
    """Resolve one path from a list file (see the module docstring)."""
    # Some old lists were written on Windows ("test_images\\x/0.png").
    path = path.replace("\\", "/")
    if os.path.isabs(path):
        return path
    if root is not None:
        return os.path.join(root, path)
    parts = os.path.normpath(path).split("/")
    for i in range(len(parts)):
        candidate = os.path.join(list_dir, *parts[i:])
        if os.path.exists(candidate):
            return candidate
    # Nothing exists: return the full form so the error message shows it.
    return os.path.join(list_dir, path)


def load_list(path, root=None):
    """Read a list file into ``[[input_path], ...]`` or ``[[input, target], ...]``.

    Empty lines are skipped.
    """
    list_dir = os.path.dirname(os.path.abspath(path))
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append([resolve_path(p, list_dir, root) for p in line.split(",")])
    return entries


def read_image(path, size, ch, c_space="RGB"):
    """Read one frame as a float32 array (ch, H, W) scaled to [0, 1].

    path    : file path, or an open binary file object (used for HDF5 data)
    size    : (width, height) to resize to
    ch      : 1 = grayscale, 3 = colour, 4 = colour + grayscale as 4th channel
    c_space : "RGB" (default), "LAB", or any PIL mode such as "HSV", "YCbCr"

    Non-image paths are loaded with np.load and returned unchanged.
    """
    if not (isinstance(path, IOBase) or is_image_path(path)):
        return np.load(path)

    img = Image.open(path)
    img_resize = img.resize(size)
    w, h = size
    if c_space == "RGB":
        if ch == 1:
            img_gray = img_resize.convert("L")
            image = np.asarray(img_gray).reshape((h, w, 1)).transpose(2, 0, 1).astype(np.float32)
        elif ch == 4:
            img_gray = img_resize.convert("L")
            color_array = np.asarray(img_resize).transpose(2, 0, 1).astype(np.float32)
            gray_array = np.asarray(img_gray).reshape((h, w, 1)).transpose(2, 0, 1).astype(np.float32)
            image = np.concatenate([color_array, gray_array], 0)
        else:
            image = np.asarray(img_resize).transpose(2, 0, 1).astype(np.float32)
    elif c_space == "LAB":
        img_conv = ImageCms.applyTransform(img_resize, _rgb2lab)
        image = np.asarray(img_conv).transpose(2, 0, 1).astype(np.float32)
    else:
        img_conv = img_resize.convert(c_space)
        image = np.asarray(img_conv).transpose(2, 0, 1).astype(np.float32)
    image /= 255
    return image


class ImageListDataset(Dataset):
    """Non-overlapping clips of ``input_len + 1`` frames from a list file.

    img_size : (width, height) of the network input
    """

    def __init__(self, img_size=(160, 128), input_len=20, channels=3):
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.input_len = input_len
        self.img_ch = channels
        self.img_paths = None
        # "img" or "audio" (npy); decides how outputs are written back.
        self.mode = None
        self.c_space = "RGB"

    def load_images(self, img_paths, c_space="RGB"):
        """img_paths: the output of ``load_list``."""
        self.img_paths = img_paths
        self.mode = "img" if is_image_path(self.img_paths[0][0]) else "audio"
        self.c_space = c_space

    def _read(self, path):
        return read_image(path, (self.img_w, self.img_h), self.img_ch, self.c_space)

    def __getitem__(self, index):
        assert self.img_paths is not None
        start = index * self.input_len
        end = (index + 1) * self.input_len
        X = np.ndarray((1, self.input_len, self.img_ch, self.img_h, self.img_w), dtype=np.float32)
        X[0] = [self._read(p[0]) for p in self.img_paths[start:end]]
        if len(self.img_paths[0]) > 1:
            # Paired list: the target is the 2nd column of the last input line.
            y = np.array([[self._read(self.img_paths[end - 1][1])]])
        else:
            # Single list: the target is simply the next frame.
            y = np.array([[self._read(self.img_paths[end][0])]])
        return np.concatenate([X, y], axis=1).reshape(
            self.input_len + 1, self.img_ch, self.img_h, self.img_w)

    def __len__(self):
        # Number of clip starts 0, L, 2L, ... that still leave a target frame.
        return len(self.img_paths[:-self.input_len:self.input_len])


class ImageHDF5Dataset(Dataset):
    """Same as ImageListDataset, but frames are stored in one HDF5 file.

    Each HDF5 dataset holds the raw bytes of one image file; keys are the
    file names (see legacy/generate_h5.py). Frames are taken in key order.
    """

    def __init__(self, img_size=(160, 128), input_len=20, channels=3):
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.input_len = input_len
        self.img_ch = channels
        self.hf_data = None
        self.keys = None
        self.mode = None
        self.c_space = "RGB"

    def load_hdf5(self, h5path, c_space="RGB"):
        self.hf_data = h5py.File(h5path, "r")
        self.keys = list(self.hf_data.keys())
        self.mode = "img" if is_image_path(self.keys[0]) else "audio"
        self.c_space = c_space

    def _read(self, key):
        return read_image(BytesIO(np.array(self.hf_data[key])),
                          (self.img_w, self.img_h), self.img_ch, self.c_space)

    def __getitem__(self, index):
        assert self.hf_data is not None
        start = index * self.input_len
        end = (index + 1) * self.input_len
        X = np.ndarray((1, self.input_len, self.img_ch, self.img_h, self.img_w), dtype=np.float32)
        X[0] = [self._read(k) for k in self.keys[start:end]]
        y = np.array([[self._read(self.keys[end])]])
        return np.concatenate([X, y], axis=1).reshape(
            self.input_len + 1, self.img_ch, self.img_h, self.img_w)

    def __len__(self):
        return len(self.keys[:-self.input_len:self.input_len])


def make_dataset(source, img_size, input_len, channels, c_space="RGB", root=None):
    """Build the right dataset for a list file, an .h5/.hdf5 file, or entries.

    source : path of a list file or HDF5 file, or an already loaded list of
             entries (``[[path], ...]``, e.g. from prednet.stimuli)
    Returns (dataset, n_entries) where n_entries is the number of frames.
    """
    if isinstance(source, str) and source.endswith((".h5", ".hdf5")):
        ds = ImageHDF5Dataset(img_size=img_size, input_len=input_len, channels=channels)
        ds.load_hdf5(source, c_space=c_space)
        return ds, len(ds.keys)
    entries = load_list(source, root) if isinstance(source, str) else source
    ds = ImageListDataset(img_size=img_size, input_len=input_len, channels=channels)
    if entries:
        ds.load_images(img_paths=entries, c_space=c_space)
    return ds, len(entries)
