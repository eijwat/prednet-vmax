"""Test stimuli: a catalog of named frame sequences (illusions, test movies).

All stimuli are described in one YAML file, by default
``/mnt/buffalo/prednet_data/test_images/stimuli.yaml``. Each entry is

    bht_rev:                      # the name used on the command line
      frames: [bht/2.jpg, bht/1.jpg, bht/0.jpg, bht/3.jpg]
      length: 24                  # optional: repeat frames up to this many
      note: Benham top, reversed  # optional free text

``frames`` are relative to the catalog's folder. ``path*N`` is short for
the same frame N times in a row (e.g. ``FW/1.jpg*10``). Without ``length``
the frames are used once, in order. With ``length`` the frame list is
cycled: frame i of the sequence is ``frames[i % len(frames)]``.

Command line:

    python3 -m prednet stimuli                 # list all names
    python3 -m prednet stimuli bht bht_rev     # show the expanded frames
    python3 -m prednet stimuli --convert DIR   # build a catalog from old lists

The old one-file-per-sequence lists (``*test_list.txt``) can be turned into
a catalog with ``catalog_from_lists``; ``--convert`` also checks that every
converted entry expands to exactly the same frames as its list.
"""

import os
import re

import yaml

DEFAULT_CATALOG = "/mnt/buffalo/prednet_data/test_images/stimuli.yaml"
LIST_SUFFIX = "test_list.txt"
_REPEAT = re.compile(r"^(.*)\*(\d+)$")


def load_catalog(path=DEFAULT_CATALOG):
    with open(path) as f:
        catalog = yaml.safe_load(f) or {}
    for name, entry in catalog.items():
        if not entry.get("frames"):
            raise ValueError("Stimulus %s has no frames" % name)
    return catalog


def unroll(frames):
    """Expand ``path*N`` items into N copies of ``path``."""
    out = []
    for item in frames:
        m = _REPEAT.match(item)
        out.extend([m.group(1)] * int(m.group(2)) if m else [item])
    return out


def compress(frames):
    """Inverse of unroll: runs of the same frame become ``path*N``."""
    out = []
    i = 0
    while i < len(frames):
        j = i
        while j < len(frames) and frames[j] == frames[i]:
            j += 1
        out.append(frames[i] if j - i == 1 else "%s*%d" % (frames[i], j - i))
        i = j
    return out


def expand(catalog, name, catalog_dir):
    """Absolute frame paths of stimulus ``name``, as load_list entries."""
    if name not in catalog:
        raise KeyError("Unknown stimulus %r. Known: %s" % (name, ", ".join(sorted(catalog))))
    entry = catalog[name]
    frames = unroll(entry["frames"])
    length = entry.get("length", len(frames))
    return [[os.path.join(catalog_dir, frames[i % len(frames)])] for i in range(length)]


def stimulus_entries(name, catalog_path=DEFAULT_CATALOG):
    """Shortcut: load the catalog and expand one stimulus."""
    return expand(load_catalog(catalog_path), name, os.path.dirname(os.path.abspath(catalog_path)))


# ---------------------------------------------------------------- conversion

def _shortest_cycle(seq):
    """Smallest p such that seq[i] == seq[i % p] for all i."""
    for p in range(1, len(seq) + 1):
        if all(seq[i] == seq[i % p] for i in range(len(seq))):
            return p
    return len(seq)


def _relative_frames(list_path, catalog_dir, root=None):
    """Frames of an old list, relative to catalog_dir, via prednet.data rules."""
    from .data import load_list
    rel = []
    for entry in load_list(list_path, root):
        if len(entry) != 1:
            raise ValueError("Paired lists are not supported here: " + list_path)
        rel.append(os.path.relpath(entry[0], catalog_dir))
    return rel


def list_name(fname):
    """Stimulus name of an old list file: bht_test_list.txt -> bht,
    ibi_vgatest_list.txt -> ibi_vga."""
    return fname[:-len(LIST_SUFFIX)].rstrip("_")


def catalog_from_lists(list_dir, catalog_dir=None, root=None):
    """Build a catalog dict from every ``*test_list.txt`` in ``list_dir``.

    Returns (catalog, sources) where sources maps each name to its list file.
    A list that repeats a short cycle of frames becomes ``frames`` (one cycle)
    plus ``length``; any other list keeps all its frames. Runs of one frame
    are written as ``path*N``.
    """
    catalog_dir = catalog_dir or list_dir
    catalog, sources = {}, {}
    for fname in sorted(os.listdir(list_dir)):
        if not fname.endswith(LIST_SUFFIX):
            continue
        name = list_name(fname)
        sources[name] = os.path.join(list_dir, fname)
        frames = _relative_frames(sources[name], catalog_dir, root)
        p = _shortest_cycle(frames)
        if p < len(frames):
            catalog[name] = {"frames": compress(frames[:p]), "length": len(frames)}
        else:
            catalog[name] = {"frames": compress(frames)}
    return catalog, sources


def check_catalog_against_lists(catalog, sources, root=None, catalog_dir=None):
    """Raise if any entry does not expand to the same frames as its old list."""
    from .data import load_list
    for name, list_path in sources.items():
        old = load_list(list_path, root)
        new = expand(catalog, name, catalog_dir or os.path.dirname(list_path))
        if [os.path.realpath(e[0]) for e in old] != [os.path.realpath(e[0]) for e in new]:
            raise AssertionError("Stimulus %s differs from its list" % name)


def save_catalog(catalog, path, header=""):
    """Write a catalog; frame lists are written inline and wrapped."""
    class _Dumper(yaml.SafeDumper):
        pass

    def _list(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
    _Dumper.add_representer(list, _list)
    with open(path, "w") as f:
        f.write(header)
        yaml.dump(catalog, f, Dumper=_Dumper, sort_keys=False, width=100)
