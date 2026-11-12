"""Check prednet.data against legacy/dataset.py, and the new path rules.

Run from the project root:

    python3 tests/test_data_equivalence.py
"""

import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from prednet.data import ImageListDataset, load_list, make_dataset  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "legacy_dataset", os.path.join(ROOT, "legacy", "dataset.py"))
legacy_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy_dataset)

DATA = "/home/deepimac/Claude/prednet_data"
SAMPLE = os.path.join(DATA, "frames", "sample")
EGO = os.path.join(DATA, "frames", "ego4d_00")


def legacy_entries(list_path):
    """What legacy main.py would load, with the frames found by file name."""
    return [[os.path.join(os.path.dirname(list_path), os.path.basename(p))]
            for p in open(list_path).read().split()]


def test_old_lists_resolve_after_move():
    # The lists still say "data/..." and "data_e/...", but must be found.
    for folder, n in [(SAMPLE, 21), (EGO, 46201)]:
        entries = load_list(os.path.join(folder, "train_list.txt"))
        assert len(entries) == n
        for e in (entries[0], entries[-1]):
            assert os.path.exists(e[0]), e[0]
            assert os.path.dirname(e[0]) == folder


def test_same_samples_as_legacy():
    lst = os.path.join(SAMPLE, "train_list.txt")
    for ch, c_space in [(3, "RGB"), (1, "RGB"), (4, "RGB"), (3, "LAB"), (3, "HSV")]:
        old = legacy_dataset.ImageListDataset(img_size=(160, 120), input_len=10, channels=ch)
        old.load_images(img_paths=legacy_entries(lst), c_space=c_space)
        new, n = make_dataset(lst, (160, 120), 10, ch, c_space)
        assert n == 21
        assert len(old) == len(new) == 2
        assert old.mode == new.mode == "img"
        for i in range(len(new)):
            a, b = old[i], new[i]
            assert a.shape == b.shape == (11, ch, 120, 160)
            assert np.array_equal(a, b), "sample %d differs (%d, %s)" % (i, ch, c_space)


def test_lists_next_to_frame_folder():
    # Old lists such as test_images/bht_test_list.txt sat next to the frame
    # folder and said "test_images/bht/0.jpg"; one used a Windows backslash.
    # (The real old lists are now checked in tests/test_stimuli.py.)
    import tempfile
    tmp = tempfile.mkdtemp(prefix="prednet_list_test_")
    try:
        os.symlink(os.path.join(DATA, "test_images", "bht"), os.path.join(tmp, "bht"))
        lst = os.path.join(tmp, "bht_test_list.txt")
        with open(lst, "w") as f:
            f.write("test_images/bht/0.jpg\ntest_images\\bht/1.jpg\n")
        entries = load_list(lst)
        assert entries == [[os.path.join(tmp, "bht", "0.jpg")], [os.path.join(tmp, "bht", "1.jpg")]]
        assert all(os.path.exists(e[0]) for e in entries)
    finally:
        import shutil
        shutil.rmtree(tmp)


def test_root_option_keeps_legacy_rule():
    entries = load_list(os.path.join(SAMPLE, "train_list.txt"), root="/some/root")
    assert entries[0] == ["/some/root/data/frame_00000.jpg"]


def test_empty_list():
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "prednet_empty_list.txt")
    open(path, "w").close()
    ds, n = make_dataset(path, (160, 120), 10, 3)
    assert n == 0 and isinstance(ds, ImageListDataset)
    os.remove(path)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("all %d tests passed" % len(tests))
