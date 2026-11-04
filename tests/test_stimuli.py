"""Check the stimuli catalog against the old list files it was made from.

Run from the project root:

    python3 tests/test_stimuli.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from prednet import stimuli as S  # noqa: E402

DATA = "/mnt/buffalo/prednet_data"
TEST_IMAGES = os.path.join(DATA, "test_images")
CATALOG = os.path.join(TEST_IMAGES, "stimuli.yaml")
LEGACY = os.path.join(TEST_IMAGES, "legacy_lists")


def test_catalog_matches_every_legacy_list():
    catalog = S.load_catalog(CATALOG)
    sources = {S.list_name(f): os.path.join(LEGACY, f) for f in os.listdir(LEGACY)}
    assert set(catalog) == set(sources)
    # The old lists say "test_images/...", relative to the data folder.
    S.check_catalog_against_lists(catalog, sources, root=DATA, catalog_dir=TEST_IMAGES)


def test_every_frame_exists():
    catalog = S.load_catalog(CATALOG)
    for name in catalog:
        for e in S.expand(catalog, name, TEST_IMAGES):
            assert os.path.exists(e[0]), (name, e[0])


def test_repeat_and_cycle_notation():
    catalog = {"a": {"frames": ["x.jpg*2", "y.jpg"], "length": 7}}
    got = [os.path.basename(e[0]) for e in S.expand(catalog, "a", "/d")]
    assert got == ["x.jpg", "x.jpg", "y.jpg", "x.jpg", "x.jpg", "y.jpg", "x.jpg"]
    assert S.compress(["a", "a", "a", "b", "a"]) == ["a*3", "b", "a"]
    assert S.unroll(S.compress(["a", "a", "b"])) == ["a", "a", "b"]


def test_benham_entries():
    catalog = S.load_catalog(CATALOG)
    assert catalog["bht"] == {"frames": ["bht/0.jpg", "bht/1.jpg", "bht/2.jpg", "bht/3.jpg"],
                              "length": 24}
    assert catalog["bht_static"] == {"frames": ["bht/3.jpg"], "length": 24}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("all %d tests passed" % len(tests))
