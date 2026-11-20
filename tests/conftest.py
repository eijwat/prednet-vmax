"""Skip the tests when the data folder (not part of the repository) is missing."""
import os

import pytest

DATA = os.environ.get("PREDNET_DATA", os.path.expanduser("~/prednet_data"))


def pytest_collection_modifyitems(config, items):
    if os.path.isdir(os.path.join(DATA, "baseline")):
        return
    skip = pytest.mark.skip(reason="PREDNET_DATA folder with the test data not found: " + DATA)
    for item in items:
        item.add_marker(skip)
