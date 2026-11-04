"""Cut a video into frames (``python3 -m prednet prepare VIDEO --out DIR``).

Writes ``<prefix>_00000000.jpg``, ``<prefix>_00000001.jpg``, ... into DIR,
plus two list files in the same folder:

    train_list.txt   the first (1 - test_ratio) of the frames
    test_list.txt    the remaining frames at the end

The lists contain bare file names, so the folder can be moved freely.

This is the replacement for the original generate_image_s.py (git 1e000af)
and makes the same frames. Like that script it skips the very first frame
of the video, so frame numbers match frame sets made earlier (e.g.
frames/ego4d_00).
"""

import os
import sys

import cv2


def prepare(video, out_dir, width=-1, height=-1, test_ratio=0.1, prefix="frame"):
    """Extract frames of ``video`` into ``out_dir``. Returns the frame count.

    width  > 0 : resize to this width; the height follows the aspect ratio
                 unless ``height`` > 0 is also given.
    """
    os.makedirs(out_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video)
    if not vidcap.isOpened():
        raise FileNotFoundError("Cannot open video: " + video)
    vidcap.read()  # legacy behaviour: the first frame is dropped
    names = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        name = "%s_%08d.jpg" % (prefix, len(names))
        if width > 0:
            h = height if height > 0 else int(image.shape[0] * float(width) / image.shape[1])
            image = cv2.resize(image, (width, h))
        cv2.imwrite(os.path.join(out_dir, name), image)
        names.append(name)
        sys.stdout.write("\rSaved %d frames" % len(names))
        sys.stdout.flush()
    print()

    ratio = max(0.0, min(1.0, test_ratio))
    split = int(len(names) * (1.0 - ratio))
    for list_name, part in (("train_list.txt", names[:split]), ("test_list.txt", names[split:])):
        with open(os.path.join(out_dir, list_name), "w") as f:
            f.write("".join(n + "\n" for n in part))
        print("Wrote %s (%d frames)" % (os.path.join(out_dir, list_name), len(part)))
    return len(names)
