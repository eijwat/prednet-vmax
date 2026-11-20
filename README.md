# PredNet in PyTorch

Kenta Tanaka, Manabu Kosaka & Eiji Watanabe, 2021. Refactored 2026.

PredNet learns to predict the next frame of a video. This is our PyTorch
port, reorganised so that every experiment is one config file and every run
gets its own folder.

- Original PredNet: https://coxlab.github.io/prednet/
- Illusory motion reproduced by PredNet: https://doi.org/10.3389/fpsyg.2018.00345

## Setup

```
pip3 install -r requirements.txt
export PREDNET_DATA=/path/to/prednet_data   # frames, weights, test images (default ~/prednet_data)
export PREDNET_RUNS=/path/to/prednet_runs   # run folders are created here (default ~/prednet_runs)
```

Config files can refer to the two folders as `${PREDNET_DATA}` and
`${PREDNET_RUNS}`. The data folder is not part of this repository; the tests
need it and are skipped when it is missing.

## Citation

If you use this code, please cite:

> Ueda K, Sinapayen L, Watanabe E. Predictive networks generate motion-induced
> color illusions. *Scientific Reports* 16, 27582 (2026).
> https://doi.org/10.1038/s41598-026-57953-w

This repository is a refactored version of the code used in that study. See
also `CITATION.cff`.

Licensed under the MIT License (see `LICENSE`). Third-party sources are listed
in `THIRD_PARTY_NOTICES.md`.

## Layout

```
prednet_refactor/
  prednet/        the code (a Python package)
    model.py      PredNet network
    layers.py     ConvLSTMCell, SatLU, AddSin
    data.py       frame lists, images, HDF5
    train.py      training
    predict.py    prediction
    batch.py      repeated training + testing on stimuli
    stimuli.py    catalog of test stimuli
    prepare.py    video -> frames
    config.py     config defaults and loading
    io.py         run folders, checkpoints, images
    losses/       CorrWise loss (optical-flow based)
  configs/        one YAML file per experiment
  tests/          checks against the original code
  legacy/         the original scripts, kept for reference
```

Data and results are kept on the external disk, not in the project:

```
$PREDNET_DATA/
  videos/         source videos
  frames/<name>/  frames + train_list.txt / test_list.txt
  weights/init/   initial weights (init4.pth, ...)
  weights/legacy_models/  models trained with the original code
  test_images/    test stimuli (illusions etc.)
    stimuli.yaml  the catalog: every stimulus by name
    legacy_lists/ the old *_test_list.txt files (no longer needed)
  baseline/       reference outputs of the original code (for tests)
  old_outputs/    old prediction results found among the stimuli
$PREDNET_RUNS/
  <date>_<time>_<name>/   one folder per training run or batch
```

## Quick start

All commands are run from the project folder:

```
cd ~/Claude/prednet_refactor
```

1. Cut a video into frames:

   ```
   python3 -m prednet prepare $PREDNET_DATA/videos/cat_snake.mp4 \
       --out $PREDNET_DATA/frames/cat_snake --width 160 --height 120
   ```

2. Write a config (copy `configs/sample.yaml` and change `name` and the lists).

3. Train:

   ```
   python3 -m prednet train configs/sample.yaml
   ```

   This creates e.g. `$PREDNET_RUNS/2026-11-05_1430_sample/`.

4. Predict with the latest checkpoint of that run:

   ```
   python3 -m prednet predict $PREDNET_RUNS/2026-11-05_1430_sample
   ```

## Configs

A config only needs the values that differ from the defaults. To see every
setting with its default value:

```
python3 -m prednet config configs/sample.yaml
```

The defaults and the meaning of each key are listed in `prednet/config.py`.
Unknown keys are reported as errors, so typos do not pass silently.

To change a value for one run without editing the file, use `--set`:

```
python3 -m prednet train configs/ego4d.yaml --set train.period=40000 --set train.lr_rate=0.9
```

`train.flush_denormal` (default `true`) flushes denormal floats to zero on the
CPU. Without it CPU training slows down 5-10x within a few steps (about 6 s ->
45 s per 20-frame batch on the imac), because gradients and activations become
denormal numbers, which the CPU computes very slowly. The GPU is not affected.
Set it to `false` only to reproduce the old numerics exactly.

Training on several lists in turn (the old `sequence_list.txt`):

```yaml
data:
  train_list:
    - $PREDNET_DATA/frames/a/train_list.txt
    - $PREDNET_DATA/frames/b/train_list.txt
```

## What a run folder contains

```
config.yaml       the full config used (all defaults filled in)
log.csv           frames,loss for every step
checkpoints/      <frames>.pth every train.save_every frames
predictions/<checkpoint>_<frames>_<list>/
    test_<n>x.jpg     input frame n
    test_<n>y_0.jpg   prediction of the frame after n
    test_<n>y_<k>.jpg extended prediction, k frames further ahead
    log.csv           eval index (MSE) of every layer per frame
```

A run folder is never reused: running the same config again makes a new one.

## Prediction options

```
python3 -m prednet predict RUN_DIR --checkpoint 100000      # a given checkpoint
python3 -m prednet predict RUN_DIR --list path/to/test_list.txt
python3 -m prednet predict RUN_DIR --set predict.ext=20 --set predict.input_len=10
```

Old models without a run folder:

```
python3 -m prednet predict --config configs/ego4d.yaml \
    --weights $PREDNET_DATA/weights/legacy_models/1100000.pth \
    --list $PREDNET_DATA/frames/sample/test_list.txt --out /tmp/pred
```

## Test stimuli

All test stimuli are named in one file,
`$PREDNET_DATA/test_images/stimuli.yaml`:

```yaml
bht:                        # Benham top
  frames: [bht/0.jpg, bht/1.jpg, bht/2.jpg, bht/3.jpg]
  length: 24                # cycle the frames up to 24
bht_static:
  frames: [bht/3.jpg]
  length: 24
FW:
  frames: [FW/1.jpg*22, FW/2.jpg*22]   # "path*N" = that frame N times
```

Paths are relative to `test_images/`. A new stimulus is a few lines here;
an optional `note:` describes it.

```
python3 -m prednet stimuli                    # list all stimuli
python3 -m prednet stimuli bht                # show the frames of one
python3 -m prednet predict RUN_DIR --stimulus bht --stimulus bht_rev
```

Output: `RUN_DIR/predictions/<checkpoint>_<stimulus>/`.

## Batches (repeated training and testing)

`prednet batch` replaces `auto_train_test.py` and `auto_benham.py`. Add a
`batch` section to a config (example: `configs/benham_batch.yaml`):

```yaml
batch:
  repeats: 20                         # train each dataset 20 times
  datasets:                           # name: train list
    ego4d_00: $PREDNET_DATA/frames/ego4d_00/train_list.txt
  stimuli: [bht, bht_rev, bht_static, bhtr, bhtr_rev, bhtr_static]
  checkpoints: [10000, 20000, 30000, 40000]
  collect: [test_19y_0.jpg]           # copied into summary/
```

```
python3 -m prednet batch configs/benham_batch.yaml
python3 -m prednet batch --resume $PREDNET_RUNS/<batch folder>
```

The batch folder holds `batch.yaml`, `status.csv`, one run folder per
`<dataset>/<repeat>/`, and `summary/<dataset>_<repeat>_<stimulus>_<checkpoint>_<file>`.
A stopped batch can be resumed; finished parts (marked `DONE`) are skipped.

Note: with `init_weights` set and `shuffle` off, training on the CPU is
deterministic, so all repeats give the same result. To get different
repeats, set `batch.seeds` (one per repeat) together with
`train.shuffle: true`, or leave `init_weights` empty.

## Tests

The new code is checked against the original one: same weights, same losses,
byte-identical prediction images.

```
python3 -m pytest -q tests                # all, ~6 min
python3 tests/test_model_equivalence.py   # ~40 s
python3 tests/test_data_equivalence.py    # ~2 s
python3 tests/test_stimuli.py             # ~2 s
python3 tests/test_train_predict.py       # ~70 s
python3 tests/test_batch.py               # ~4 min
```

## Changes from the original code

- Each run gets its own folder; nothing is written into the current folder.
- Settings live in YAML configs; the used config is saved with the run.
- Weights saved on a GPU now load on a CPU-only machine.
- Prediction uses the run's own model options (round_mode, amp, omg were
  ignored by the original `--test`).
- List files are read relative to their own folder, so frames can be moved.
- `.png` frames are read as images.
- Test stimuli are one catalog instead of 112 list files.
- Batches are one config and one command, and can be resumed.
- Removed from `legacy/` because they are replaced (still in git history):
  `main.py`, `auto_train_test.py`, `auto_benham.py`, `generate_image_s.py`.
- Not ported yet (still in `legacy/`): audio spectrograms
  (`generate_spectrum.py`), weight tools (`csv_serializer.py`,
  `weight_editor.py`), other helper scripts. (RAFT and PWC-Net optical flow were removed from
  this repository; they are third-party code, see `THIRD_PARTY_NOTICES.md`.)
