# PredNet in PyTorch

Kenta Tanaka, Manabu Kosaka & Eiji Watanabe, 2021. Refactored 2026.

PredNet learns to predict the next frame of a video. This is our PyTorch
port, reorganised so that every experiment is one config file and every run
gets its own folder.

- Original PredNet: https://coxlab.github.io/prednet/
- Illusory motion reproduced by PredNet: https://doi.org/10.3389/fpsyg.2018.00345

## Layout

```
prednet_refactor/
  prednet/        the code (a Python package)
    model.py      PredNet network
    layers.py     ConvLSTMCell, SatLU, AddSin
    data.py       frame lists, images, HDF5
    train.py      training
    predict.py    prediction
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
/mnt/buffalo/prednet_data/
  videos/         source videos
  frames/<name>/  frames + train_list.txt / test_list.txt
  weights/init/   initial weights (init4.pth, ...)
  weights/legacy_models/  models trained with the original code
  baseline/       reference outputs of the original code (for tests)
/mnt/buffalo/prednet_runs/
  <date>_<time>_<name>/   one folder per training run
```

## Quick start

All commands are run from the project folder:

```
cd ~/Claude/prednet_refactor
```

1. Cut a video into frames:

   ```
   python3 -m prednet prepare /mnt/buffalo/prednet_data/videos/cat_snake.mp4 \
       --out /mnt/buffalo/prednet_data/frames/cat_snake --width 160 --height 120
   ```

2. Write a config (copy `configs/sample.yaml` and change `name` and the lists).

3. Train:

   ```
   python3 -m prednet train configs/sample.yaml
   ```

   This creates e.g. `/mnt/buffalo/prednet_runs/2026-11-05_1430_sample/`.

4. Predict with the latest checkpoint of that run:

   ```
   python3 -m prednet predict /mnt/buffalo/prednet_runs/2026-11-05_1430_sample
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

Training on several lists in turn (the old `sequence_list.txt`):

```yaml
data:
  train_list:
    - /mnt/buffalo/prednet_data/frames/a/train_list.txt
    - /mnt/buffalo/prednet_data/frames/b/train_list.txt
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
    --weights /mnt/buffalo/prednet_data/weights/legacy_models/1100000.pth \
    --list /mnt/buffalo/prednet_data/frames/sample/test_list.txt --out /tmp/pred
```

## Tests

The new code is checked against the original one: same weights, same losses,
byte-identical prediction images.

```
python3 tests/test_model_equivalence.py   # ~40 s
python3 tests/test_data_equivalence.py    # ~2 s
python3 tests/test_train_predict.py       # ~70 s
```

## Changes from the original code

- Each run gets its own folder; nothing is written into the current folder.
- Settings live in YAML configs; the used config is saved with the run.
- Weights saved on a GPU now load on a CPU-only machine.
- Prediction uses the run's own model options (round_mode, amp, omg were
  ignored by the original `--test`).
- List files are read relative to their own folder, so frames can be moved.
- `.png` frames are read as images.
- Not ported yet (still in `legacy/`): audio spectrograms
  (`generate_spectrum.py`), batch scripts (`auto_*.py`), weight tools
  (`csv_serializer.py`, `weight_editor.py`), RAFT / PWC-Net flow.

---

# PredNet in PyTorch（日本語）

田中健太・小坂学・渡辺英治, 2021年。2026年に整理し直しました。

PredNetは、動画の次のフレームを予測することを学習します。これは私たちがPyTorchに移植したもので、1つの実験を1つの設定ファイルで表し、1回の学習ごとに専用のフォルダを作る形に整理しました。

- 元のPredNet: https://coxlab.github.io/prednet/
- PredNetで錯視の動きが再現される研究: https://doi.org/10.3389/fpsyg.2018.00345

## フォルダの構成

```
prednet_refactor/
  prednet/        コード（Pythonパッケージ）
    model.py      PredNetのネットワーク本体
    layers.py     部品（ConvLSTMCell, SatLU, AddSin）
    data.py       フレーム一覧・画像・HDF5の読み込み
    train.py      学習
    predict.py    予測
    prepare.py    動画をフレームに分ける
    config.py     設定の初期値と読み込み
    io.py         学習フォルダ・重み・画像の保存
    losses/       CorrWise損失（オプティカルフローを使う）
  configs/        実験ごとのYAML設定ファイル
  tests/          元のコードと同じ結果になるかの確認
  legacy/         元のスクリプト（参照用に保存）
```

データと結果は、プロジェクトの中ではなく外付けHDDに置きます。

```
/mnt/buffalo/prednet_data/
  videos/         元の動画
  frames/<名前>/  フレーム画像と train_list.txt / test_list.txt
  weights/init/   初期の重み（init4.pth など）
  weights/legacy_models/  元のコードで学習したモデル
  baseline/       元のコードの出力（テスト用の基準）
/mnt/buffalo/prednet_runs/
  <日付>_<時刻>_<名前>/   学習1回ごとのフォルダ
```

## 使い方

コマンドは、すべてプロジェクトのフォルダで実行します。

```
cd ~/Claude/prednet_refactor
```

1. 動画をフレームに分けます。

   ```
   python3 -m prednet prepare /mnt/buffalo/prednet_data/videos/cat_snake.mp4 \
       --out /mnt/buffalo/prednet_data/frames/cat_snake --width 160 --height 120
   ```

2. 設定ファイルを作ります（`configs/sample.yaml` をコピーして、`name` と一覧のパスを変えます）。

3. 学習します。

   ```
   python3 -m prednet train configs/sample.yaml
   ```

   たとえば `/mnt/buffalo/prednet_runs/2026-11-05_1430_sample/` が作られます。

4. その学習の最新のモデルで予測します。

   ```
   python3 -m prednet predict /mnt/buffalo/prednet_runs/2026-11-05_1430_sample
   ```

## 設定ファイル

設定ファイルには、初期値と違う値だけを書けば大丈夫です。すべての設定と初期値を見るには、次を実行します。

```
python3 -m prednet config configs/sample.yaml
```

初期値と各項目の意味は `prednet/config.py` に書いてあります。存在しない項目を書くとエラーになるので、書き間違いに気づけます。

ファイルを書き換えずに、その回だけ値を変えたいときは `--set` を使います。

```
python3 -m prednet train configs/ego4d.yaml --set train.period=40000 --set train.lr_rate=0.9
```

複数の一覧を順番に学習するとき（以前の `sequence_list.txt`）は、次のように書きます。

```yaml
data:
  train_list:
    - /mnt/buffalo/prednet_data/frames/a/train_list.txt
    - /mnt/buffalo/prednet_data/frames/b/train_list.txt
```

## 学習フォルダの中身

```
config.yaml       実際に使った設定（初期値もすべて入っています）
log.csv           ステップごとの frames,loss
checkpoints/      train.save_every フレームごとに保存した <フレーム数>.pth
predictions/<チェックポイント>_<フレームのフォルダ>_<一覧>/
    test_<n>x.jpg     入力フレーム n
    test_<n>y_0.jpg   n の次のフレームの予測
    test_<n>y_<k>.jpg 予測をもとにさらに k フレーム先まで予測したもの
    log.csv           フレームごとの各層の評価値（MSE）
```

学習フォルダが使い回されることはありません。同じ設定でもう一度学習すると、新しいフォルダができます。

## 予測のオプション

```
python3 -m prednet predict 学習フォルダ --checkpoint 100000      # チェックポイントを指定
python3 -m prednet predict 学習フォルダ --list path/to/test_list.txt
python3 -m prednet predict 学習フォルダ --set predict.ext=20 --set predict.input_len=10
```

学習フォルダのない古いモデルで予測するときは、次のようにします。

```
python3 -m prednet predict --config configs/ego4d.yaml \
    --weights /mnt/buffalo/prednet_data/weights/legacy_models/1100000.pth \
    --list /mnt/buffalo/prednet_data/frames/sample/test_list.txt --out /tmp/pred
```

## テスト

新しいコードが、元のコードと同じ重み・同じ損失・まったく同じ予測画像になるかを確かめます。

```
python3 tests/test_model_equivalence.py   # 約40秒
python3 tests/test_data_equivalence.py    # 約2秒
python3 tests/test_train_predict.py       # 約70秒
```

## 元のコードから変わったこと

- 学習ごとに専用のフォルダを作ります。今いるフォルダには何も書き出しません。
- 設定はYAMLファイルに書き、使った設定は学習フォルダに保存されます。
- GPUで保存した重みを、GPUのないマシンでも読み込めます。
- 予測のときも、その学習で使ったモデルの設定を使います（元の `--test` では round_mode, amp, omg が使われていませんでした）。
- 一覧ファイルの中のパスを、一覧ファイルが置いてあるフォルダから探すので、フレームを別の場所に移しても動きます。
- `.png` のフレームも画像として読み込めます。
- まだ移していないもの（`legacy/` に残っています）: 音声スペクトログラム（`generate_spectrum.py`）、一括実行スクリプト（`auto_*.py`）、重みの道具（`csv_serializer.py`, `weight_editor.py`）、RAFT / PWC-Net のフロー。
