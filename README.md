# Wood workshop YOLO detector

Three classes: **pen** (including pencil), **screwdriver**, and **screw**.
Background photographs confirmed by the dataset owner are kept without `.txt`
files; YOLO treats their absence as no target objects. There is no fourth class.

## Train on Colab GPU

[Open the Colab notebook](https://colab.research.google.com/github/omar-salem23/WoodWorkShop/blob/main/notebooks/train_colab.ipynb).
Choose **Runtime → Change runtime type → GPU**, then **Runtime → Run all**.
Authorize Google Drive when prompted. The notebook downloads and audits the
dataset, trains on GPU, retries when validation is below target, and saves
checkpoints and final metrics in `MyDrive/WoodWorkShop_runs`.
It starts a separate training run; it does not stop or resume the cloud CPU run.
GPU allocation is controlled by Colab and is not guaranteed. If batch 16 exceeds
available GPU memory, reduce it to 8 in the training cell. The notebook has been
syntax-checked; an actual Colab GPU session must be started by the account owner.

## Environment

Python 3.12 and Ultralytics 8.4.177 were checked in this environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For a GPU machine, install the PyTorch build matching its CUDA version instead
of the CPU wheels, and pass `--device 0` when training.

## Download and prepare

The original ZIP is stored outside Git. Download the shared export:

```bash
gdown 1vz95eqHeMBBZoNrfANFlW_Mzv_Tf6LE6 -O dataset.zip
```

Extract it into a directory containing `dataset_split/data.yaml` and the
`train`, `valid`, and `test` directories, each with `images` and `labels`.
Then prepare a **new, empty** destination:

```bash
python prepare_dataset.py /path/to/dataset_split/data.yaml \
  --output /path/to/prepared --missing-labels background
python audit_dataset.py /path/to/prepared
```

Use `--missing-labels background` only when the owner confirms that unlabelled
images contain no target objects. Otherwise the default is to exclude them.
Preparation preserves source files, remaps class IDs, removes zero-area boxes,
and clips boxes extending beyond image edges. Image pixels are not altered;
hard links or byte-for-byte copies are used. Confirmed backgrounds keep their
missing labels; no empty txt files are invented for them.

Identical decoded images are retained once, prioritizing test, then validation,
then training. Each exclusion and annotation repair is recorded in
`preparation_report.json`. This only detects exact pixel duplicates. Similar
frames from the same video may still occur across splits, so evaluation does
not establish generalization to entirely new workshops or camera recordings.

The verified prepared export has:

| Split | Images | Backgrounds | Pen boxes | Screwdriver boxes | Screw boxes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 6,693 | 1,464 | 623 | 1,726 | 16,051 |
| Validation | 1,071 | 402 | 94 | 228 | 2,001 |
| Test | 920 | 250 | 73 | 219 | 2,095 |

Background counts include two existing empty training labels. From 8,846 source
images, 162 duplicate copies were excluded in the prepared version. Twelve
annotation rows were repaired: one zero-area box removed and eleven clipped.
Original files remain available unchanged.

## Train, validate, and test

```bash
python train.py --data /path/to/prepared/data.yaml --model yolo11n.pt \
  --epochs 40 --retry-epochs 60 --imgsz 512 --batch 8 \
  --device cpu --output runs/woodwork
```

Training uses pretrained YOLO11 nano weights, online augmentation, a fixed
random seed and early stopping. Use a fresh output folder for each experiment.
The initial stage runs up to 40 epochs. If validation is below the configured
targets, a second stage trains from the best initial weights for up to 60 more
epochs. Checkpoint selection uses validation, and the selected model is then
reported on the full test split after validation has selected a checkpoint.

Default validation targets are mAP50 >= 0.65, mAP50-95 >= 0.35, each class's
mAP50-95 >= 0.20, and at most 10% of background images receiving a false box
at confidence >= 0.25. These are configurable initial criteria, not a promise
of accuracy. The dataset has far more screw boxes than pen boxes, so inspect
per-class results rather than only aggregate mAP.

Outputs include `metrics.json`, best/last checkpoints, confusion matrices,
validation plots and prediction JSON. Background false detections are included
in the validation and test report. If the targets remain unmet after retry,
`meets_validation_targets` is false; inspect errors and obtain better labels or
more representative examples before deciding how to proceed.

In the current cloud workspace, full training was launched with:

- Data: `/workspace/datasets/WoodWorkShop/prepared_with_backgrounds/data.yaml`
- Output: `/workspace/WoodWorkShop/runs/woodwork_full`
- Progress: `runs/woodwork_full/training.log` and `initial/results.csv`
- Process metadata: `runs/woodwork_full/process.json`

This is a CPU-only environment. Training takes hours; the process needs the
execution environment to remain alive. Launching it is not evidence that it
finished. Check `metrics.json` and the logs for completion.

The active run is monitored by `monitor_and_improve.py`. If the final test
misses a target, it starts one additional 640-pixel training cycle from the
selected checkpoint. That cycle chooses checkpoints by validation, then checks
the result on test. This extra CPU run can take many hours. Status is written to
`runs/woodwork_full/unattended_status.json`. Once the selected run finishes,
annotated test examples from each class and backgrounds are saved under its
`test_examples/` folder.

## Latest CPU training result

The completed CPU run met all configured targets; no additional training cycle
was needed. On 920 held-out test images, mAP50 was 93.7% and mAP50-95 was
78.1%. Class mAP50-95 was 82.9% for pens, 66.2% for screwdrivers, and 85.0%
for screws. Four of 250 background images received a detection at confidence
0.25 or above (1.6%).

- [Selected trained weights](models/best.pt)
- [Test metrics](previews/test_predictions/metrics.json)
- [Ten annotated test examples](previews/test_predictions/overview.jpg)

Run inference with:

```bash
yolo predict model=models/best.pt source=/path/to/photos conf=0.25 save=True device=cpu
```

## Predict

After training, use the selected checkpoint path recorded in `metrics.json`:

```bash
yolo predict model=/path/to/best.pt source=/path/to/photos conf=0.25 save=True device=cpu
```

## Verification

```bash
python -m unittest discover -s tests -v
python -m pip check
```

A separate tiny end-to-end smoke run tested training, retry, evaluation and
background metrics. Its results under `runs/smoke` are software checks, not
usable model quality estimates. The full training run is independent.
