

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from opensoundscape.ml.lightning import LightningSpectrogramModule
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

EXCEL_PATH      = BASE_DIR / "merged_gravel_FRT_labels_FINAL.xlsx"
CLIP_AUDIO_ROOT = BASE_DIR / "pooled_candidates_normalized"
FROZEN_TEST_CSV = BASE_DIR / "frozen_test_set_v2.csv"

OUTPUT_DIR      = BASE_DIR / "cnn_multitarget_out_v4_epoch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = OUTPUT_DIR / "cnn_v4_epoch20.model"
CONFUSION_MATRIX_CSV = OUTPUT_DIR / "confusion_matrix_v4_epoch20.csv"
TAG = "v4_epoch20"

SAMPLE_DURATION = 1.92
SAMPLE_RATE     = 48000
EPOCHS          = 20
BATCH_SIZE      = 16
NUM_WORKERS     = 4

VAL_FRACTION    = 0.10   # fraction of the non-test pool held out for validation
VAL_RANDOM_STATE = 42

CLASSES = ["FRT", "gravel", "insect", "noise"]

# ============================================================
# GPU check
# ============================================================

gpu_available = torch.cuda.is_available()
print(f"\nGPU available: {gpu_available}")
if gpu_available:
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no GPU detected.")
DEVICE = "cuda" if gpu_available else "cpu"

# ============================================================
# File finder
# ============================================================

def find_clip_file(file_name: str, path_hint: str = "") -> Path:
    if path_hint:
        p = Path(path_hint.replace("\\", "/"))
        if p.exists():
            return p
        p = Path.home() / path_hint.replace("\\", "/")
        if p.exists():
            return p
    p = CLIP_AUDIO_ROOT / file_name
    if p.exists():
        return p
    matches = list(CLIP_AUDIO_ROOT.rglob(file_name))
    if matches:
        return matches[0]
    stem = Path(file_name).stem
    matches = list(CLIP_AUDIO_ROOT.rglob(stem + "*.wav"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find clip: {file_name}")

# ============================================================
# Read Excel sheets
# ============================================================

events_df = pd.read_excel(EXCEL_PATH, sheet_name="sound_event_labels")
clips_df  = pd.read_excel(EXCEL_PATH, sheet_name="clip_labels")

print(f"sound_event_labels rows : {len(events_df)}")
print(f"clip_labels rows        : {len(clips_df)}")

# ============================================================
# Parse sound_event_labels
# ============================================================

def parse_sound_event_row(label_str: str):
    has_frt = has_gravel = has_insect = False
    if not isinstance(label_str, str):
        return has_frt, has_gravel, has_insect
    for part in label_str.split(";"):
        part = part.strip().lower()
        if part.startswith("fish") and "frt" in part and "gravel" not in part:
            has_frt = True
        if "gravel" in part:
            has_gravel = True
        if "insect" in part:
            has_insect = True
    return has_frt, has_gravel, has_insect

events_df[["has_frt", "has_gravel", "has_insect"]] = events_df["sound_event_label"].apply(
    lambda x: pd.Series(parse_sound_event_row(x))
)

event_agg = (
    events_df.groupby("file_name")
    .agg(path=("path", "first"), FRT=("has_frt", "max"), gravel=("has_gravel", "max"), insect=("has_insect", "max"))
    .reset_index()
)

print(f"\nUnique clips in sound_event_labels : {len(event_agg)}")
print(f"  FRT=1    : {event_agg['FRT'].sum()}")
print(f"  gravel=1 : {event_agg['gravel'].sum()}")
print(f"  insect=1 : {event_agg['insect'].sum()}")

def parse_clip_row(label_str: str) -> bool:
    if not isinstance(label_str, str):
        return False
    return "noise" in label_str.lower()

clips_df["has_noise"] = clips_df["clip_label"].apply(parse_clip_row)

clip_agg = (
    clips_df.groupby("file_name")
    .agg(path=("path", "first"), noise=("has_noise", "max"))
    .reset_index()
)

print(f"\nUnique clips in clip_labels : {len(clip_agg)}")
print(f"  noise=1 : {clip_agg['noise'].sum()}")

# ============================================================
# Merge both sources
# ============================================================

merged = pd.merge(
    event_agg[["file_name", "path", "FRT", "gravel", "insect"]],
    clip_agg[["file_name", "path", "noise"]],
    on="file_name", how="outer", suffixes=("_event", "_clip"),
)
for col in ["FRT", "gravel", "insect"]:
    merged[col] = merged[col].fillna(0).astype(int)
merged["noise"] = merged["noise"].fillna(0).astype(int)
merged["path"] = merged["path_event"].fillna(merged["path_clip"])

all_df = merged[["file_name", "path", "FRT", "gravel", "insect", "noise"]].copy()

print(f"\nTotal unique clips after merge : {len(all_df)}")
print(f"  FRT=1    : {all_df['FRT'].sum()}")
print(f"  gravel=1 : {all_df['gravel'].sum()}")
print(f"  insect=1 : {all_df['insect'].sum()}")
print(f"  noise=1  : {all_df['noise'].sum()}")

# ============================================================
# Resolve real file paths
# ============================================================

real_paths, missing = [], []
for _, row in all_df.iterrows():
    try:
        real_paths.append(find_clip_file(row["file_name"], str(row.get("path", ""))))
    except FileNotFoundError as e:
        real_paths.append(None)
        missing.append(str(e))

all_df["real_path"] = real_paths
all_df = all_df.dropna(subset=["real_path"])

print(f"\nClips found on disk : {len(all_df)}")
print(f"Clips missing       : {len(missing)}")
if missing:
    print("First 10 missing:")
    for m in missing[:10]:
        print(" ", m)

if len(all_df) == 0:
    raise ValueError("No clips found on disk.")

# ============================================================
# Build label_df
# ============================================================

label_df = (
    all_df[["real_path", "FRT", "gravel", "insect", "noise"]]
    .copy()
    .assign(real_path=lambda df: df["real_path"].astype(str))
    .set_index("real_path")
    .rename_axis("file")
)
label_df = label_df[CLASSES]

print("\nLabel distribution:")
print(label_df.sum())

# ============================================================
# Train/test split — FROZEN test set (same as v3, for a fair comparison)
# ============================================================

frozen_test_names = set(pd.read_csv(FROZEN_TEST_CSV)["file_name"])
file_basenames = label_df.index.to_series().apply(lambda p: Path(p).name)
is_test = file_basenames.isin(frozen_test_names)

test_df = label_df[is_test.values]
train_pool_df = label_df[~is_test.values]

n_found_in_data = is_test.sum()
n_frozen_total = len(frozen_test_names)
if n_found_in_data < n_frozen_total:
    print(f"\nWARNING: {n_frozen_total - n_found_in_data} of {n_frozen_total} frozen test clips not found.")

# ------------------------------------------------------------
# FIX: hold out a validation split from the TRAIN pool only.
# The frozen test set must never be seen during training/checkpointing,
# otherwise the confusion matrix computed on it later is leaked/biased.
# ------------------------------------------------------------
train_df, val_df = train_test_split(
    train_pool_df,
    test_size=VAL_FRACTION,
    random_state=VAL_RANDOM_STATE,
    shuffle=True,
)

print(f"\nTrain      : {len(train_df)} clips")
print(f"Val (from train pool, NOT test set) : {len(val_df)} clips")
print(f"Test (frozen, untouched until final eval) : {len(test_df)} clips")
print("\nTrain label distribution:")
print(train_df.sum())
print("\nVal label distribution:")
print(val_df.sum())
print("\nTest label distribution:")
print(test_df.sum())

# ============================================================
# Train
# ============================================================

model = LightningSpectrogramModule(
    architecture="resnet18",
    classes=CLASSES,
    sample_duration=SAMPLE_DURATION,
    sample_rate=SAMPLE_RATE,
    single_target=False,
)
print(f"\nTraining for {EPOCHS} epochs with batch size {BATCH_SIZE}")

# OpenSoundscape 0.13.x (including current 0.13.2): use the Lightning interface for true epoch-based training.
# This avoids converting epochs to a number of optimization steps.
trainer = model.fit_with_trainer(
    train_df,
    validation_df=val_df,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    save_path=OUTPUT_DIR / "epoch_training",
    accelerator="gpu" if gpu_available else "cpu",
    devices=1,
)

# Save the trained OpenSoundscape model object after epoch-based training.
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# ============================================================
# Evaluation + confusion matrix export
# (test_df is touched here for the first and only time)
# ============================================================

print("\n--- Test set evaluation (held out, never used in training/validation) ---")
scores = model.predict_with_trainer(
    test_df,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    lightning_trainer_kwargs={
        "accelerator": "gpu" if gpu_available else "cpu",
        "devices": 1,
    },
)

# predict_with_trainer() returns raw logits by default (pre-sigmoid), not probabilities —
# confirmed empirically while testing the full-corpus scan (raw values
# outside [0,1]). Apply sigmoid so threshold=0.5 below is a real 50%
# probability cutoff.
scores[CLASSES] = 1 / (1 + np.exp(-scores[CLASSES].astype(float)))

scores_agg = scores.reset_index()
group_col = "file" if "file" in scores_agg.columns else scores_agg.columns[0]
scores_agg = scores_agg.groupby(group_col)[CLASSES].max().reindex(test_df.index)

y_true = test_df[CLASSES].values
y_pred = (scores_agg[CLASSES].values >= 0.5).astype(int)

cm = multilabel_confusion_matrix(y_true, y_pred)

rows = []
for i, cls in enumerate(CLASSES):
    tn, fp, fn, tp = cm[i].ravel()
    n_pos, n_neg = tp + fn, tn + fp
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    rows.append({
        "snapshot": TAG, "class_name": cls, "threshold": 0.5,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "precision": precision, "recall": recall,
        "n_positive": n_pos, "n_negative": n_neg,
    })

cm_df = pd.DataFrame(rows)
cm_df.to_csv(CONFUSION_MATRIX_CSV, index=False)
print(f"\nSaved confusion matrix to: {CONFUSION_MATRIX_CSV}")
print(cm_df.to_string(index=False))
