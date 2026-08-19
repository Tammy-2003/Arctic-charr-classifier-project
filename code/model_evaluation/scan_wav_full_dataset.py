"""
scan_wav_hpc.py
================
Scans an entire folder tree of long audio recordings on the HPC for
FRT / gravel_displacement detections, using the trained multi-target
CNN (cnn_frt_gravel_multitarget.model).

Audio root is `done/`, which contains 4 hydrophone subfolders
(Hydrophone1, Hydrophone2, Hydrophone3, Hydrophone5) — rglob searches
all of them recursively, no need to list them individually.

Output is in TIDY/LONG format — one row per (file, time window, class),
not one column per class — so you get a clean "class" column and a
"time" column to filter/group by:

    file            start_time  end_time  class    probability  detected
    rec1.wav        0.0         1.92      FRT      0.83         True
    rec1.wav        0.0         1.92      gravel   0.04         False
    rec1.wav        0.96        2.88      FRT      0.12         False
    rec1.wav        0.96        2.88      gravel   0.61         True
    ...
"""

from pathlib import Path
import pandas as pd
import numpy as np
from opensoundscape import CNN

# ============================================================
# SETTINGS — edit these paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "cnn_multitarget_out" / "cnn_frt_gravel_insect_noise_multitarget.model"
AUDIO_ROOT = BASE_DIR / "done"              # contains Hydrophone1/2/3/5 subfolders
OUTPUT_DIR = BASE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The trained model now has 4 output classes (FRT/gravel/insect/noise),
# but this scan only needs to report FRT and gravel detections — insect
# and noise are still scored internally by model.predict() (it always
# scores every class the model has), we just don't include them here.
CLASSES = ["FRT", "gravel"]
THRESHOLD = 0.5
OVERLAP_FRACTION = 0.0   # 0 = no overlap between windows; raise (e.g. 0.5) for finer time localization
BATCH_SIZE = 32
NUM_WORKERS = 4          # match ncpus in your PBS job script

# ============================================================
# Load model
# ============================================================

print("Loading model...")
model = CNN.load(MODEL_PATH)
print(f"Model loaded. Classes: {model.classes}")

# ============================================================
# Recursively collect audio files (across all 4 hydrophone subfolders)
# ============================================================

audio_files = (
    list(AUDIO_ROOT.rglob("*.wav")) + list(AUDIO_ROOT.rglob("*.WAV")) +
    list(AUDIO_ROOT.rglob("*.mp3")) + list(AUDIO_ROOT.rglob("*.MP3"))
)
print(f"Found: {len(audio_files)} files under {AUDIO_ROOT}")

if len(audio_files) == 0:
    raise ValueError(f"No audio files found under {AUDIO_ROOT} — check the path.")

# ============================================================
# Predict — splits each long file into windows automatically
# ============================================================

scores = model.predict(
    [str(p) for p in audio_files],
    overlap_fraction=OVERLAP_FRACTION,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

scores = scores.reset_index()
print(scores.head())

# ============================================================
# Sanity-check whether outputs are already probabilities (0-1)
# or raw logits needing a sigmoid
# ============================================================

for cls in CLASSES:
    lo, hi = scores[cls].min(), scores[cls].max()
    print(f"{cls} value range: {lo:.3f} ~ {hi:.3f}")
    if lo < 0 or hi > 1:
        print(f"  -> looks like raw logits, applying sigmoid to {cls}")
        scores[cls] = 1 / (1 + np.exp(-scores[cls]))
    else:
        print(f"  -> already probabilities, leaving {cls} as-is")

# ============================================================
# Reshape to TIDY/LONG format: one row per (file, window, class)
# ============================================================

long_df = scores.melt(
    id_vars=["file", "start_time", "end_time"],
    value_vars=CLASSES,
    var_name="class",
    value_name="probability",
)
long_df["time"] = (
    long_df["start_time"].round(2).astype(str)
    + "-"
    + long_df["end_time"].round(2).astype(str)
)
long_df["detected"] = long_df["probability"] >= THRESHOLD

long_df = long_df[["file", "time", "start_time", "end_time", "class", "probability", "detected"]]

# ============================================================
# Save
# ============================================================

out_all = OUTPUT_DIR / "scan_results_all_windows.csv"
long_df.to_csv(out_all, index=False)
print(f"\nSaved all windows (long format) to: {out_all}")

out_detected = OUTPUT_DIR / "scan_results_detections_only.csv"
long_df[long_df["detected"]].to_csv(out_detected, index=False)
print(f"Saved detections-only rows to: {out_detected}")

print(f"\nTotal windows scored : {len(scores)}")
for cls in CLASSES:
    n_det = long_df[(long_df["class"] == cls) & (long_df["detected"])].shape[0]
    print(f"  {cls} detections (threshold={THRESHOLD}): {n_det}")
