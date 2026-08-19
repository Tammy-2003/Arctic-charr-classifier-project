
from pathlib import Path
import pandas as pd
from opensoundscape import CNN
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score


# ============================================================
# SETTINGS — edit these paths for your local machine
# ============================================================

MODEL_PATH      = Path(r"D:\Autumn_deployment\cnn_for_comparison\cnn_v4.model")
EXCEL_PATH      = Path(r"D:\Autumn_deployment\merged_gravel_FRT_labels_FINAL.xlsx")
CLIP_AUDIO_ROOT = Path(r"D:\Autumn_deployment\pooled_candidates_normalized")
FROZEN_TEST_CSV = Path(r"D:\Autumn_deployment\frozen_test_set_v2.csv")

CLASSES = ["FRT", "gravel"]

THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
              0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

# ============================================================
# File finder (same logic as the training script)
# ============================================================

def find_clip_file(file_name: str) -> Path:
    p = CLIP_AUDIO_ROOT / file_name
    if p.exists():
        return p
    matches = list(CLIP_AUDIO_ROOT.rglob(file_name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find clip: {file_name}")

# ============================================================
# Rebuild the frozen test set exactly as training did
# ============================================================

se = pd.read_excel(EXCEL_PATH, sheet_name="sound_event_labels")
cl = pd.read_excel(EXCEL_PATH, sheet_name="clip_labels")

def parse_sound_event_row(label_str):
    has_frt, has_gravel = False, False
    if not isinstance(label_str, str):
        return has_frt, has_gravel
    for part in label_str.split(";"):
        part = part.strip().lower()
        if part.startswith("fish") and "frt" in part and "gravel" not in part:
            has_frt = True
        if "gravel" in part:
            has_gravel = True
    return has_frt, has_gravel

se[["has_frt", "has_gravel"]] = se["sound_event_label"].apply(
    lambda x: pd.Series(parse_sound_event_row(x))
)
event_agg = se.groupby("file_name").agg(
    FRT=("has_frt", "max"), gravel=("has_gravel", "max")
).reset_index()

clip_agg = cl[["file_name"]].drop_duplicates().copy()
clip_agg["FRT"] = 0
clip_agg["gravel"] = 0

merged = pd.merge(
    event_agg, clip_agg, on="file_name", how="outer", suffixes=("_e", "_c")
)
for c in ["FRT_e", "gravel_e", "FRT_c", "gravel_c"]:
    merged[c] = merged[c].fillna(0).astype(int)
merged["FRT"] = (merged["FRT_e"] | merged["FRT_c"]).astype(int)
merged["gravel"] = (merged["gravel_e"] | merged["gravel_c"]).astype(int)
clip_level = merged[["file_name", "FRT", "gravel"]]

frozen_names = set(pd.read_csv(FROZEN_TEST_CSV)["file_name"])
test_rows = clip_level[clip_level["file_name"].isin(frozen_names)].copy()

print(f"Frozen test clips expected : {len(frozen_names)}")
print(f"Frozen test clips found in label sheet : {len(test_rows)}")

test_rows["real_path"] = test_rows["file_name"].apply(find_clip_file)
n_missing_audio = test_rows["real_path"].isna().sum()
test_rows = test_rows.dropna(subset=["real_path"])
print(f"Frozen test clips with audio found on disk : {len(test_rows)}")
if n_missing_audio:
    print(f"WARNING: {n_missing_audio} clips missing audio — excluded from evaluation.")

test_df = (
    test_rows[["real_path", "FRT", "gravel"]]
    .assign(real_path=lambda df: df["real_path"].astype(str))
    .set_index("real_path")
    .rename_axis("file")
)

print("\nTest label distribution:")
print(test_df.sum())

# ============================================================
# Load model and predict
# ============================================================

print(f"\nLoading model from: {MODEL_PATH}")

# Model was saved from a GPU (CUDA) session on the HPC, but this
# machine has no GPU — CNN.load() doesn't expose a device/map_location
# argument, so we intercept torch.load() itself and force it to map
# everything onto the CPU instead.
import torch
_original_torch_load = torch.load
def _load_to_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", torch.device("cpu"))
    return _original_torch_load(*args, **kwargs)
torch.load = _load_to_cpu

model = CNN.load(MODEL_PATH)

torch.load = _original_torch_load  # restore normal behavior afterwards

# Use whatever classes THIS model actually predicts, instead of the
# hardcoded ["FRT", "gravel"] — older models may only have "FRT"/"noise".
CLASSES = model.classes
print(f"Model classes: {CLASSES}")

# Only evaluate classes we actually have ground-truth labels for
# (this label sheet only has FRT/gravel truth columns — a class like
# "noise" from an older model has no ground truth here and can't be
# scored against this test set).
available_truth_cols = ["FRT", "gravel"]
skipped = [c for c in CLASSES if c not in available_truth_cols]
if skipped:
    print(f"WARNING: model has class(es) {skipped} with no ground-truth "
          f"column in this label sheet — skipping those.")
CLASSES = [c for c in CLASSES if c in available_truth_cols]
print(f"Classes actually being evaluated: {CLASSES}")

print("Running prediction on frozen test set (CPU is fine, ~473 clips)...")
scores = model.predict(test_df)

# predict() can occasionally split a clip into >1 window internally
# (e.g. if actual duration is a hair longer than sample_duration due
# to rounding), so scores can have MORE rows than test_df had inputs.
# Aggregate back to one score per file (max across any sub-windows)
# before comparing against per-file ground truth.
scores_reset = scores.reset_index()
file_col = "file" if "file" in scores_reset.columns else scores_reset.columns[0]
scores_per_file = scores_reset.groupby(file_col)[CLASSES].max()

# align to the same file order as test_df
scores = scores_per_file.reindex(test_df.index)
missing_scores = scores[CLASSES].isna().any(axis=1).sum()
if missing_scores:
    print(f"WARNING: {missing_scores} test clips got no prediction score — check for load errors.")

y_true = test_df[CLASSES].values

# ============================================================
# Threshold sweep — precision/recall/F1 per class per threshold
# ============================================================

results = []
for cls_idx, cls in enumerate(CLASSES):
    y_true_cls = y_true[:, cls_idx]
    y_score_cls = scores[cls].values

    # threshold-independent summary metric
    ap = average_precision_score(y_true_cls, y_score_cls)

    for t in THRESHOLDS:
        y_pred_cls = (y_score_cls >= t).astype(int)
        p = precision_score(y_true_cls, y_pred_cls, zero_division=0)
        r = recall_score(y_true_cls, y_pred_cls, zero_division=0)
        f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
        results.append({
            "class": cls,
            "threshold": t,
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3),
            "average_precision_(threshold-independent)": round(ap, 3),
        })

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("THRESHOLD SWEEP RESULTS")
print("=" * 70)
for cls in CLASSES:
    print(f"\n--- {cls} ---")
    print(results_df[results_df["class"] == cls]
          [["threshold", "precision", "recall", "f1"]]
          .to_string(index=False))
    ap = results_df[results_df["class"] == cls]["average_precision_(threshold-independent)"].iloc[0]
    print(f"Average Precision (PR-AUC): {ap}")

out_csv = MODEL_PATH.parent / "threshold_sweep_results.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved full results to: {out_csv}")
