from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(r"D:\Autumn_deployment")
EXCEL_PATH = BASE_DIR / "merged_gravel_FRT_labels_FINAL.xlsx"
CLIP_AUDIO_ROOT = BASE_DIR / "pooled_candidates_normalized"
FROZEN_TEST_CSV = BASE_DIR / "frozen_test_set_v2.csv"

OUT_PNG = BASE_DIR / "pr_roc_comparison_v1_v2_v3_v4.png"
OUT_XLSX = BASE_DIR / "pr_roc_auc_summary_all_versions.xlsx"

CLASSES = ["FRT", "gravel", "insect", "noise"]

OUT_SIG_XLSX = BASE_DIR / "model_comparison_significance_tests.xlsx"

MODELS = {
    "v1": BASE_DIR / "cnn_for_comparison" / "best_v1.model",
    "v2": BASE_DIR / "cnn_for_comparison" / "cnn_frt_noise_multitarget_v2.model",
    "v3": BASE_DIR / "cnn_for_comparison" / "cnn_frt_gravel_insect_noise_multitargetv3.model",
    "v4": BASE_DIR / "cnn_for_comparison" / "cnn_v4.model",
}

# each version keeps its own color; only v4 (the final model) is solid,
# v1/v2/v3 are all dashed
STYLES = {
    "v1": {"color": "tab:blue",   "linestyle": "--"},
    "v2": {"color": "tab:orange", "linestyle": "--"},
    "v3": {"color": "tab:green",  "linestyle": "--"},
    "v4": {"color": "tab:red",    "linestyle": "-"},
}

# only plot these classes (drop insect/noise per request)
CLASSES_TO_PLOT = ["FRT", "gravel"]


def _compute_midrank(x):
    """Helper for the fast DeLong algorithm (Sun & Xu 2014)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, score_a, score_b):
    """Paired DeLong test comparing two ROC-AUCs computed from the SAME
    labeled examples (i.e. two models scored on the same held-out clips).
    Returns (auc_a, auc_b, z, p). Standard test for exactly this
    'compare classifier versions on one fixed test set' scenario -
    accounts for the correlation between the two AUCs instead of
    (incorrectly) treating them as independent samples."""
    from scipy import stats
    y_true = np.asarray(y_true, dtype=float)
    order = (-y_true).argsort()
    y_true_sorted = y_true[order]
    score_a_sorted = np.asarray(score_a)[order]
    score_b_sorted = np.asarray(score_b)[order]
    label_1_count = int(y_true_sorted.sum())
    preds = np.vstack([score_a_sorted, score_b_sorted])
    aucs, delongcov = _fast_delong(preds, label_1_count)
    auc_a, auc_b = aucs
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    if var <= 0:
        return auc_a, auc_b, 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, z, p


def paired_bootstrap_ap_test(y_true, score_a, score_b, n_boot=2000, seed=0):
    """DeLong's test only applies to ROC-AUC. For PR-AUC (average
    precision) there's no equivalent closed-form test, so this resamples
    the SAME clips (paired) with replacement n_boot times, recomputes
    both models' AP each time, and reports the fraction of resamples
    where the sign of the AP difference flips - a standard paired
    bootstrap p-value."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    n = len(y_true)
    ap_a = average_precision_score(y_true, score_a)
    ap_b = average_precision_score(y_true, score_b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if yb.sum() == 0 or yb.sum() == n:
            diffs[i] = np.nan
            continue
        diffs[i] = (average_precision_score(yb, score_a[idx])
                    - average_precision_score(yb, score_b[idx]))
    diffs = diffs[~np.isnan(diffs)]
    observed = ap_a - ap_b
    # two-sided: fraction of bootstrap diffs at least as extreme as 0 relative to observed direction
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = min(p, 1.0)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return ap_a, ap_b, observed, ci_lo, ci_hi, p


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


def parse_sound_event_row(label_str):
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


def parse_clip_row(label_str) -> bool:
    if not isinstance(label_str, str):
        return False
    return "noise" in label_str.lower()


def main():
    import torch
    from opensoundscape import CNN
    from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

    # ---- build the same full/correct ground truth as evaluate_v1_v2_full_testset.py ----
    events_df = pd.read_excel(EXCEL_PATH, sheet_name="sound_event_labels")
    clips_df = pd.read_excel(EXCEL_PATH, sheet_name="clip_labels")

    events_df[["has_frt", "has_gravel", "has_insect"]] = events_df["sound_event_label"].apply(
        lambda x: pd.Series(parse_sound_event_row(x)))
    event_agg = (events_df.groupby("file_name")
                 .agg(path=("path", "first"), FRT=("has_frt", "max"),
                      gravel=("has_gravel", "max"), insect=("has_insect", "max"))
                 .reset_index())

    clips_df["has_noise"] = clips_df["clip_label"].apply(parse_clip_row)
    clip_agg = (clips_df.groupby("file_name")
                .agg(path=("path", "first"), noise=("has_noise", "max"))
                .reset_index())

    merged = pd.merge(event_agg[["file_name", "path", "FRT", "gravel", "insect"]],
                       clip_agg[["file_name", "path", "noise"]],
                       on="file_name", how="outer", suffixes=("_event", "_clip"))
    for col in ["FRT", "gravel", "insect"]:
        merged[col] = merged[col].fillna(0).astype(int)
    merged["noise"] = merged["noise"].fillna(0).astype(int)
    merged["path"] = merged["path_event"].fillna(merged["path_clip"])
    all_df = merged[["file_name", "path", "FRT", "gravel", "insect", "noise"]].copy()

    frozen_names = set(pd.read_csv(FROZEN_TEST_CSV)["file_name"])
    test_rows = all_df[all_df["file_name"].isin(frozen_names)].copy()
    print(f"Frozen test clips expected: {len(frozen_names)}, found in master sheet: {len(test_rows)}")

    real_paths, missing = [], []
    for _, row in test_rows.iterrows():
        try:
            real_paths.append(find_clip_file(row["file_name"], str(row.get("path", ""))))
        except FileNotFoundError as e:
            real_paths.append(None)
            missing.append(str(e))
    test_rows["real_path"] = real_paths
    test_rows = test_rows.dropna(subset=["real_path"])
    if missing:
        print(f"WARNING: {len(missing)} clips missing audio - excluded.")
    print(f"Test clips with audio found on disk: {len(test_rows)}")

    test_df = (test_rows[["real_path", "FRT", "gravel", "insect", "noise"]]
               .assign(real_path=lambda df: df["real_path"].astype(str))
               .set_index("real_path").rename_axis("file"))
    test_df = test_df[CLASSES]
    print("\nTest set label distribution (same for every model version):")
    print(test_df.sum())

    # ---- score each model version ----
    _original_torch_load = torch.load
    def _load_to_cpu(*a, **kw):
        kw.setdefault("map_location", torch.device("cpu"))
        return _original_torch_load(*a, **kw)

    per_version_scores = {}
    per_version_classes = {}
    for tag, model_path in MODELS.items():
        if not model_path.exists():
            print(f"\n{tag}: model file not found at {model_path} - skipping")
            continue
        print(f"\n=== {tag}: loading {model_path.name} ===")
        torch.load = _load_to_cpu
        model = CNN.load(model_path)
        torch.load = _original_torch_load

        available_classes = [c for c in CLASSES if c in model.classes]
        print(f"  model.classes = {model.classes}  ->  evaluating: {available_classes}")
        per_version_classes[tag] = available_classes

        scores = model.predict(test_df)
        scores[model.classes] = 1 / (1 + np.exp(-scores[model.classes].astype(float)))
        scores_agg = scores.reset_index()
        group_col = "file" if "file" in scores_agg.columns else scores_agg.columns[0]
        scores_agg = scores_agg.groupby(group_col)[model.classes].max().reindex(test_df.index)
        per_version_scores[tag] = scores_agg

    # ---- pairwise significance tests between model versions, per class ----
    # DeLong's test for ROC-AUC (closed-form, accounts for paired/correlated
    # AUCs from the same test clips) + paired bootstrap for PR-AUC (no
    # closed-form equivalent exists for average precision).
    from itertools import combinations
    sig_rows = []
    for cls in CLASSES_TO_PLOT:
        y_true = test_df[cls].values.astype(float)
        versions_with_cls = [v for v in MODELS if v in per_version_scores
                              and cls in per_version_classes.get(v, [])]
        for va, vb in combinations(versions_with_cls, 2):
            sa = per_version_scores[va][cls].values
            sb = per_version_scores[vb][cls].values

            auc_a, auc_b, z, p_roc = delong_roc_test(y_true, sa, sb)
            ap_a, ap_b, ap_diff, ci_lo, ci_hi, p_ap = paired_bootstrap_ap_test(y_true, sa, sb)

            sig_rows.append({
                "class": cls, "version_A": va, "version_B": vb,
                "ROC_AUC_A": round(auc_a, 4), "ROC_AUC_B": round(auc_b, 4),
                "ROC_AUC_diff (A-B)": round(auc_a - auc_b, 4),
                "DeLong_z": round(z, 3), "DeLong_p": p_roc,
                "PR_AUC_A": round(ap_a, 4), "PR_AUC_B": round(ap_b, 4),
                "PR_AUC_diff (A-B)": round(ap_diff, 4),
                "PR_AUC_diff_95CI_lo": round(ci_lo, 4), "PR_AUC_diff_95CI_hi": round(ci_hi, 4),
                "bootstrap_p": p_ap,
            })
            print(f"  {cls} {va} vs {vb}: ROC-AUC {auc_a:.4f} vs {auc_b:.4f} "
                  f"(DeLong p={p_roc:.4g}); PR-AUC {ap_a:.4f} vs {ap_b:.4f} "
                  f"(bootstrap p={p_ap:.4g})")

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_excel(OUT_SIG_XLSX, index=False)
    print(f"\nSaved: {OUT_SIG_XLSX}")

    # ---- build PR + ROC curves per class, all 4 versions overlaid (thesis style) ----
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15,
        "axes.titlesize": 16, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "axes.edgecolor": "black", "axes.linewidth": 0.9,
        "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False,
    })
    panel_letters = ["a", "b", "c", "d"]
    panel_i = 0

    fig, axes = plt.subplots(len(CLASSES_TO_PLOT), 2, figsize=(9.5, 4.1 * len(CLASSES_TO_PLOT)))
    if len(CLASSES_TO_PLOT) == 1:
        axes = axes.reshape(1, 2)
    summary_rows = []

    for row, cls in enumerate(CLASSES_TO_PLOT):
        ax_pr, ax_roc = axes[row, 0], axes[row, 1]
        y_true = test_df[cls].values

        for tag in MODELS:
            if tag not in per_version_scores or cls not in per_version_classes.get(tag, []):
                continue
            y_score = per_version_scores[tag][cls].values

            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)

            style = STYLES[tag]
            # standard convention: x = recall, y = precision (reverted - the
            # precision-on-x version doubles back on itself and looks tangled
            # whenever precision isn't monotonic, e.g. the noisier v1 curve)
            ax_pr.plot(recall, precision, color=style["color"], linestyle=style["linestyle"],
                       lw=1.3, label=f"{tag} (AP = {ap:.2f})")
            ax_roc.plot(fpr, tpr, color=style["color"], linestyle=style["linestyle"],
                        lw=1.3, label=f"{tag} (AUC = {auc:.2f})")

            summary_rows.append({"class": cls, "version": tag, "PR_AUC (average_precision)": round(ap, 4),
                                  "ROC_AUC": round(auc, 4), "n_positive": int(y_true.sum()),
                                  "n_negative": int(len(y_true) - y_true.sum())})

        ax_pr.axhline(y_true.mean(), color="gray", lw=0.8, linestyle=":", label="baseline")
        ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
        ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)
        ax_pr.legend(fontsize=12, loc="lower left")
        panel_i += 1

        ax_roc.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle=":", label="chance")
        ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
        ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)
        ax_roc.legend(fontsize=12, loc="lower right")
        panel_i += 1

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.subplots_adjust(hspace=0.4)

    # panel letters placed just outside the bottom-left corner of each axes' own box
    for ax, letter in zip(axes.flat, panel_letters[: axes.size]):
        bbox = ax.get_position()
        fig.text(bbox.x0 - 0.045, bbox.y0 - 0.045, f"({letter})", fontsize=15,
                  fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.02,
             "FIGURE X. Performance of the FRT (a-b) and gravel (c-d) detectors across four training iterations "
             "(v1-v4), evaluated on the frozen 925-clip held-out test set. v1 (dashed): trained on the original "
             "candidate pool only. v2 (dashed): candidate pool expanded via active learning. v3 (dashed): gravel and "
             "insect introduced as explicit classes. v4 (solid): final model, with additional hard-negative gravel "
             "examples added to reduce false positives observed during full-corpus deployment. (a, c) Precision-"
             "recall curves. (b, d) ROC curves.",
             fontsize=11, wrap=True, ha="left", va="bottom")

    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    print(f"\nSaved: {OUT_PNG}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(OUT_XLSX, index=False)
    print(f"Saved: {OUT_XLSX}")
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
