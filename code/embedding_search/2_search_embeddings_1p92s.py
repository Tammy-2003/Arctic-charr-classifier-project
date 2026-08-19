## Build gravel displacement prototype from 4.8s examples, then search 1.92s embeddings

import os
import glob
import heapq
import numpy as np
import librosa
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from panns_inference import AudioTagging


# ============================================================
# SETTINGS
# ============================================================

# Directory containing gravel displacement example .wav files
# NOTE: these example clips are 4.8s long, but we are searching a library
# of 1.92s embeddings, so the prototype must be built at 1.92s resolution.
EXAMPLE_WAV_DIR = Path(
    r"D:\Autumn_deployment\example_sounds\gravel_displacement"
)

# Where to save the prototype embedding
PROTOTYPE_FILE = (
    r"D:\Autumn_deployment\example_embeddings"
    r"\gravel_displacement_example_panns_1p92s.npz"
)

# Root folder of 1.92s embeddings to search
EMBEDDING_ROOT = Path(
    r"D:\Autumn_deployment\embeddings\panns_1p92s_embeddings"
)

OUTPUT_CSV = (
    r"D:\Autumn_deployment"
    r"\gravel_displacement_top_candidates_pann_1p92s.csv"
)

SAMPLE_RATE = 32000
WINDOW_SEC = 1.92
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)

# Hop size for slicing the 4.8s example clips into 1.92s sub-windows.
# 4.8s is not an integer multiple of 1.92s (4.8 / 1.92 = 2.5), so we use
# an overlapping sliding window instead of non-overlapping chunks, to make
# sure we don't throw away the tail of each example and to get a smoother
# average over the whole clip. 50% overlap is a reasonable default.
EXAMPLE_HOP_SEC = 0.96
EXAMPLE_HOP_SAMPLES = int(EXAMPLE_HOP_SEC * SAMPLE_RATE)

TOP_N = 3000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD MODEL
# ============================================================

model = AudioTagging(
    checkpoint_path=None,
    device=DEVICE
)

print("Using device:", DEVICE)


# ============================================================
# BUILD PROTOTYPE FROM EXAMPLE CLIPS (sliding 1.92s windows)
# ============================================================

wav_files = sorted([
    f for f in EXAMPLE_WAV_DIR.glob("*.wav")
] + [
    f for f in EXAMPLE_WAV_DIR.glob("*.WAV")
])

if not wav_files:
    raise FileNotFoundError(
        f"No .wav files found in: {EXAMPLE_WAV_DIR}"
    )

print(f"Found {len(wav_files)} example clip(s).")


def sliding_windows(audio, window_samples, hop_samples):
    """
    Yield fixed-length windows from audio using a sliding hop.
    Always includes a final window aligned to the end of the clip
    (even if it overlaps more than `hop_samples` with the previous one),
    so the tail of the clip is never dropped.
    """
    n = len(audio)

    if n <= window_samples:
        # pad short/equal-length audio up to one window
        padded = np.pad(audio, (0, window_samples - n), mode="constant")
        yield padded
        return

    starts = list(range(0, n - window_samples + 1, hop_samples))

    # make sure the last window reaches the end of the clip
    last_start = n - window_samples
    if starts[-1] != last_start:
        starts.append(last_start)

    for s in starts:
        yield audio[s:s + window_samples]


all_embeddings = []

for wav_path in wav_files:

    print(f"  Processing: {wav_path.name}")

    audio, sr = librosa.load(
        str(wav_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    n_sub_windows = 0

    for chunk in sliding_windows(audio, WINDOW_SAMPLES, EXAMPLE_HOP_SAMPLES):

        chunk = chunk[np.newaxis, :].astype(np.float32)

        with torch.no_grad():
            _, emb = model.inference(chunk)

        all_embeddings.append(emb)
        n_sub_windows += 1

    print(f"    -> {n_sub_windows} sub-window(s) of {WINDOW_SEC}s")

all_embeddings = np.concatenate(all_embeddings, axis=0)
prototype = all_embeddings.mean(axis=0)

np.savez_compressed(
    PROTOTYPE_FILE,
    embeddings_1p92s=all_embeddings,
    prototype_embedding=prototype
)

print(f"Prototype saved: {PROTOTYPE_FILE}")
print(f"All embeddings shape : {all_embeddings.shape}")
print(f"Prototype shape      : {prototype.shape}")


# ============================================================
# LOAD PROTOTYPE
# ============================================================

data = np.load(PROTOTYPE_FILE)

if "prototype_embedding" in data:
    gravel_proto = data["prototype_embedding"]
elif "embeddings_1p92s" in data:
    gravel_proto = data["embeddings_1p92s"].mean(axis=0)
else:
    raise Exception("Cannot find prototype embedding")

gravel_proto = gravel_proto.reshape(1, -1)

print("\nLoaded gravel displacement prototype:", gravel_proto.shape)


# ============================================================
# FIND ALL 1.92s EMBEDDING FILES
# ============================================================

all_npz = sorted(
    glob.glob(str(EMBEDDING_ROOT / "**/*.npz"), recursive=True)
)

print(f"Found {len(all_npz)} embedding files to search.")


# ============================================================
# SEARCH — KEEP TOP N RESULTS
# ============================================================

top_hits = []

for file_i, npz_path in enumerate(all_npz, start=1):

    if file_i % 500 == 0:
        print(f"  {file_i}/{len(all_npz)} searched...")

    try:

        data = np.load(npz_path, allow_pickle=True)

        embeddings  = data["embeddings"]
        start_times = data["start_time_sec"]
        end_times   = data["end_time_sec"]
        source_file = str(data["source_file"])

        if "hydrophone" in data:
            hydrophone = str(data["hydrophone"])
        else:
            hydrophone = Path(npz_path).parent.name

        similarities = cosine_similarity(
            embeddings,
            gravel_proto
        ).ravel()

        for idx, sim in enumerate(similarities):

            row = (
                sim,
                hydrophone,
                source_file,
                start_times[idx],
                end_times[idx],
                npz_path,
                idx
            )

            if len(top_hits) < TOP_N:
                heapq.heappush(top_hits, row)
            elif sim > top_hits[0][0]:
                heapq.heapreplace(top_hits, row)

    except Exception as e:
        print(f"  FAILED: {npz_path}")
        print(f"  {e}")


# ============================================================
# SAVE RESULTS
# ============================================================

top_hits = sorted(top_hits, reverse=True)

rows = []
for hit in top_hits:
    rows.append({
        "similarity"    : hit[0],
        "hydrophone"    : hit[1],
        "source_file"   : hit[2],
        "start_time_sec": hit[3],
        "end_time_sec"  : hit[4],
        "embedding_file": hit[5],
        "window_id"     : hit[6]
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("\nFinished.")
print(f"Saved: {OUTPUT_CSV}")