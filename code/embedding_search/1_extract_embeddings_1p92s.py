## Extract PANNs embeddings using 4.8s non-overlapping sliding window

import os
import glob
import numpy as np
import librosa
import torch
from pathlib import Path
from panns_inference import AudioTagging


# ============================================================
# SETTINGS
# ============================================================

# Root folder containing Hydrophone subfolders with WAV files
WAV_ROOT = Path(r"D:\Autumn_deployment\done")

# Output folder for embeddings (mirrors the hydrophone subfolder structure)
OUTPUT_ROOT = Path(
    r"D:\Autumn_deployment\embeddings\panns_1.92_embeddings"
)

SAMPLE_RATE = 32000

WINDOW_SEC = 1.92
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)  # 153600 samples

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
# FIND ALL WAV FILES
# ============================================================

all_wavs = sorted(
    glob.glob(str(WAV_ROOT / "**/*.WAV"), recursive=True)
    + glob.glob(str(WAV_ROOT / "**/*.wav"), recursive=True)
)

print(f"Found {len(all_wavs)} WAV files.")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROCESS EACH FILE
# ============================================================

for file_i, wav_path in enumerate(all_wavs, start=1):

    wav_path = Path(wav_path)

    # Mirror subfolder structure in output
    rel_parts = wav_path.relative_to(WAV_ROOT).parts
    hydrophone = rel_parts[0]   # e.g. "Hydrophone1"

    out_dir = OUTPUT_ROOT / hydrophone
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / (wav_path.stem + "_panns_1.92s.npz")

    if out_path.exists():
        print(f"[{file_i}/{len(all_wavs)}] Skipping (already done): {wav_path.name}")
        continue

    print(f"[{file_i}/{len(all_wavs)}] Processing: {hydrophone}/{wav_path.name}")

    try:

        audio, sr = librosa.load(
            str(wav_path),
            sr=SAMPLE_RATE,
            mono=True
        )

        # Slice into non-overlapping 4.8s windows
        n_windows = len(audio) // WINDOW_SAMPLES

        if n_windows == 0:
            print(f"  Too short to slice, skipping.")
            continue

        start_times = []
        end_times = []
        embeddings = []

        for w in range(n_windows):

            start_sample = w * WINDOW_SAMPLES
            end_sample = start_sample + WINDOW_SAMPLES

            chunk = audio[start_sample:end_sample]
            chunk = chunk[np.newaxis, :].astype(np.float32)

            with torch.no_grad():
                _, emb = model.inference(chunk)

            embeddings.append(emb)
            start_times.append(w * WINDOW_SEC)
            end_times.append((w + 1) * WINDOW_SEC)

        embeddings = np.concatenate(embeddings, axis=0)
        start_times = np.array(start_times, dtype=np.float32)
        end_times = np.array(end_times, dtype=np.float32)

        np.savez_compressed(
            str(out_path),
            embeddings=embeddings,
            start_time_sec=start_times,
            end_time_sec=end_times,
            source_file=wav_path.name,
            hydrophone=hydrophone
        )

        print(f"  Saved {n_windows} windows → {out_path.name}")

    except Exception as e:
        print(f"  FAILED: {e}")


print("\nAll done.")
