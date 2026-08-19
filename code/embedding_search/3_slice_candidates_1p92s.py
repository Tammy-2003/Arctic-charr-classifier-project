## Slice audio clips identified by embedding search

import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import os
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = (
    r"D:\Autumn_deployment"
    r"\gravel_displacement_top_candidates_pann_4p8s.csv"
)

# Root folder containing Hydrophone1, Hydrophone2, etc.
WAV_ROOT = Path(r"D:\Autumn_deployment\done")

OUTPUT_DIR = Path(r"D:\Autumn_deployment\done\sliced_candidates_4p8s")

SAMPLE_RATE = 32000

# Small buffer around each window (seconds)
BUFFER_SEC = 0


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_FILE)

print(f"Loaded {len(df)} candidates.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SLICE AND SAVE
# ============================================================

success = 0
failed  = 0

for i, row in df.iterrows():

    hydrophone  = row["hydrophone"]       # e.g. "Hydrophone5"
    source_file = row["source_file"]      # e.g. "20251126_020300.WAV"
    start_sec   = row["start_time_sec"]
    end_sec     = row["end_time_sec"]
    similarity  = row["similarity"]

    wav_path = WAV_ROOT / hydrophone / source_file

    if not wav_path.exists():
        print(f"MISSING: {wav_path}")
        failed += 1
        continue

    try:

        start_buffered = max(0.0, start_sec - BUFFER_SEC)
        end_buffered   = end_sec + BUFFER_SEC

        audio, sr = librosa.load(
            str(wav_path),
            sr=SAMPLE_RATE,
            mono=True,
            offset=start_buffered,
            duration=end_buffered - start_buffered
        )

        # Filename: Hydrophone5_20251126_020300_t0018.24_sim0.931.wav
        stem     = Path(source_file).stem
        out_name = (
            f"{hydrophone}_{stem}"
            f"_t{start_sec:07.2f}"
            f"_sim{similarity:.3f}.wav"
        )

        sf.write(str(OUTPUT_DIR / out_name), audio, SAMPLE_RATE)

        success += 1

        if success % 100 == 0:
            print(f"  Saved {success} clips so far...")

    except Exception as e:
        print(f"FAILED: {wav_path} @ {start_sec:.2f}s — {e}")
        failed += 1


# ============================================================
# SUMMARY
# ============================================================

print("\nDone.")
print(f"  Successfully sliced : {success}")
print(f"  Failed              : {failed}")
print(f"  Output folder       : {OUTPUT_DIR}")
