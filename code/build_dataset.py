#!/usr/bin/env python3
"""Align a key log to audio onsets and cut a labeled keystroke dataset.

    python code/build_dataset.py sess1      # reads sound/sess1.wav + .csv
    -> data/sess1_dataset.npz  (X = per-press 64 ms waveforms, y = keys, fs)

For a single recording made by record_keys.py (audio and log on one clock). See
segment.py for the shared onset-detection / alignment / matching helpers.
"""
import sys, csv, os
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL, WIN, PRE, MINGAP, PROM = 0.120, 0.064, 0.005, 0.300, 0.015

if len(sys.argv) < 2:
    sys.exit("usage: python code/build_dataset.py <prefix>")
prefix = sys.argv[1]
x, fs = segment.read_mono(os.path.join(ROOT, "sound", f"{prefix}.wav"))
onsets, on_t = segment.detect_onsets(x, fs, min_gap_s=MINGAP, prominence=PROM)
print(f"onsets detected: {len(onsets)}")

rows = list(csv.DictReader(open(os.path.join(ROOT, "sound", f"{prefix}.csv"))))
kt = np.array([float(r["time_s"]) for r in rows]); kk = [r["key"] for r in rows]
print(f"keys logged    : {len(kk)}")

off, score = segment.search_offset(on_t, kt, tol=TOL)
print(f"best time offset: {off*1000:.0f} ms  ({score}/{len(kk)} keys matched)")

win, pre = int(WIN*fs), int(PRE*fs)
X, y = [], []
for ki, oi in segment.match_onsets(on_t, kt, off, TOL):
    s = max(0, onsets[oi] - pre); e = s + win
    if e > len(x):
        continue                               # press too close to end of audio
    X.append(x[s:e]); y.append(kk[ki])
miss = len(kk) - len(X)
X, y = np.array(X), np.array(y)
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
np.savez(os.path.join(ROOT, "data", f"{prefix}_dataset.npz"), X=X, y=y, fs=fs)
print(f"\nmatched {len(y)} presses, {miss} keys had no onset")
print("per-key counts:", dict(Counter(y)))
print(f"saved data/{prefix}_dataset.npz")
