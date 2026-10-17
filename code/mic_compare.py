#!/usr/bin/env python3
"""Compare two microphones on the SAME typing session (the SNR experiment).

    python code/mic_compare.py dual sound/mobile/home_row.wav

Uses <prefix>.csv as the shared key log and <prefix>.wav as mic 1 (laptop). The
second-mic wav has no log of its own, so it is aligned to the key log by
cross-correlating onset trains (segment.py). Reports each mic's press-to-noise
ratio and cross-validated accuracy, and saves figures/mic_compare.png.
"""
import sys, csv, os
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segment, features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prefix, mic2 = sys.argv[1], sys.argv[2]
WIN, PRE, TOL = 0.064, 0.006, 0.08

rows = list(csv.DictReader(open(os.path.join(ROOT, "sound", f"{prefix}.csv"))))
ks = [(float(r["time_s"]), r["key"]) for r in rows
      if len(r["key"]) == 1 and r["key"].isalnum()]
kt = np.array([t for t, _ in ks]); kk = [k for _, k in ks]


def score(path, prom, label):
    x, fs = segment.read_mono(path)
    onsets, on_t = segment.detect_onsets(x, fs, prominence=prom)
    env = segment.envelope(x, fs)
    snr = np.median(env[onsets]) / np.median(env)
    off = segment.xcorr_offset(on_t, kt, len(x) / fs)
    win, pre = int(WIN * fs), int(PRE * fs)
    waves, y = [], []
    for ki, oi in segment.match_onsets(on_t, kt, off, TOL):
        s = onsets[oi] - pre
        if 0 <= s and s + win <= len(x):
            waves.append(x[s:s + win]); y.append(kk[ki])
    X = features.spectrogram(np.array(waves), fs)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = cross_val_score(make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
                          X, np.array(y), cv=cv).mean() * 100
    print(f"{label}: SNR {snr:.1f}x, matched {len(y)}/{len(kt)}, accuracy {acc:.1f}%")
    return snr, acc


print(f"Same {len(set(kk))}-key session, two microphones:\n")
s1, a1 = score(os.path.join(ROOT, "sound", f"{prefix}.wav"), 0.015, "laptop mic")
s2, a2 = score(mic2, 0.010, "phone (close)")

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["laptop\n(%.0fx SNR)" % s1, "phone close\n(%.0fx SNR)" % s2],
              [a1, a2], color=["#8a94a6", "#2f80ed"])
ax.bar_label(bars, fmt="%.1f%%"); ax.set_ylabel("accuracy (%)")
ax.set_title(f"Microphone effect, same {len(set(kk))}-key session")
ax.set_ylim(0, 100); plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "mic_compare.png"), dpi=110)
print("saved figures/mic_compare.png")
