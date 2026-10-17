#!/usr/bin/env python3
"""Shift-detection experiment for the Thai contribution.

    python code/shift_detect.py shift2                       # laptop mic
    python code/shift_detect.py shiftp sound/mobile/shiftp.wav   # phone mic

A letter is 'shifted' if a SHIFT key-down immediately precedes it in the log. The
analysis window reaches BACK seconds before the letter's onset so it contains the
earlier Shift press, and classifies shifted vs plain from the spectrogram (SVM,
5-fold CV). Record in RANDOM plain/shift order, or the classifier learns the typing
rhythm instead of the sound.
"""
import sys, csv, os
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segment, features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACK, FWD, TOL = 0.32, 0.064, 0.08

pre = sys.argv[1] if len(sys.argv) > 1 else "shift2"
phone = len(sys.argv) > 2
wav = sys.argv[2] if phone else os.path.join(ROOT, "sound", f"{pre}.wav")
x, fs = segment.read_mono(wav)
onsets, on_t = segment.detect_onsets(x, fs, min_gap_s=0.10,
                                     prominence=0.010 if phone else 0.015)

ev = [(float(r["time_s"]), r["key"]) for r in
      csv.DictReader(open(os.path.join(ROOT, "sound", f"{pre}.csv")))]
letters = [(t, (i > 0 and "SHIFT" in ev[i-1][1]))
           for i, (t, k) in enumerate(ev) if len(k) == 1 and k.isalpha()]
lt = np.array([t for t, _ in letters]); ls = np.array([s for _, s in letters])
print(f"{len(letters)} letters: {int(ls.sum())} shifted, {int((~ls).sum())} plain")

off = (segment.xcorr_offset(on_t, lt, len(x)/fs) if phone
       else segment.search_offset(on_t, lt, tol=TOL)[0])
back, fwd = int(BACK*fs), int(FWD*fs)
waves, lab = [], []
for ki, oi in segment.match_onsets(on_t, lt, off, TOL):
    o = onsets[oi]
    if o - back >= 0 and o + fwd <= len(x):
        waves.append(x[o-back:o+fwd]); lab.append(ls[ki])
X = features.spectrogram(np.array(waves), fs); lab = np.array(lab)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
acc = cross_val_score(make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
                      X, lab, cv=cv).mean() * 100
print(f"Shift detection (spectrogram SVM, 5-fold CV): {acc:.1f}%  "
      f"(chance {100*max(lab.mean(), 1-lab.mean()):.0f}%)")
