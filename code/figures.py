#!/usr/bin/env python3
"""Make the two core report figures from a dataset built by build_dataset.py.

    python figures.py home_dataset.npz

-> accuracy_vs_keys.png   (how accuracy falls as more keys are added)
-> confusion.png          (row-normalised confusion for all keys, SVM)
"""
import sys, itertools, random
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import features
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND = os.path.join(_ROOT, "sound")
DATA  = os.path.join(_ROOT, "data")
FIGS  = os.path.join(_ROOT, "figures")
npz = sys.argv[1] if len(sys.argv) > 1 else "home"
if not os.path.exists(npz):
    _c = os.path.join(DATA, npz if npz.endswith(".npz") else f"{npz}_dataset.npz")
    if os.path.exists(_c): npz = _c
os.makedirs(FIGS, exist_ok=True)
d = np.load(npz, allow_pickle=True)
waves, y, fs = d["X"].astype(float), d["y"], int(d["fs"])
from collections import Counter as _C
_ok = {k for k,c in _C(y.tolist()).items() if c >= 5}
_m = np.isin(y, list(_ok)); waves, y = waves[_m], y[_m]
X = features.make(waves, fs, "spectrogram")
keys = sorted(set(y.tolist()))
clf = lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
cv = StratifiedKFold(5, shuffle=True, random_state=0)

# ---- Figure 1: accuracy vs number of keys ----
random.seed(0)
ks = list(range(2, len(keys)+1))
means, stds = [], []
for k in ks:
    combos = list(itertools.combinations(keys, k))
    random.shuffle(combos); combos = combos[:12]          # sample subsets
    accs = []
    for sub in combos:
        m = np.isin(y, sub)
        accs.append(cross_val_score(clf(), X[m], y[m], cv=cv).mean()*100)
    means.append(np.mean(accs)); stds.append(np.std(accs))
    print(f"{k} keys: {means[-1]:.1f}% (chance {100/k:.1f}%)")

fig, ax = plt.subplots(figsize=(6,4))
ax.errorbar(ks, means, yerr=stds, marker="o", capsize=3, label="SVM accuracy")
ax.plot(ks, [100/k for k in ks], "--", color="gray", label="chance")
ax.set_xlabel("number of keys"); ax.set_ylabel("accuracy (%)")
ax.set_title("Keystroke ID accuracy vs vocabulary size")
ax.legend(); ax.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(FIGS,"accuracy_vs_keys.png"), dpi=110); print("saved figures/accuracy_vs_keys.png")

# ---- Figure 2: row-normalised confusion, all keys ----
pred = cross_val_predict(clf(), X, y, cv=cv)
C = np.zeros((len(keys), len(keys)))
for i,t in enumerate(keys):
    for j,p in enumerate(keys):
        C[i,j] = ((pred[y==t])==p).sum()
C = C / C.sum(1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(5.5,5))
im = ax.imshow(C, cmap="viridis", vmin=0, vmax=100)
ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys)
ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title("Confusion (%), SVM")
for i in range(len(keys)):
    for j in range(len(keys)):
        ax.text(j,i,f"{C[i,j]:.0f}",ha="center",va="center",
                color="white" if C[i,j]<55 else "black",fontsize=8)
fig.colorbar(im, fraction=0.046); plt.tight_layout()
plt.savefig(os.path.join(FIGS,"confusion.png"), dpi=110); print("saved figures/confusion.png")
