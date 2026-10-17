#!/usr/bin/env python3
"""Classify a labeled keystroke dataset built by build_dataset.py.

    python classify.py home_dataset.npz                 # SVM on spectrogram (best)
    python classify.py home_dataset.npz --clf knn       # knn / lda / svm / mlp
    python classify.py home_dataset.npz --feat single   # single frame (weaker)

Reports stratified 5-fold cross-validated accuracy and a confusion matrix.
"""
import argparse, sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND = os.path.join(_ROOT, "sound")
DATA  = os.path.join(_ROOT, "data")
FIGS  = os.path.join(_ROOT, "figures")

import numpy as np
import features
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_clf(name):
    core = {"knn": KNeighborsClassifier(5),
            "lda": LinearDiscriminantAnalysis(),
            "svm": SVC(C=10, gamma="scale"),
            "mlp": MLPClassifier(hidden_layer_sizes=(128,), max_iter=600, random_state=0)
            }.get(name)
    if core is None:
        sys.exit(f"unknown classifier {name!r}")
    return make_pipeline(StandardScaler(), core)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--clf", default="svm", choices=["knn", "lda", "svm", "mlp"])
    ap.add_argument("--feat", default="spectrogram", choices=["spectrogram", "single"])
    args = ap.parse_args()

    path = args.npz
    if not os.path.exists(path):
        cand = os.path.join(DATA, path if path.endswith(".npz") else f"{path}_dataset.npz")
        if os.path.exists(cand): path = cand
    d = np.load(path, allow_pickle=True)
    waves, y, fs = d["X"].astype(float), d["y"], int(d["fs"])
    from collections import Counter as _C          # drop stray/typo keys (too few samples)
    _ok = {k for k,c in _C(y.tolist()).items() if c >= 5}
    _m = np.isin(y, list(_ok)); waves, y = waves[_m], y[_m]
    if waves.ndim != 2:
        sys.exit("dataset stores spectra, not waveforms - rebuild with the updated build_dataset.py")
    if len(y) < 10:
        sys.exit("dataset too small")

    X = features.make(waves, fs, args.feat)
    clf = make_clf(args.clf)
    labels = sorted(set(y.tolist()))
    print("per-key counts:", {l: int((y == l).sum()) for l in labels})
    print(f"{len(y)} presses, {len(labels)} keys, chance {100/len(labels):.1f}%")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = cross_val_score(clf, X, y, cv=cv).mean() * 100
    print(f"\n{args.clf.upper()} + {args.feat}, 5-fold CV: {acc:.1f}%")

    pred = cross_val_predict(clf, X, y, cv=cv)
    if len(labels) <= 20:
        print("\nConfusion (rows=true):")
        print("     " + " ".join(f"{l:>3}" for l in labels))
        for t in labels:
            print(f"{t:>3}  " + " ".join(f"{int(((pred[y==t])==p).sum()):3d}" for p in labels))


if __name__ == "__main__":
    main()
