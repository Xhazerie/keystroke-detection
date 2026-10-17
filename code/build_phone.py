#!/usr/bin/env python3
"""Build the phone-microphone datasets and confusion figures.

Each phone recording has no key log of its own, so it is aligned to the matching
laptop key log (<prefix>.csv) by cross-correlating onset trains (segment.py).
Alphabet + digits + punctuation = the 46 physical keys of the Kedmanee main block.
Physical keys are layout-independent, so one alignment gives two datasets:

    data/phone36_dataset.npz    26 letters + 10 digits, ENGLISH labels
    data/thai_full_dataset.npz  + 10 punctuation keys, relabelled to THAI (46 keys)

and the figures phone36_confusion.png / thai_full_confusion.png. That the Thai run
scores the same as the English one (the label is cosmetic; the sound is the physical
key) is the point of the Thai contribution.

    python code/build_phone.py
"""
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segment, features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROM, TOL, WIN, PRE = 0.010, 0.08, 0.064, 0.006
THAI_FONT = "/usr/share/fonts/google-noto-vf/NotoSansThai[wght].ttf"

# physical key (single char or evdev name) -> Kedmanee unshifted Thai character.
THAI = {
    "Q": "ๆ", "W": "ไ", "E": "ำ", "R": "พ", "T": "ะ", "Y": "ั", "U": "ี",
    "I": "ร", "O": "น", "P": "ย", "A": "ฟ", "S": "ห", "D": "ก", "F": "ด",
    "G": "เ", "H": "้", "J": "่", "K": "า", "L": "ส", "Z": "ผ", "X": "ป",
    "C": "แ", "V": "อ", "B": "ิ", "N": "ื", "M": "ท",
    "1": "ๅ", "2": "/", "3": "-", "4": "ภ", "5": "ถ",
    "6": "ุ", "7": "ึ", "8": "ค", "9": "ต", "0": "จ",
    # punctuation keys (evdev names). (-)(=)(\\) provisional, verify before report.
    "SEMICOLON": "ว", "APOSTROPHE": "ง", "COMMA": "ม", "DOT": "ใ", "SLASH": "ฝ",
    "LEFTBRACE": "บ", "RIGHTBRACE": "ล", "MINUS": "ข", "EQUAL": "ช", "BACKSLASH": "ฃ",
}
LETTERS = [c for c in "QWERTYUIOPASDFGHJKLZXCVBNM"]
DIGITS = list("1234567890")
# session -> phone wav; built once, sliced into the two datasets afterwards.
SESSIONS = [("alpha2", "sound/mobile/alphabet.wav"),
            ("digits2", "sound/mobile/digits.wav"),
            ("punct", "sound/mobile/punctuation.wav")]


def extract(prefix, wav):
    """Align one phone wav to <prefix>.csv and return (waves 64 ms, physical keys, fs)."""
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, "sound", f"{prefix}.csv"))))
    ks = [(float(r["time_s"]), r["key"]) for r in rows if r["key"] in THAI]
    kt = np.array([t for t, _ in ks]); kk = [k for _, k in ks]
    x, fs = segment.read_mono(os.path.join(ROOT, wav))
    onsets, on_t = segment.detect_onsets(x, fs, prominence=PROM)
    off = segment.xcorr_offset(on_t, kt, len(x) / fs)
    win, pre = int(WIN * fs), int(PRE * fs)
    waves, keys = [], []
    for ki, oi in segment.match_onsets(on_t, kt, off, TOL):
        s = onsets[oi] - pre
        if 0 <= s and s + win <= len(x):
            waves.append(x[s:s + win]); keys.append(kk[ki])
    print(f"  {prefix}: matched {len(keys)}/{len(kt)}")
    return np.array(waves), keys, fs


def score_and_plot(Xw, y, fs, name, title, thai=False):
    np.savez(os.path.join(ROOT, "data", f"{name}_dataset.npz"), X=Xw, y=np.array(y), fs=fs)
    n = len(set(y))
    X = features.spectrogram(Xw, fs)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
    acc = cross_val_score(clf, X, y, cv=cv).mean() * 100
    print(f"{name}: {Xw.shape[0]} presses, {n} keys -> {acc:.1f}%  (chance {100/n:.1f}%)")

    if thai and os.path.exists(THAI_FONT):
        font_manager.fontManager.addfont(THAI_FONT)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=THAI_FONT).get_name()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    pred = cross_val_predict(clf, X, y, cv=cv)
    labels = sorted(set(y))
    cm = confusion_matrix(y, pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 9) if n > 36 else (8, 7))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{title}: {acc:.1f}%  (chance {100/n:.1f}%)")
    fig.colorbar(im, fraction=0.046); plt.tight_layout()
    out = os.path.join(ROOT, "figures", f"{name}_confusion.png")
    plt.savefig(out, dpi=110); plt.close(fig)
    print(f"  saved figures/{name}_confusion.png")


# align each session once
sess = {p: extract(p, w) for p, w in SESSIONS}
fs = next(iter(sess.values()))[2]

# phone36: alphabet + digits, English labels
w36 = np.vstack([sess["alpha2"][0], sess["digits2"][0]])
y36 = sess["alpha2"][1] + sess["digits2"][1]
score_and_plot(w36, y36, fs, "phone36", "Phone, 36 keys (EN)")

# thai_full: + punctuation, relabelled to Thai (46 keys)
wth = np.vstack([sess[p][0] for p, _ in SESSIONS])
yth = [THAI[k] for p, _ in SESSIONS for k in sess[p][1]]
score_and_plot(np.array(wth), yth, fs, "thai_full", "Thai Kedmanee, 46 keys (phone)", thai=True)
