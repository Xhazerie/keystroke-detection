#!/usr/bin/env python3
"""Real-time keystroke-from-sound demo (for showing the professor).

Trains the SVM on a saved dataset, then listens to the laptop mic. Every time it
hears a key press it shows the predicted key beside the key you actually pressed
(read from evdev) with a running accuracy, so the audience sees "pressed B ->
heard O" live. Green = hit, red = miss.

    # live, English 36-key model, raw mic device 4:
    python code/live_demo.py --data phone36 --device 4
    # live, full Thai labels:
    python code/live_demo.py --data thai_full --device 4
    # restrict to an easy, well-separated set (higher live accuracy):
    python code/live_demo.py --data phone36 --device 4 --keys QWERUIOP
    # top-3 instead of top-1:
    python code/live_demo.py --data phone36 --device 4 --topk 3

    # offline self-test on an existing recording (no mic, no GUI):
    python code/live_demo.py --data phone36 --replay sound/dual.wav sound/dual.csv

ESC in the window (or Ctrl-C in replay) stops.
"""
import argparse, os, sys, time, threading, queue, collections
import numpy as np
import soundfile as sf
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN, PRE = 0.064, 0.006          # press window (s) and pre-roll before the onset
REFRACTORY = 0.18                # min gap between detected presses (s)
PAIR_TOL = 0.25                  # match a prediction to a real keypress within (s)

THAI = {  # physical key / evdev name -> Kedmanee unshifted Thai char
    "Q": "ๆ", "W": "ไ", "E": "ำ", "R": "พ", "T": "ะ", "Y": "ั", "U": "ี",
    "I": "ร", "O": "น", "P": "ย", "A": "ฟ", "S": "ห", "D": "ก", "F": "ด",
    "G": "เ", "H": "้", "J": "่", "K": "า", "L": "ส", "Z": "ผ", "X": "ป",
    "C": "แ", "V": "อ", "B": "ิ", "N": "ื", "M": "ท",
    "1": "ๅ", "2": "/", "3": "-", "4": "ภ", "5": "ถ",
    "6": "ุ", "7": "ึ", "8": "ค", "9": "ต", "0": "จ",
    "SEMICOLON": "ว", "APOSTROPHE": "ง", "COMMA": "ม", "DOT": "ใ", "SLASH": "ฝ",
    "LEFTBRACE": "บ", "RIGHTBRACE": "ล", "MINUS": "ข", "EQUAL": "ช", "BACKSLASH": "ฃ",
}


def train(data, keys, thai):
    d = np.load(os.path.join(ROOT, "data", f"{data}_dataset.npz"), allow_pickle=True)
    Xw, y, fs = d["X"], d["y"].astype(str), int(d["fs"])
    if keys:                                     # restrict to a chosen key subset
        want = set(THAI.get(k, k) for k in keys) if thai else set(keys)
        m = np.array([yy in want for yy in y]); Xw, y = Xw[m], y[m]
    clf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
    clf.fit(features.spectrogram(Xw, fs), y)
    print(f"trained on {data}: {len(y)} presses, {len(set(y))} classes, fs={fs}")
    return clf, fs, int(WIN*fs), int(PRE*fs)


def predict(clf, window, fs, topk):
    X = features.spectrogram(window[None, :], fs)
    if topk == 1:
        return [str(clf.predict(X)[0])]
    scores = clf.decision_function(X)[0]         # ovr scores, one per class
    order = np.argsort(scores)[::-1][:topk]
    return [str(clf.classes_[i]) for i in order]


# ---- online onset detector over a stream of mono audio blocks -----------------
class Detector:
    """Online onset detector. ring[k] holds absolute sample (n - R + k). When a
    block crosses the noise-relative threshold we mark the onset's absolute index
    and defer the cut until `win` samples past (onset - pre) have arrived, so the
    full post-press window exists (the press rings out ~58 ms after the strike)."""
    def __init__(self, fs, win, pre):
        self.fs, self.win, self.pre = fs, win, pre
        self.R = fs                              # 1 s raw ring buffer
        self.ring = np.zeros(self.R)
        self.n = 0                               # total samples seen
        self.env_hist = collections.deque(maxlen=200)
        self.last = -1e9
        self.pending = None                      # absolute onset sample index

    def push(self, block):
        """Append a block; return a press window (or None) when one is ready."""
        b = block.astype(np.float32); L = len(b)
        self.ring = np.roll(self.ring, -L); self.ring[-L:] = b
        self.n += L
        e = float(np.sqrt(np.mean(b**2)))        # block RMS
        self.env_hist.append(e)
        if len(self.env_hist) >= 20 and self.pending is None:
            noise = np.median(self.env_hist)
            t = self.n / self.fs
            if e > max(noise * 3.5, 1.5e-4) and (t - self.last) > REFRACTORY:
                self.last = t
                self.pending = self.n - L // 2   # rough onset, refined at extract
        if self.pending is not None and self.n >= self.pending + self.win:
            # refine onset to the smoothed-envelope peak near the crossing, then
            # cut [peak-pre, peak-pre+win] exactly as build_dataset.py does.
            base = self.n - self.R
            c = self.pending - base
            sm = int(0.005 * self.fs)
            seg = self.ring[max(0, c - sm): c + self.win]
            env = np.convolve(np.abs(seg), np.ones(sm) / sm, mode="same")
            peak = max(0, c - sm) + int(np.argmax(env))
            k = peak - self.pre
            self.pending = None
            if 0 <= k and k + self.win <= self.R:
                return self.ring[k:k + self.win].copy()
        return None


# ---- ground-truth keyboard reader (evdev) -------------------------------------
def start_keyboard(thai):
    import evdev
    from evdev import ecodes
    best, score = None, -1
    for p in evdev.list_devices():
        dv = evdev.InputDevice(p); caps = dv.capabilities().get(ecodes.EV_KEY, [])
        s = (ecodes.KEY_A in caps) + (ecodes.KEY_SPACE in caps) + len(caps) / 1000
        if s > score: best, score = dv, s
    recent = collections.deque(maxlen=8)

    def loop():
        for ev in best.read_loop():
            if ev.type == ecodes.EV_KEY and ev.value == 1:
                name = ecodes.KEY[ev.code]
                if isinstance(name, list): name = name[0]
                name = name.replace("KEY_", "")
                lab = THAI.get(name, name) if thai else name
                recent.append((time.time(), lab))
    threading.Thread(target=loop, daemon=True).start()

    def truth_at():
        now = time.time()
        for ts, lab in reversed(recent):
            if now - ts <= PAIR_TOL:
                return lab
        return None
    return truth_at


# ---- replay mode: feed an existing wav + csv through the same detector --------
def replay(clf, fs, win, pre, wav, csvp, topk, thai):
    import csv as _csv
    x, fsr = sf.read(os.path.join(ROOT, wav))
    if x.ndim > 1: x = x.mean(1)
    rows = [r for r in _csv.DictReader(open(os.path.join(ROOT, csvp)))]
    kt = np.array([float(r["time_s"]) for r in rows])
    kk = [THAI.get(r["key"], r["key"]) if thai else r["key"] for r in rows]
    det = Detector(fs, win, pre); bs = int(0.01 * fs)
    hit = tot = 0
    for i in range(0, len(x) - bs, bs):
        w = det.push(x[i:i+bs])
        if w is None or len(w) < win: continue
        t = (i + bs) / fs
        j = int(np.argmin(np.abs(kt - t)))
        if abs(kt[j] - t) > 0.15: continue       # no real key near this onset
        preds = predict(clf, w, fs, topk)
        ok = kk[j] in preds; hit += ok; tot += 1
        if tot <= 25:
            print(f"  pressed {kk[j]:>3}  heard {preds}  {'OK' if ok else 'x'}")
    print(f"\nreplay streaming accuracy: {hit}/{tot} = {100*hit/max(tot,1):.1f}% (top-{topk})")


# ---- live GUI -----------------------------------------------------------------
def live(clf, fs, win, pre, device, topk, thai):
    import tkinter as tk
    import sounddevice as sd
    truth_at = start_keyboard(thai)
    det = Detector(fs, win, pre)
    q = queue.Queue()

    def cb(indata, frames, t, status):
        det_out = det.push(indata[:, 0].copy())
        if det_out is not None and len(det_out) >= win:
            q.put(det_out)
    stream = sd.InputStream(device=device, channels=1, samplerate=fs,
                            blocksize=int(0.01*fs), callback=cb)

    root = tk.Tk(); root.title("Keystroke from sound — live")
    root.configure(bg="#111"); root.geometry("760x460")
    thai_font = ("Noto Sans Thai", 120, "bold")
    big = ("DejaVu Sans", 120, "bold")
    tk.Label(root, text="pressed", fg="#888", bg="#111", font=("DejaVu Sans", 18)).place(x=120, y=40)
    tk.Label(root, text="heard", fg="#888", bg="#111", font=("DejaVu Sans", 18)).place(x=520, y=40)
    L = tk.Label(root, text="–", fg="#eee", bg="#111", font=thai_font if thai else big)
    R = tk.Label(root, text="–", fg="#2f80ed", bg="#111", font=thai_font if thai else big)
    L.place(x=90, y=90); R.place(x=490, y=90)
    arrow = tk.Label(root, text="→", fg="#555", bg="#111", font=("DejaVu Sans", 80)); arrow.place(x=360, y=120)
    acc = tk.Label(root, text="accuracy 0/0", fg="#aaa", bg="#111", font=("DejaVu Sans", 22)); acc.place(x=250, y=300)
    hist = tk.Label(root, text="", fg="#777", bg="#111", font=("DejaVu Sans", 20)); hist.place(x=40, y=370)
    state = {"hit": 0, "tot": 0, "h": collections.deque(maxlen=14)}

    def poll():
        try:
            while True:
                w = q.get_nowait()
                preds = predict(clf, w, fs, topk)
                real = truth_at()
                R.config(text=preds[0] + (f" ({','.join(preds[1:])})" if topk > 1 else ""))
                if real is not None:
                    ok = real in preds
                    L.config(text=real); R.config(fg="#27ae60" if ok else "#e74c3c")
                    state["hit"] += ok; state["tot"] += 1
                    state["h"].append(("·" if ok else "x"))
                    acc.config(text=f"accuracy {state['hit']}/{state['tot']} = "
                                    f"{100*state['hit']/state['tot']:.0f}%")
                    hist.config(text="  ".join(state["h"]))
                else:
                    R.config(fg="#2f80ed"); L.config(text="–")
        except queue.Empty:
            pass
        root.after(30, poll)

    root.bind("<Escape>", lambda e: (stream.stop(), root.destroy()))
    stream.start(); print("listening… press keys. ESC to quit.")
    poll(); root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="phone36")
    ap.add_argument("--device", type=int, default=4)
    ap.add_argument("--keys", default="")         # e.g. QWERUIOP to restrict
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--thai", action="store_true")
    ap.add_argument("--replay", nargs=2, metavar=("WAV", "CSV"))
    a = ap.parse_args()
    thai = a.thai or a.data.startswith("thai")
    clf, fs, win, pre = train(a.data, list(a.keys), thai)
    if a.replay:
        replay(clf, fs, win, pre, a.replay[0], a.replay[1], a.topk, thai)
    else:
        live(clf, fs, win, pre, a.device, a.topk, thai)


if __name__ == "__main__":
    main()
