# Keyboard acoustic side-channel (Thai layout)

**DSP course project — 01204496**

Work out which keyboard key was pressed from the **sound** of the press, using only
DSP from the course syllabus, then extend it to the **Thai Kedmanee** layout. Framed as
defensive research: measure how well the attack works, where it fails, and what defends
against it.

The written report is in [`docs/report-th.pdf`](docs/report-th.pdf).

## Repository layout
```
code/        all Python (run from the project root: `python code/<script>.py …`)
figures/     output .png (used in the report)
docs/        report (TH) + presentation PDF + course template
data/        datasets <name>_dataset.npz   — NOT in git (large; see "Data")
sound/       recordings <name>.wav + .csv  — NOT in git (large; see "Data")
```

## Code
Two shared libraries, plus scripts that use them.

| File | Role |
|---|---|
| `segment.py` | **shared** — find presses (FIR envelope + prominence peaks), align audio to the key log (offset search / onset-train cross-correlation), match & consume onsets |
| `features.py` | **shared** — per-press waveform → mini log-mel **spectrogram** (the strong feature); `single_frame` is the weak baseline |
| `record_keys.py` | capture mic audio + a timestamped key log on one clock |
| `build_dataset.py` | one same-clock recording → labelled dataset `.npz` |
| `build_phone.py` | phone sessions → `phone36` (36 keys, EN) + `thai_full` (46 keys, Thai) datasets + figures |
| `classify.py` | dataset → 5-fold CV accuracy + confusion (`--clf knn/lda/svm/mlp`, `--feat spectrogram/single`) |
| `figures.py` | accuracy-vs-#keys curve + confusion figure |
| `mic_compare.py` | laptop vs phone on the **same** session (the SNR experiment) |
| `shift_detect.py` | Thai shift layer: was Shift held? (binary, randomised order) |
| `live_demo.py` | real-time demo: pressed key vs heard key + running accuracy |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# for live capture via evdev (Linux):
sudo usermod -aG input $USER     # once, then log out/in
# portaudio is needed by sounddevice (e.g. `sudo dnf install -y portaudio`)
```

## Pipeline
The method is: **record → detect onsets (FIR envelope) → window 64 ms → log-mel
spectrogram → SVM (5-fold CV)**.

```bash
source .venv/bin/activate
python code/record_keys.py sess1 4     # type, mixing keys, ESC to stop (4 = raw mic)
python code/build_dataset.py sess1     # -> data/sess1_dataset.npz
python code/classify.py sess1          # 5-fold CV accuracy + confusion
```

## Results (phone mic, within-session 5-fold CV)
| Experiment | Accuracy | Chance |
|---|---|---|
| 9 keys | 86% | 11% |
| 36 keys (letters + digits) | 52% | 2.8% |
| Full Thai Kedmanee, 46 keys | 60.5% | 2.2% |
| Shift detection | 74.6% | ~60% |
| SNR: laptop vs phone (same session) | 35% → 52% | — |
| Within-session vs cross-session | 95% vs 34% | — |

**Takeaways:** the **spectrogram** feature is the big lever (single-frame 33% →
spectrogram 52% on 9 keys); a classical **SVM** beats a small neural net at this data
size; recording **SNR** (microphone) is the ceiling; real use is **cross-session** — the
hardest setting, and the basis of the defense.

## Data
The raw recordings (`sound/`, ~620 MB) and built datasets (`data/`, ~140 MB) are **not
committed** — they exceed GitHub's limits and would bloat the clone. To reproduce from
your own typing, run the pipeline above; `build_phone.py` rebuilds the phone/Thai
datasets from a session. The result figures in `figures/` are checked in.

## Note
Three Thai punctuation glyphs (`-` `=` `\`) were provisional during development; verify
against an authoritative Kedmanee chart before citing them.

> Defensive security research — built to measure the attack and its defenses, not to
> capture anyone's keystrokes.
