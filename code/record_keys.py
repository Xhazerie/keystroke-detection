#!/usr/bin/env python3
"""Record audio + a timestamped key log at the same time, for building a
labeled keystroke dataset. Your own keyboard, your own machine only.

Setup (once):
    sudo dnf install -y portaudio
    pip install sounddevice evdev soundfile numpy
    sudo usermod -aG input $USER     # then log out/in, or: newgrp input

Run:
    python record_keys.py sess1      # writes sess1.wav + sess1.csv
Type normally for a few minutes, mixing all the keys you care about, at a
firm even force, ~2-4 per second is fine. Press ESC to stop.

Then build the dataset with build_dataset.py.
"""
import sys, time, threading, csv, os
import numpy as np
import sounddevice as sd
import soundfile as sf
import evdev
from evdev import ecodes

FS = 48000
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND = os.path.join(_ROOT, "sound")
DATA  = os.path.join(_ROOT, "data")
FIGS  = os.path.join(_ROOT, "figures")
os.makedirs(SOUND, exist_ok=True)
prefix = sys.argv[1] if len(sys.argv) > 1 else "session"
# optional raw input device index (bypasses PipeWire AGC/noise-suppression).
# find indices with: python -c "import sounddevice as sd; print(sd.query_devices())"
device = int(sys.argv[2]) if len(sys.argv) > 2 else None

def find_keyboard():
    best, best_score = None, -1
    for path in evdev.list_devices():
        d = evdev.InputDevice(path)
        caps = d.capabilities().get(ecodes.EV_KEY, [])
        if ecodes.KEY_A in caps and ecodes.KEY_SPACE in caps:
            score = len(caps)                 # the real keyboard has the most keys
            if score > best_score:
                best, best_score = d, score
    return best

kbd = find_keyboard()
if kbd is None:
    sys.exit("No keyboard found. Are you in the 'input' group? (newgrp input)")
print(f"keyboard: {kbd.path}  {kbd.name}")

frames = []
def audio_cb(indata, n, t, status):
    frames.append(indata.copy())

stream = sd.InputStream(samplerate=FS, channels=1, device=device, callback=audio_cb)
print(f"audio device: {sd.query_devices(device)['name'] if device is not None else 'system default'}  @ {FS} Hz")
keys = []
stop = threading.Event()

stream.start()
t0 = time.monotonic()                          # shared clock origin
print("RECORDING. Type now. ESC to stop.")

try:
    for ev in kbd.read_loop():
        if ev.type == ecodes.EV_KEY and ev.value == 1:   # key DOWN only
            name = ecodes.KEY.get(ev.code, str(ev.code))
            name = name[4:] if name.startswith("KEY_") else name
            if name == "ESC":
                break
            keys.append((time.monotonic() - t0, name))
except KeyboardInterrupt:
    pass
finally:
    stream.stop(); stream.close()

audio = np.concatenate(frames)[:, 0] if frames else np.zeros(1)
sf.write(os.path.join(SOUND, f"{prefix}.wav"), audio, FS, subtype="PCM_16")
with open(os.path.join(SOUND, f"{prefix}.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["time_s", "key"])
    w.writerows([[f"{t:.4f}", k] for t, k in keys])

print(f"\nsaved sound/{prefix}.wav ({len(audio)/FS:.1f}s) and {prefix}.csv ({len(keys)} keys)")
