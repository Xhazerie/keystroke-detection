"""Shared feature extraction for the keystroke project.

Datasets built by build_dataset.py store the raw waveform window per press
(X = N x samples). Turn those into features here so every script agrees.
"""
import numpy as np


def _mel_fb(nfft, fs, nmels=40, fmin=200, fmax=10000):
    f = np.fft.rfftfreq(nfft, 1 / fs)
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(fmin), mel(fmax), nmels + 2))
    fb = np.zeros((nmels, len(f)))
    for i in range(nmels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((f - lo) / (ce - lo), (hi - f) / (hi - ce)), 0, None)
    return fb


def spectrogram(waves, fs, frame_s=0.016, hop_s=0.008, nmels=40, fmin=200, fmax=10000):
    """Mini log-mel spectrogram per press, flattened. Captures the attack over time."""
    W = waves.shape[1]
    fl, hop = int(frame_s * fs), int(hop_s * fs)
    fb = _mel_fb(fl, fs, nmels, fmin, fmax)
    win = np.hanning(fl)
    starts = range(0, W - fl + 1, hop)
    out = np.empty((len(waves), nmels * len(list(starts))))
    for n, w in enumerate(waves):
        frames = []
        for st in range(0, W - fl + 1, hop):
            P = np.abs(np.fft.rfft(w[st:st + fl] * win)) ** 2
            frames.append(np.log(fb @ P + 1e-8))
        out[n] = np.concatenate(frames)
    return out


def single_frame(waves, fs, nmels=40, fmin=200, fmax=10000):
    """One log-mel spectrum over the whole window (the old, weaker feature)."""
    W = waves.shape[1]
    fb = _mel_fb(W, fs, nmels, fmin, fmax)
    S = np.abs(np.fft.rfft(waves * np.hanning(W), axis=1)) ** 2
    return np.log(S @ fb.T + 1e-8)


def make(waves, fs, kind="spectrogram"):
    return spectrogram(waves, fs) if kind == "spectrogram" else single_frame(waves, fs)
