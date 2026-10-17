"""Shared signal segmentation: find key presses in audio and line them up with a
key log. Every builder (build_dataset / build_phone / mic_compare / shift_detect)
used to re-implement these three steps; they live here once now.

    read_mono      load a wav as a mono float signal
    detect_onsets  FIR-smoothed energy envelope -> prominent, min-spaced peaks
    search_offset  best constant audio<->log time offset (small, same-clock)
    xcorr_offset   best offset by cross-correlating onset trains (phone, any clock)
    match_onsets   snap each logged key to its nearest unused onset within a tol
"""
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, fftconvolve


def read_mono(path):
    """Load a wav, averaging channels to mono. Returns (x, fs)."""
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, fs


def envelope(x, fs, smooth_s=0.005):
    """Rectify and low-pass with a moving-average FIR filter -> energy envelope."""
    n = int(smooth_s * fs)
    return np.convolve(np.abs(x), np.ones(n) / n, mode="same")


def detect_onsets(x, fs, min_gap_s=0.12, prominence=0.015, smooth_s=0.005):
    """Return (onset_samples, onset_times). Prominence (not a fraction of the max)
    keeps one loud spike/clip from hiding the real presses; min-gap suppresses the
    quieter release and decay ripples of a single press."""
    env = envelope(x, fs, smooth_s)
    pk, _ = find_peaks(env, distance=int(min_gap_s * fs), prominence=prominence)
    return pk, pk / fs


def search_offset(on_t, key_t, tol=0.12, span=0.5, step=0.005):
    """Small-range offset search (same-clock recording): slide the key log over
    +/-span and keep the shift where the most keys land on an onset within tol."""
    offs = np.arange(-span, span, step)
    best, score = 0.0, -1
    for o in offs:
        c = sum(len(on_t) and np.min(np.abs(on_t - (t + o))) <= tol for t in key_t)
        if c > score:
            best, score = o, c
    return best, score


def xcorr_offset(on_t, key_t, duration_s, res=0.002):
    """Wide-range offset by cross-correlating the two onset/key spike trains via
    FFT (for a second mic whose clock origin is unknown). Correlation = convolution
    with one signal reversed."""
    n = int(duration_s / res)

    def train(ts):
        a = np.zeros(n)
        idx = (ts / res).astype(int)
        idx = idx[(idx >= 0) & (idx < n)]
        a[idx] = 1
        return a

    c = fftconvolve(train(on_t), train(key_t)[::-1], mode="full")
    return (np.argmax(c) - (n - 1)) * res


def match_onsets(on_t, key_t, offset, tol):
    """Snap each logged key (shifted by offset) to its NEAREST UNUSED onset within
    tol. Resolves collisions in time order and consumes each onset once, so two keys
    can never share one segment. Returns list of (key_index, onset_index)."""
    used, pairs = set(), []
    for ki in np.argsort(key_t):
        d = np.abs(on_t - (key_t[ki] + offset))
        if used:
            d[list(used)] = np.inf
        cand = np.where(d <= tol)[0]
        if not len(cand):
            continue
        oi = int(cand[np.argmin(d[cand])])
        used.add(oi)
        pairs.append((int(ki), oi))
    return pairs
