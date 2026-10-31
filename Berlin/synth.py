#!/usr/bin/env python3
"""
synth.py - render a beat-locked score + sound effects to a WAV, from a JSON spec. numpy only.

    python synth.py spec.json out.wav                  # mixed mono 16-bit WAV
    python synth.py spec.json out.wav --stems stems/   # also writes stems/music.wav and stems/sfx.wav

Everything is placed on one grid: seconds = (bar * beats_per_bar + beat) * 60 / bpm.  The picture should read its
event times from the same spec (or the same constants), so sound and picture can never drift apart.

Spec (see ../references/spec-format.md for every key):
{
  "duration": 24.0, "bpm": 120, "sample_rate": 44100, "seed": 4242,
  "mix": {"music": 0.55, "sfx": 0.75, "drive": 1.1, "peak": 0.9},
  "music": {"roots": [33,29,36,31], "chords": [[69,72,76],...], "layers": [ {"type":"kick", ...}, ... ]},
  "cues": [ {"t": 0.0, "kind": "thump", "gain": 0.9}, {"t": 12.0, "kind": "whoosh", "dur": 1.6, "peak": 0.9, "gain": 0.7} ],
  "ambient": [ {"type": "hum", "from": 16, "to": 23.2, "gain": 0.05}, {"type": "steps", ...}, {"type": "passes", ...} ]
}
"""
import argparse
import json
import math
import os
import sys
import wave

import numpy as np


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)


class Synth:
    """Instruments are plain functions of time. Each returns a float array (about -1..1)."""

    def __init__(self, sr=44100, seed=4242):
        self.sr = sr
        self.rng = np.random.default_rng(seed)

    def t(self, dur):
        return np.arange(int(dur * self.sr)) / self.sr

    # ---- tonal / percussive ---------------------------------------------------------------
    def kick(self, f0=45.0, sweep=95.0, rate=30.0, decay=9.0, dur=0.38, **_):
        x = self.t(dur)
        return np.sin(2 * np.pi * (f0 * x + (sweep / rate) * (1 - np.exp(-rate * x)))) * np.exp(-x * decay)

    def bass(self, freq=55.0, dur=0.95, decay=3.2, **_):
        x = self.t(dur)
        return (np.sin(2 * np.pi * freq * x) + 0.28 * np.sin(4 * np.pi * freq * x) * np.exp(-x * 5)) * np.exp(-x * decay)

    def pluck(self, freq=440.0, dur=0.7, brightness=5.0, **_):
        x = self.t(dur)
        s = sum((1.0 / h) * np.sin(2 * np.pi * freq * h * x) * np.exp(-x * (brightness + 3 * h)) for h in range(1, 5))
        return s * np.minimum(1, x / 0.003)

    def hat(self, dur=0.09, decay=55.0, **_):
        x = self.t(dur)
        return np.diff(self.rng.standard_normal(len(x)), prepend=0) * np.exp(-x * decay)

    def tick(self, freq=1500.0, dur=0.07, decay=60.0, **_):
        x = self.t(dur)
        return np.sin(2 * np.pi * freq * x) * np.exp(-x * decay)

    def click(self, freq=3200.0, dur=0.03, decay=160.0, **_):
        x = self.t(dur)
        return (np.sin(2 * np.pi * freq * x) + 0.5 * self.rng.standard_normal(len(x))) * np.exp(-x * decay)

    def pop(self, f0=420.0, rise=300.0, dur=0.14, decay=24.0, **_):
        x = self.t(dur)
        return np.sin(2 * np.pi * (f0 * x + rise * x * x)) * np.exp(-x * decay)

    def thump(self, f0=52.0, sweep=70.0, rate=14.0, dur=0.6, decay=6.5, **_):
        x = self.t(dur)
        return np.sin(2 * np.pi * (f0 * x + (sweep / rate) * (1 - np.exp(-rate * x)))) * np.exp(-x * decay)

    def chime(self, freq=880.0, dur=0.9, **_):
        x = self.t(dur)
        return (np.sin(2 * np.pi * freq * x) + 0.4 * np.sin(2 * np.pi * freq * 2.76 * x)) * np.exp(-x * 4.5)

    # ---- noise sweeps ------------------------------------------------------------------------
    def whoosh(self, dur=1.4, peak=0.6, lo=300.0, hi=5500.0, **_):
        """Filtered noise that brightens toward `peak` (seconds from the start) and falls away. Peak = the landing."""
        m = int(dur * self.sr)
        z = self.rng.standard_normal(m)
        x = np.arange(m) / self.sr
        peak = min(max(peak, 0.05), dur * 0.95)
        env = np.sin(np.pi * np.clip(x / dur, 0, 1) ** (math.log(0.5) / math.log(peak / dur))) ** 2
        a = 1 - np.exp(-2 * np.pi * (lo + (hi - lo) * env) / self.sr)
        y = np.empty(m)
        s = 0.0
        for i in range(m):                       # one-pole low-pass with a moving cutoff
            s += a[i] * (z[i] - s)
            y[i] = s
        return y * env * 3.0

    def riser(self, dur=1.5, **_):
        m = int(dur * self.sr)
        x = np.arange(m) / self.sr
        env = (x / dur) ** 2
        z = self.rng.standard_normal(m)
        return np.diff(z, prepend=0) * env * 0.5 + np.sin(2 * np.pi * (200 + 1800 * (x / dur) ** 2) * x) * env * 0.3

    def render(self, kind, **p):
        fn = getattr(self, kind, None)
        if fn is None or kind.startswith("_") or kind in ("t", "render"):
            raise ValueError("unknown sound kind: %s" % kind)
        return fn(**p)


def build(spec):
    sr = int(spec.get("sample_rate", 44100))
    dur = float(spec["duration"])
    bpm = float(spec.get("bpm", 120))
    beat = 60.0 / bpm
    bpb = int(spec.get("beats_per_bar", 4))
    n = int(dur * sr)
    S = Synth(sr, int(spec.get("seed", 4242)))
    music, sfx = np.zeros(n), np.zeros(n)

    def add(buf, start, sig, gain=1.0):
        i0 = int(round(start * sr))
        if i0 < 0 or i0 >= n:
            return
        i1 = min(n, i0 + len(sig))
        buf[i0:i1] += gain * sig[:i1 - i0]

    def at(bar, beat_in_bar):
        return (bar * bpb + beat_in_bar) * beat

    mus = spec.get("music", {})
    roots = mus.get("roots", [33])
    chords = mus.get("chords", [[69, 72, 76]])
    cache = {}

    def cached(kind, **p):
        key = (kind, tuple(sorted(p.items())))
        if key not in cache:
            cache[key] = S.render(kind, **p)
        return cache[key]

    for L in mus.get("layers", []):
        typ = L["type"]
        g = L.get("gain", 0.5)
        b0, b1 = L.get("bars", [0, int(dur / (beat * bpb))])
        t_to = L.get("to", dur)
        if typ == "kick":
            step = L.get("every_beats", 1) * beat
            t = L.get("from", 0.0)
            while t <= t_to + 1e-9:
                add(music, t, cached("kick"), g)
                t += step
        elif typ == "bass":
            for bar in range(b0, b1 + 1):
                for bt in L.get("beats", [0, 2]):
                    f = midi(roots[bar % len(roots)]) * (1.5 if L.get("fifth") else 1.0)
                    if at(bar, bt) < t_to:
                        add(music, at(bar, bt), cached("bass", freq=f), g)
        elif typ == "arp":
            pat = L.get("pattern", [0, 1, 2, 1, 0, 2, 1, 2])
            step = L.get("step_beats", 0.5)
            per_bar = int(bpb / step)
            for bar in range(b0, b1 + 1):
                chord = chords[bar % len(chords)]
                for j in range(per_bar):
                    tm = at(bar, j * step)
                    if tm >= t_to:
                        break
                    note = chord[pat[j % len(pat)] % len(chord)]
                    if L.get("octave_every") and j % L["octave_every"] == L["octave_every"] - 1 and bar >= L.get("octave_from_bar", 0):
                        note += 12
                    gg = g if bar < L.get("late_from_bar", 10 ** 6) else L.get("gain_late", g)
                    add(music, tm, cached("pluck", freq=midi(note)), gg)
        elif typ == "hat":
            for bar in range(b0, b1 + 1):
                for k in range(bpb):
                    tm = at(bar, k + 0.5)
                    if tm < t_to:
                        add(music, tm, cached("hat"), g)
        else:
            raise ValueError("unknown music layer: %s" % typ)

    for c in spec.get("cues", []):
        p = {k: v for k, v in c.items() if k not in ("t", "kind", "gain")}
        add(sfx, c["t"], cached(c["kind"], **p), c.get("gain", 0.5))

    for a in spec.get("ambient", []):
        typ = a["type"]
        g = a.get("gain", 0.05)
        t0, t1 = a.get("from", 0.0), a.get("to", dur)
        if typ == "hum":                                   # rotor / engine bed with a slow swell
            m = int((t1 - t0) * sr)
            x = np.arange(m) / sr
            env = np.clip(x / 0.8, 0, 1) * np.clip(((t1 - t0) - x) / 0.9, 0, 1)
            lfo = a.get("lfo_period", 3.5)
            near = 0.65 + 0.35 * np.sin(2 * np.pi * x / lfo)
            f0 = a.get("f0", 150.0) * (1 + 0.01 * np.sin(2 * np.pi * 0.7 * x))
            ph = 2 * np.pi * np.cumsum(f0) / sr
            hum = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph)) * (0.65 + 0.35 * np.sin(2 * np.pi * a.get("am_hz", 48) * x))
            add(sfx, t0, hum * env * near, g)
        elif typ == "steps":
            t = t0
            k = 0
            while t < t1:
                x = S.t(0.03)
                step = np.diff(S.rng.standard_normal(len(x)), prepend=0) * np.exp(-x * 140)
                add(sfx, t, step, g * (1.0 + 0.4 * (k % 2)))
                t += a.get("every", 0.6)
                k += 1
        elif typ == "passes":
            for tm in a.get("times", []):
                add(sfx, tm, S.whoosh(dur=a.get("dur", 0.7), peak=a.get("peak", 0.35)), g)
        else:
            raise ValueError("unknown ambient type: %s" % typ)

    return music, sfx, spec.get("mix", {}), sr


def finish(music, sfx, mx):
    mix = mx.get("music", 0.55) * music + mx.get("sfx", 0.75) * sfx
    mix = np.tanh(mix * mx.get("drive", 1.1))                  # soft limiter, no hard clipping
    mix /= max(1e-6, np.abs(mix).max()) / mx.get("peak", 0.9)
    return mix


def write_wav(path, x, sr):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser(description="Render a score + SFX spec to WAV (numpy only).")
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--stems", default=None, help="folder to also write music.wav and sfx.wav")
    args = ap.parse_args()
    spec = json.load(open(args.spec))
    music, sfx, mx, sr = build(spec)
    mix = finish(music, sfx, mx)
    write_wav(args.out, mix, sr)
    if args.stems:
        os.makedirs(args.stems, exist_ok=True)
        for name, buf in (("music", music), ("sfx", sfx)):
            write_wav(os.path.join(args.stems, name + ".wav"), buf / max(1e-6, np.abs(buf).max()) * 0.9, sr)
    tail = mix[-int(0.3 * sr):]
    rms = float(np.sqrt(np.mean(tail ** 2)))
    print("wrote %s  (%.1f s, peak %.2f)" % (args.out, len(mix) / sr, np.abs(mix).max()))
    if rms > 0.01:
        print("WARNING: the last 0.3 s is not quiet (rms %.3f). For a loop, end on silence: stop layers early and let tails decay." % rms)


if __name__ == "__main__":
    main()
