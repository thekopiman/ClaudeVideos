#!/usr/bin/env python3
"""
analyze.py - verify a synthesized soundtrack: cue sync, tail silence, and (with ffmpeg) integrated loudness.

    python analyze.py out.wav spec.json                 # onset of each percussive cue vs its planned time
    python analyze.py stems/sfx.wav spec.json           # best: check the SFX stem so music does not mask the cues
    python analyze.py out.wav spec.json --lufs film.mp4 # also print EBU R128 loudness of the muxed video (needs ffmpeg)

A cue passes when a detected onset lands within [-20 ms, +45 ms] of the planned time (sound may not lead the picture).
Whooshes are checked by their envelope peak (cue time + `peak`) against the planned landing.
Exit code 1 when any cue fails, so it can sit in a build script.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import wave

import numpy as np

PERC = {"tick", "click", "pop", "thump", "chime"}


def load(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(1)
    return x, sr


def onsets(x, sr, hop=64, win=256):
    d = np.diff(x, prepend=0)                                   # emphasise attacks
    e = np.array([np.sum(d[i:i + win] ** 2) for i in range(0, len(d) - win, hop)]) + 1e-12
    le = np.log(e)
    flux = np.maximum(0, np.diff(le, prepend=le[0]))
    thr = max(1.5, np.percentile(flux, 95))
    out, last = [], -1.0
    for i, f in enumerate(flux):
        t = i * hop / sr
        if f >= thr and t - last > 0.04:
            out.append(t)
            last = t
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("spec")
    ap.add_argument("--lufs", default=None, help="a video/audio file to measure with ffmpeg ebur128")
    args = ap.parse_args()
    x, sr = load(args.wav)
    spec = json.load(open(args.spec))
    on = onsets(x, sr)
    bad, rows = 0, []
    for c in spec.get("cues", []):
        t, k = c["t"], c["kind"]
        if k in PERC and t < 0.02:                              # an attack at t=0 has no previous frame to compare with
            seg = x[: int(0.06 * sr)]
            ok = float(np.sqrt(np.mean(seg ** 2))) > 0.02
            bad += 0 if ok else 1
            rows.append((t, k, 0.0 if ok else None, "ok (energy at start)" if ok else "silent start"))
        elif k in PERC:
            near = on[(on >= t - 0.02) & (on <= t + 0.08)]
            if len(near) == 0:
                rows.append((t, k, None, "no onset found (masked or too quiet?)"))
                bad += 1
                continue
            off = (near[0] - t) * 1000.0
            ok = -20.0 <= off <= 45.0
            bad += 0 if ok else 1
            rows.append((t, k, off, "ok" if ok else "OFF"))
        elif k == "whoosh":
            a, b = int(t * sr), int((t + c.get("dur", 1.4)) * sr)
            seg = np.abs(x[a:b])
            if len(seg) < 100:
                continue
            env = np.convolve(seg, np.ones(int(0.05 * sr)) / (0.05 * sr), mode="same")
            pk = t + np.argmax(env) / sr
            off = (pk - (t + c.get("peak", 0.6))) * 1000.0
            ok = abs(off) <= 250.0
            bad += 0 if ok else 1
            rows.append((t, k + " (peak)", off, "ok" if ok else "OFF"))
    print("%-8s %-14s %10s  %s" % ("t (s)", "kind", "offset ms", "status"))
    for t, k, off, st in rows:
        print("%-8.2f %-14s %10s  %s" % (t, k, "-" if off is None else "%+.1f" % off, st))
    offs = [r[2] for r in rows if r[2] is not None]
    if offs:
        print("\n%d cues checked, %d problem(s); mean offset %+.1f ms, max |offset| %.1f ms" % (len(rows), bad, np.mean(offs), np.max(np.abs(offs))))
    n = int(0.3 * sr)
    tail = float(np.sqrt(np.mean(x[-n:] ** 2)))
    print("tail rms (last 0.3 s): %.4f  %s" % (tail, "ok, loop-safe" if tail < 0.01 else "NOT quiet: a loop will thump"))
    print("peak: %.3f" % float(np.abs(x).max()))
    if args.lufs:
        if not shutil.which("ffmpeg"):
            print("ffmpeg not found; skipping loudness")
        else:
            r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", args.lufs, "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
            m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
            if m:
                print("integrated loudness: %s LUFS (target -14)" % m[-1])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
