---
name: video-audio-mix
description: Add music, voiceover and synthesized UI sound effects (pops, clicks, whooshes) to a finished MP4 without re-encoding the picture, mixed to -14 LUFS.
---

# Video audio mix

Adds sound to a finished video in one ffmpeg pass. The picture is copied (`-c:v copy`), so there is no re-render. Pairs with the `launch-motion-video` skill but works on any MP4.

## Workflow

1. **Ask (one round, only what's missing):** music (a file the user supplies and has rights to, or none), voiceover (a recorded file, or none), and whether they want UI sound effects.
   - Never download music from the web without explicit permission. Prefer the user's file.
2. **Write `audiomix.py`** (below) into the working folder exactly as given.
3. **SFX cues (optional):** build `cues.json` from the video's timeline. The script's shot list gives the times: sum the shot durations to get each shot's start, then add the in-shot times of events.
   - `whoosh`: a headline's first word (shot start + about 0.1 s).
   - `pop`: a chip or badge appearing.
   - `click`: the cursor landing, or a row being selected.
   - `tick`: small steps, like a counter or pipeline steps.
   - Keep it sparse: about one cue per second at most. Use `"gain"` values of 0.5–1.0.
4. **Mix:**

   ```bash
   python3 audiomix.py video.mp4 video_audio.mp4 --music music.mp3 --voice vo.wav --cues cues.json --music-vol 0.35
   ```

   - The music loops or trims to the video length, fades in over 1 s and out over 2.5 s, and is ducked under the voice.
   - Everything is loudness-normalised to -14 LUFS.
   - Any of `--music`, `--voice` and `--cues` can be left out.
5. **Verify:**
   - `ffprobe` shows one video and one audio stream with equal durations.
   - Spot-listen to the start, a busy shot and the end: no clipping, the voice stays intelligible, the SFX don't overpower.
   - Deliver the new file. Keep the silent original.

## Manual one-liners

- **Music only:**

  ```bash
  ffmpeg -i v.mp4 -stream_loop -1 -i m.mp3 -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -af "volume=0.8,afade=t=in:d=1,afade=t=out:st=<D-3>:d=3" -shortest out.mp4
  ```

- **Replace audio** on a video that already has some: the same command. `-map 0:v` drops the old track.
- **Shift voiceover timing:** `-itsoffset 1.5 -i vo.wav` delays the voice by 1.5 s.

## audiomix.py (write this file verbatim)

```python
"""audiomix - add music, voiceover and synthesized UI SFX to a finished video without re-encoding the picture.
python3 audiomix.py video.mp4 out.mp4 [--music m.mp3] [--voice v.wav] [--cues cues.json] [--music-vol 0.35]
cues.json: [{"t": 1.2, "sfx": "pop"}, {"t": 3.0, "sfx": "click"}, {"t": 5.5, "sfx": "whoosh"}]
"""
import argparse, json, os, subprocess, wave
import numpy as np

SR = 48000

def _env(n, a=0.002, d=0.08):
    t = np.arange(n) / SR
    return np.minimum(1, t / a) * np.exp(-t / d)

def synth(kind):
    rng = np.random.default_rng(1)
    if kind == "click":      # soft UI tick
        n = int(.06 * SR); t = np.arange(n) / SR
        x = np.sin(2 * np.pi * 2400 * t) * _env(n, .0005, .008) + .3 * rng.normal(0, 1, n) * _env(n, .0003, .003)
    elif kind == "pop":      # chip / badge appearing
        n = int(.12 * SR); t = np.arange(n) / SR; f = 900 + 700 * np.exp(-t / .02)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, .001, .04)
    elif kind == "whoosh":   # headline / transition
        n = int(.45 * SR); t = np.arange(n) / SR; noise = rng.normal(0, 1, n)
        k = np.exp(-((t - .22) / .12) ** 2)
        spec = np.fft.rfft(noise); fr = np.fft.rfftfreq(n, 1 / SR); spec *= np.exp(-((fr - 1800) / 1500) ** 2)
        x = np.fft.irfft(spec, n) * k
    elif kind == "tick":     # number counting / small step
        n = int(.03 * SR); t = np.arange(n) / SR
        x = np.sin(2 * np.pi * 3200 * t) * _env(n, .0003, .004)
    else:
        raise ValueError(kind)
    return x / (np.max(np.abs(x)) + 1e-9)

def write_wav(path, x):
    y = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(y.tobytes())

def duration(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "default=nw=1:nk=1", path]).strip())

def sfx_track(cues, dur, path, gain=0.35):
    buf = np.zeros(int(dur * SR) + SR)
    cache = {}
    for c in cues:
        x = cache.setdefault(c["sfx"], synth(c["sfx"])) * c.get("gain", 1.0)
        i = int(c["t"] * SR); buf[i:i + len(x)] += x[:len(buf) - i]
    write_wav(path, buf[:int(dur * SR)] * gain)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--music"); ap.add_argument("--voice"); ap.add_argument("--cues")
    ap.add_argument("--music-vol", type=float, default=0.35); ap.add_argument("--fade", type=float, default=2.5)
    a = ap.parse_args()
    D = duration(a.video)
    inputs, chains, labels = ["-i", a.video], [], []
    idx = 1
    if a.music:
        inputs += ["-stream_loop", "-1", "-i", a.music]
        chains.append(f"[{idx}:a]atrim=0:{D},asetpts=N/SR/TB,volume={a.music_vol},afade=t=in:d=1,"
                      f"afade=t=out:st={max(0, D - a.fade)}:d={a.fade}[mus]")
        mus = "[mus]"; idx += 1
    else:
        mus = None
    if a.voice:
        inputs += ["-i", a.voice]; vo = f"[{idx}:a]"; idx += 1
        if mus:  # duck music under the voice
            chains.append(f"{vo}asplit=2[vo1][vo2]")
            chains.append(f"{mus}[vo1]sidechaincompress=threshold=0.04:ratio=8:attack=20:release=300[duck]")
            labels += ["[duck]", "[vo2]"]
        else:
            labels.append(vo)
    elif mus:
        labels.append(mus)
    if a.cues:
        cues = json.load(open(a.cues)); p = os.path.splitext(a.out)[0] + "_sfx.wav"
        sfx_track(cues, D, p); inputs += ["-i", p]; labels.append(f"[{idx}:a]"); idx += 1
    if not labels:
        raise SystemExit("nothing to add: pass --music, --voice and/or --cues")
    chains.append("".join(labels) + f"amix=inputs={len(labels)}:duration=first:normalize=0,"
                  f"loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(chains),
           "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
           "-t", str(D), "-movflags", "+faststart", a.out]
    subprocess.run(cmd, check=True); print("wrote", a.out)

if __name__ == "__main__":
    main()
```