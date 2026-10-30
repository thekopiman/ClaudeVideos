---
name: code-audio-synth
description: Synthesize a complete soundtrack in code (numpy, no samples, no audio models) for videos, motion graphics, product reels, games or UI demos - an original beat-locked score, UI clicks, ticks, pops, thumps, whooshes, drone/rotor hum, footsteps and car passes - then mix to -14 LUFS, mux with ffmpeg, and verify sync automatically. Use this whenever the user asks for music, a score, sound design, SFX, "sound on the beat", audio for a Python-rendered or code-made video, a loop-safe track, or says things like "add sound", "make it beat-synced", "UI sounds", or "whoosh on the transition", even if they never say "synthesize".
---

# Code-synthesized audio for video

Make the soundtrack the same way the picture is made: as a pure function of a shared timeline. One spec file holds the tempo, the music layers and every sound cue, with times snapped to a beat grid. The picture code reads the same times, so nothing can drift. Then the result is measured, not just listened to.

## Why this works (so you can adapt it)

- A beat grid turns "does it feel in sync?" into arithmetic: at 120 BPM a beat is exactly 0.5 s, a bar 2 s. Put visual events on beats and downbeats and every hit lands on its picture by construction.
- Synthesized sounds are deterministic (seeded noise), so re-rendering never changes the track, and there are no licensing problems.
- Sound has to follow the picture's real motion: a whoosh should peak when the move *lands*, not when it starts. The `whoosh` instrument takes a `peak` time for exactly this.

## Workflow

1. **Write the timeline first.** List every visual event with its time (ring appears at 2.0, camera tilts at 12.0, callout opens at 20.4...). Snap each to a beat (0.5 s steps at 120 BPM). Put big moments (title, tilt, logo, CTA) on downbeats (every 2 s).
2. **Write `spec.json`.** Start from `assets/example_spec.json`; keys are in `references/spec-format.md`. Music = chord progression + layers (kick, bass, arpeggio, hats) that enter and leave by bar so the track builds with the picture. SFX = a `cues` list copied from the timeline. Pick sounds from `references/sound-palette.md`.
3. **Render:** `python scripts/synth.py spec.json score.wav --stems stems/` (numpy only, about a second for 12 s, longer for many whooshes).
4. **Verify:** `python scripts/analyze.py stems/sfx.wav spec.json`. Every percussive cue must land within -20/+45 ms of its time and the tail must be silent if the film loops. Fix and re-render until it exits 0. Check the SFX stem, not the full mix: music masks cues and gives false misses.
5. **Mux:** `sh scripts/mux.sh silent.mp4 score.wav final.mp4 <seconds>` (loudnorm to -14 LUFS, TP -1.5, mono to stereo, padded to picture length). Confirm with `analyze.py ... --lufs final.mp4`; anything within about 1 LU of -14 is fine.
6. **If the picture is code-rendered**, import the same cue times from one shared module/JSON in both renderers. Do not retype them.

## Rules of thumb

- **Less is more.** Minimal and clean beats lush: kick, bass, plucks, hats. Avoid pads and reverb-like washes; they smear the beat grid and fight the SFX.
- **Layer the build.** Plucks from bar 2, hats from bar 3, extra bass pickup in the second half. A track that grows tells the story of the picture.
- **Sound the landing, not the start.** Whoosh `peak` = the moment the camera or object settles. Add a thump on the same downbeat for weight.
- **Quiet beds, loud punctuation.** Ambient layers (hum, steps, passes) sit at gain 0.05-0.1; UI clicks and ticks carry the sync.
- **End on silence for loops.** Stop the kick about 1 s before the end, let tails decay, and make the last event a single tick. `synth.py` warns if the last 0.3 s is not quiet.
- **Pitch tells direction.** Rising ticks (1500 -> 1900 -> 2400 Hz) for a route or a build; a single low thump for arrival.
- **Do not narrate on-screen text with sound.** Sound design supports the motion; it never reads captions.

## Extending the instruments

Add a method to `Synth` in `scripts/synth.py` (a function of time that returns an array) and it is immediately usable as a `kind` in `cues`. Keep decays exponential and give plucks a 3 ms attack ramp to avoid clicks. Per-sample loops (the whoosh filter) are fine at this scale; cache repeated sounds (the script already does).

## Files

- `scripts/synth.py` - spec -> WAV (music, cues, ambient beds, stems).
- `scripts/analyze.py` - cue sync, tail silence, peak, optional LUFS via ffmpeg; exit code 1 on failure.
- `scripts/mux.sh` - add audio to a video with loudness normalization.
- `assets/example_spec.json` - a working 12 s spec to copy.
- `references/spec-format.md` - every key in the spec.
- `references/sound-palette.md` - which picture event gets which sound, with levels.
