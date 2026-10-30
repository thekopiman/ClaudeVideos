---
name: python-3d-animation
description: Build 3D-looking animated videos entirely in Python (Pillow + numpy, ffmpeg only to encode) - a top-view map that tilts into a perspective 3D scene with extruded buildings (gable, hip, flat roofs), domes, onion domes and spires, shaded height-field terrain with contour lines and forest, rivers, cars, people, a flying drone, soft shadows, depth blur, callout tags and cards, driven by closed-form springs and rendered across CPU cores to an MP4 with a seamless loop. Use this whenever the user wants an animated 3D city, terrain, map, flythrough, "drone shot", isometric/tilt-up motion graphic, or says "generate the video with Python" / "make the animation in code", even if they never mention 3D, Pillow or springs.
---

# 3D animation in pure Python

The film is a **pure function of time**: `render(t)` paints frame `t` from scratch, with no state carried between frames. That one rule makes the work parallel (a process pool renders frames independently), repeatable (seeded randomness only), debuggable (`--still 12.5` shows any moment), and loopable (the last frame can equal the first exactly).

`scripts/anim3d.py` is the engine. `scripts/example_city.py` is an 8 s film that uses every part of it (read it first, it is ~170 lines). `examples/` holds two complete production films (`small_hill.py`, `salzburg.py`) for patterns that are too long to repeat here.

## Workflow

1. **Plan on the bar grid** (120 BPM: 0.5 s beat, 2 s bar). Write the timeline as a list of events with times, then the camera stops (time, target, zoom, rotation, pitch). If a storyboard exists, follow it exactly; if not, make one first (the `python-storyboard` skill) and wait for approval.
2. **Write the world as data at module level**, with seeded RNG: river polyline, hills (`make_hill` + `Heightfield`), house footprints, landmarks, waypoints, tag anchors. Workers re-import the module, so the same seed gives the same world everywhere. Compute distance-from-origin per object once (used to stagger growth).
3. **Write `Scene.render(t)`** in this order (see `example_city.py`):
   1. camera: `cam.set(pitch=..., rot=180, target=..., zoom=...)` from `track()` sums
   2. `Frame(cam, bg)`, ground drawing (`grid`, `ribbon` rivers, lots)
   3. add items: `box`, `dome/onion/spire/cylinder`, `terrain`, `tree`, `person`, `obox` (cars) with springs for `hf` (height growth) and `pr` (footprint presence)
   4. `frame.flush()` (shadows + painter's algorithm), then `frame.depth_blur(tilt)`
   5. overlays that stay sharp: route (`dotted`), `drone`, `tag`, `info_card`, `callout`
4. **Look before rendering.** `python film.py --still T --res 540` for key moments; `--sheet` for one frame per beat; then view the PNGs. Fix composition problems (landmarks hidden behind tall rows, haze over the hero, labels colliding) before the full render.
5. **Close the loop.** Retract everything near the end (growth, presence, camera 'return' spring), snap tiny tails to exact values, then run `python film.py --check-loop`; it must print `OK (exact)`.
6. **Add sound** with the `code-audio-synth` skill (`run(..., audio_fn=...)` or `--audio score.wav`), reusing the same cue times.
7. **Render in the foreground**: `python film.py --res 720 --fps 30` (about 0.2 s per frame per core at 720p, so a 24 s film takes ~2.5 min on one core). Background jobs started with `nohup &` are killed when the tool call returns in many sandboxes; poll a foreground run or split it. Use `--res 1080 --fps 60 --mblur 3` only for the final.
8. **Verify the MP4, not the code**: extract frames with `ffmpeg -ss T -i film.mp4 -frames:v 1`, check the first and last frame, loudness (`analyze.py --lufs`), and that duration matches.

## Rules that keep it good

- **Springs only.** Every entrance, exit and camera move is `spring()`/`track()`. No easing curves, no `time.sleep`, no `Math.random`, no state. A value with several targets is a *sum of springs*, one per change (`track(t, [(t0, delta), ...])`); never restart a spring. Presets: `snappy` (labels, pins), `default` (growth), `heavy` (the big tilt), `pop` (visible overshoot: pins, callout cards), `glide` (pans and flight legs), `return` (settling to frame 0).
- **Nothing fades in.** Reveal by growing (height, footprint), wiping a mask (`wipe`), drawing on (progress along a polyline), or extruding. Pure opacity fades look cheap.
- **Stagger by distance.** Houses, trees and lots start growing at `t0 + dist * k`, so growth rolls outward like a wave. Nearest-to-hero first.
- **One camera tilt on a downbeat**, 0 -> ~55 degrees with the `heavy` spring, halfway through the film. Before it everything is flat footprints (height factor ~0.06) so the top view reads as a map; after it heights spring up.
- **Heading-up.** Use `rot=180` (north arrow points down) when the story runs from the near edge to a far landmark, so hills and big buildings sit behind the action instead of hiding it.
- **Terrain is a shaded height field with contour lines**, flat tops where something stands, cliffs on the steep side, trees on slopes. Never stacked extruded rings (they look like layered cake and get rejected).
- **Flights are sums of springs.** A drone path = start point + one `glide` spring per leg, legs ~1 s apart: smooth, closed-form, loop-safe. Heading comes from `pos(t + dt) - pos(t - dt)`.
- **One accent colour.** Route, drone LED/trail, tag leaders, hero marker. Everything else neutral or pastel.
- **Overlays after `flush()`** so labels are never blurred or shadowed; use panel units (`cam.U`) for sizes so the layout works at any resolution.

## Pitfalls (details in `references/pitfalls.md`)

Tall objects in front hide landmarks behind them (lower the front row, move the camera, or move the landmark). Painter's algorithm sorts by `cam.depth`; give composite objects (a fortress on a hill) a depth just after their base. Heights shift things up the screen: centre the camera on the *ground point below the hero*, not on the hero. Settle tails (`1e-5`) change pixels, so snap them (pitch < 0.02, camera < 0.05, presence < 1e-3). Variable names that shadow palette constants (`RIVER`, `ROAD`) break `from anim3d import *`.

## Files

- `scripts/anim3d.py` - the engine (springs, Camera, Frame, Heightfield, overlays, `run()` CLI).
- `scripts/example_city.py` - complete 8 s example (top view -> tilt -> drone flight -> loop), with optional audio.
- `references/engine-api.md` - every class/method with parameters.
- `references/camera-and-springs.md` - spring maths, camera plan, flight, snapping, closing sequence.
- `references/scene-recipes.md` - houses and landmarks, terrain, rivers, cars, people, tags, 2D -> 3D transition, with snippets.
- `references/pitfalls.md` - lessons from real builds.
- `examples/` - two complete production films and a README.
