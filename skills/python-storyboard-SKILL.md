---
name: python-storyboard
description: Generate a visual storyboard sheet (a single PNG) for a video, motion graphic, map film or product reel entirely in Python with Pillow - numbered frames drawn from the real scene geometry, bar/time labels, captions, SFX lines, palette, rules and a "facts used" footer - so the user can approve the plan before anything is built. Use this whenever the user asks for a storyboard, shotlist, animatic stills, "show me before you build", "PNG first", a visual plan or sequence of frames, or wants to approve or change a video before rendering, even if they don't say the word "storyboard".
---

# Python storyboards

A storyboard is a promise. The sheet shows exactly what the film will do, bar by bar, drawn with the same palette and geometry the final render will use, so approval means something. Build it fast, show it, and stop for the user's OK before writing the animation.

## Workflow

1. **Understand the subject.** If it is a real place, product or event, research it first (web search) and keep a list of the facts you will put on screen (dates, names, heights). Put that list in the sheet footer as "FACTS USED". Never label something you did not verify; say "stylized, not to scale" when the map is stylized.
2. **Fix the grid.** Pick a tempo (120 BPM = 2 s bars) and a length (16 s = 8 bars, 24 s = 12). One panel per bar is the default; split the film into sections (for example "TOP VIEW 0-12 s" and "3D 12-24 s") so the big transition sits on a section boundary.
3. **Write the shotlist** (`references/shotlist-template.md`): per panel the time, a 1-3 word title, what happens on screen, the camera move, and the sound. Rule of thumb: something new happens every 2-4 seconds; frame 0 must already read; the last frame can equal the first so the film loops.
4. **Draw the panels in Python** with `scripts/panel_kit.py` (maps, rivers, terrain with contour lines, blocks, routes, pins, badges, legends, tags, north arrow). For 3D frames reuse the `python-3d-animation` skill's renderer so the storyboard frames are real frames. Save each as a PNG at 720x405.
5. **Assemble** with `scripts/make_sheet.py storyboard.json` (see the docstring for the JSON; `scripts/example_storyboard.py` is a complete runnable example). Layout, caption wrapping and footer are automatic.
6. **Look at the result at full size** and fix collisions: tags overlapping, labels clipped by a panel edge, captions too long, things hidden by haze. Crop and view individual panels when the sheet is too large to judge.
7. **Present the PNG and stop.** Summarize what changed vs. any earlier version in 3-5 bullets, list the assumptions you made (length, palette, accent, format), and end with "Reply **go** to build it, or tell me what to change". Do not start the video before the OK.

## Design rules (learned from real reviews)

- **One accent colour.** Everything else neutral. The accent marks the hero: route, pin, crosshair, drone, tag leaders.
- **Terrain must look like terrain.** Use a shaded height field with contour lines, flat tops where something sits on it, cliffs on the steep side, trees on slopes. Stacked extruded rings look like layered cake and get rejected.
- **Banned defaults:** a centred title on a gradient, everything fading in, corner labels and frame borders, glows on UI, particle bursts.
- **Show the style reference.** If the user supplied a reference image or an earlier film, put thumbnails top-right of the sheet (`references` in the JSON) and match their look.
- **Label sparingly.** Tags are for the 4-8 things a viewer should remember; keep each under ~35 characters.
- **Keep geometry honest.** If the film will be built from a Python module of world data, import that module in the storyboard script. Do not redraw from imagination.
- **Version the file** (`storyboard-v2.png`) instead of overwriting; users often compare versions.

## Handling feedback

Quote the complaint, change only what it points at, and say what you changed ("hills redesigned; buildings untouched"). When the user says "buildings are good", freeze that code path. When they say "ugly", look at the cause (shading, proportions, colour), do not just tweak numbers.

## Files

- `scripts/make_sheet.py` - JSON + panel PNGs -> sheet PNG (auto layout, wrapped captions, palette and footer columns).
- `scripts/panel_kit.py` - `Panel` drawing class and helpers (`make_hill`, `catmull`, `terrain`, `river`, `route`, `tag`, `pin`, `badge`, `legend`, `north_arrow`, `card`).
- `scripts/example_storyboard.py` - end-to-end example producing `storyboard.png`.
- `references/shotlist-template.md` - the table to write before drawing; includes timing math.
- `references/review-checklist.md` - what to check on the finished sheet before showing it.
