---
name: launch-motion-video
description: Make or restyle an explainer/launch video in the white "launch-video" motion style (kinetic headlines, soft-shadow cards, cursors, dashed connectors), rendered in Python + ffmpeg.
---

# Launch-style motion video

Builds an MP4 in a clean product-launch look: white canvas, big kinetic Inter headlines, soft-shadow cards and chips, a blue pointer cursor, flowing green dashed connectors, black "highlight" pills, and big-number stat beats. Content is whatever the user supplies; this skill is about the **look and the pipeline**.

Use it for: a new explainer/launch/teaser video on any topic, or restyling an existing video's drawing layer into this look.

## Workflow

1. **Intake.** Ask (one round, only what's missing): topic/source, target length (default 60–90 s), aspect (default 16:9 1280×720), must-show numbers or claims, audio (silent / user's music / voiceover).
2. **Research the content** if a source is given (paper, page, doc). Keep every on-screen number traceable; label anything you simulate as illustrative.
3. **Write the library.** Save the `motionkit.py` below into the working folder exactly as given. Check `fc-list | grep -i inter`; if Inter is missing, install it (e.g. `fonts-inter` package, or download Inter from its GitHub releases into the working folder).
4. **Storyboard.** Map the content onto the shot templates (table below): one idea per shot, ≤ 6-word headline, one grey subline, 4–9 s per shot, a big-stat beat (4 s) for each key number. Show the shot list to the user.
5. **Build** `video.py` with one function per shot, `fn(s, D)` (`s` = seconds into the shot). Use the templates where they fit; for custom diagrams use `layer()` + `geo_axes()` + primitives (see `demo.py` below).
6. **Snippets first.** `python3 video.py preview` → one full-size PNG per shot at its busiest moment. LOOK at each one at full size and fix: overlaps, text touching edges, overlays misaligned with data, lines through the headline or subline zones, chips covering badges. Then **send the PNGs to the user and wait for approval.**
7. **One full render.** `python3 video.py out.mp4`. Confirm the duration with `ffprobe`, grab 3–5 frames to check, deliver the MP4 with a 1–2 sentence summary.
8. **Changes later:** re-render only the changed shots: `render(SCENES, "out.mp4", only=[3, 7])`.

## Style rules

| | |
|---|---|
| Canvas | Pure white. No panels, plot frames, axes, ticks, gridlines, legends or colored step labels |
| Headline | Inter Display Medium, black, 46–52 pt, centered at y≈640 px, ≤ 6 words. Hero 66 pt, concept 84 pt, stats 120 pt |
| Subline | Inter Regular 13–17 pt, grey `#6b6b6b`, **one line** at y≈40 px. Never paragraphs |
| Color | Black/white/grey first. Blue = primary/cursor, green = good/flow, red = rejected, amber = the focal object. Pink/orange/purple only as tiny chip icons |
| Emphasis | Invert to a **black pill with white text**. No bold, underline or colored boxes |
| Motion | Headlines word-by-word (rise 26 px + trail). Chips/badges pop with overshoot. Cards slide up 30 px. Floating things bob ±5 px. Diagrams push in 4–8 %. Cursor glides between targets. Dashed connectors grow then flow. 0.25 s white fades at shot edges |
| Pacing | Something new every 1–2 s. Prefer more, shorter shots over dense ones |

## Shot templates (in motionkit)

| Template | Call | Use for |
|---|---|---|
| Hero title | `hero_title(s, D, line1, line2, chip, source)` | Opening |
| Diagram | custom: `layer()` → `geo_axes(rect_px, xlim, ylim, zoom)` → `layer()` with `badge`, `pill`, `dashed`, `cursor` at `to_pix(geo, pt)` | Any mechanism, map, network, chart |
| Card row | custom: 3–4 `shadow_box` cards (`card_in` for entrance) + mini `geo_axes` inside + `dashed` into a dark `pill` | Parallel units, options |
| Split comparison | custom: two cards, each ending in a verdict `pill` (green text = good, `dark=True` = bad) | Good vs bad, before vs after |
| Concept beat | `concept(s, D, word, tagline, chips)` | Introducing a named idea; rejected chips get struck through |
| Leaderboard | `leaderboard(s, D, title, rows, chosen, notes)` | Ranking, selection, scoring |
| Big stat | `big_stat(s, D, number, caption, chips, source)` | One key number per shot |
| Pipeline | `pipeline(s, D, title, steps, note=...)` | Recap of a process (5–7 steps) |
| End card | `end_card(s, D, title, source)` | Close |

## Pitfalls

- Layer order = paint order: `layer()` (cards) → `geo_axes` (data) → `layer()` (chips, cursor, text). Call `fig.clf()` per frame (the renderer does).
- Never `set_aspect(..., adjustable="datalim")` on diagram axes — it moves limits and overlays land in the wrong place. `geo_axes` computes equal aspect itself.
- Angles that wrap ±180°: break lines where consecutive values jump > 90°.
- Keep long trails out of the headline/subline zones; place labels on the side away from busy geometry.
- Scene functions must be pure functions of `s` (use seeded RNG for any randomness, precomputed outside the scene).
- Audio, if wanted, is added afterwards with `-c:v copy` (see the `video-audio-mix` skill if available).

## motionkit.py (write this file verbatim)

```python
"""motionkit - white "launch-video" motion graphics with matplotlib + ffmpeg.
Scenes are pure functions fn(s, D): s = seconds into the shot, D = shot length.
Usage:  from motionkit import *;  render(SCENES, "out.mp4")  |  preview(SCENES, PREVIEW_AT, "prev/")
"""
import os, sys, glob, subprocess, logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
from matplotlib.font_manager import FontProperties
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

W, H, FPS = 1280, 720, 30

# ---------------------------------------------------------------- fonts (Inter, by file so weights are exact)
def _font(names):
    dirs = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/Library/Fonts"), "/Library/Fonts", os.path.dirname(os.path.abspath(__file__))]
    for n in names:
        for d in dirs:
            hits = glob.glob(os.path.join(d, "**", n), recursive=True)
            if hits:
                return FontProperties(fname=hits[0])
    return FontProperties(family="DejaVu Sans")  # fallback: install Inter for the real look

F_HEAD = _font(["InterDisplay-Medium.otf", "InterDisplay-Medium.ttf", "Inter-Medium.otf", "Inter-Medium.ttf"])
F_BODY = _font(["Inter-Regular.otf", "Inter-Regular.ttf"])
F_MED = _font(["Inter-Medium.otf", "Inter-Medium.ttf"])
F_SEMI = _font(["Inter-SemiBold.otf", "Inter-SemiBold.ttf"])

# ---------------------------------------------------------------- palette
WHITE = "#ffffff"; INK = "#0a0a0a"; MUTED = "#6b6b6b"; LINE = "#ececec"; GREY = "#c4c4c4"
BLUE = "#2f6bff"; GREEN = "#1f9d55"; RED = "#e5383b"; AMBER = "#f5a524"
PINK = "#ec4899"; ORANGE = "#f59e0b"; PURPLE = "#8b5cf6"

# ---------------------------------------------------------------- timing
ease = lambda x: 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))
win = lambda s, a, b: float(np.clip((s - a) / (b - a), 0, 1))
def out_back(x, c=1.4):
    x = np.clip(x, 0, 1); return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
bob = lambda s, i, amp=5: np.sin(s * 0.9 + i * 1.7) * amp          # idle drift

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
_wc = {}

def layer():
    """Full-frame overlay in pixel coords (0..W, 0..H). Call order = paint order."""
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_facecolor("none")
    return ax

def text_w(s, fp, size):
    k = (s, fp.get_file() or fp.get_name(), size)
    if k not in _wc:
        t = fig.text(0, 0, s, fontproperties=fp, fontsize=size)
        _wc[k] = t.get_window_extent(fig.canvas.get_renderer()).width; t.remove()
    return _wc[k]

# ---------------------------------------------------------------- type
def headline(ax, txt, x, y, s, t0=0.0, size=50, ha="center", color=INK, fp=None, stagger=0.07):
    """Kinetic headline: words rise 26 px + fade in, staggered, with 2 faint motion trails."""
    fp = fp or F_HEAD
    words = txt.split(" "); sp = text_w("a b", fp, size) - text_w("ab", fp, size)
    ws = [text_w(w, fp, size) for w in words]; total = sum(ws) + sp * (len(words) - 1)
    cx = x - total / 2 if ha == "center" else (x - total if ha == "right" else x)
    for i, w in enumerate(words):
        p = win(s, t0 + i * stagger, t0 + i * stagger + 0.45)
        if p > 0:
            e = ease(p); dy = (1 - e) * 26
            if p < 1:
                for k, a in ((2.2, 0.10), (1.5, 0.18)):
                    ax.text(cx, y - dy * k, w, fontproperties=fp, fontsize=size, color=color, alpha=a * e)
            ax.text(cx, y - dy, w, fontproperties=fp, fontsize=size, color=color, alpha=e)
        cx += ws[i] + sp

def subline(ax, txt, s, t0=0.35, y=40, size=15, x=W / 2, ha="center", color=MUTED):
    a = ease(win(s, t0, t0 + 0.5))
    ax.text(x, y + (1 - a) * 8, txt, fontproperties=F_BODY, fontsize=size, color=color, ha=ha, alpha=a)

# ---------------------------------------------------------------- surfaces
def shadow_box(ax, x, y, w, h, r=16, fc=WHITE, ec=LINE, alpha=1.0, lw=1.0, z=2):
    """White rounded card with a soft 5-layer drop shadow."""
    for i in range(5):
        g = 2 + i * 3
        ax.add_patch(FancyBboxPatch((x, y - 3 - i * 1.5), w, h, boxstyle=f"round,pad={g},rounding_size={r + g}",
                                    fc="#000", ec="none", alpha=0.016 * alpha, zorder=z - 0.1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                alpha=alpha, zorder=z))

def card_in(s, t0, rise=30):
    """Card entrance: returns (alpha, dy). Use: a, dy = card_in(s, 0.3); shadow_box(ax, x, y + dy, ..., alpha=a)."""
    p = ease(win(s, t0, t0 + 0.5)); return p, -(1 - p) * rise

def pill(ax, x, y, txt, s=1.0, dark=False, size=14, icon=None, icon_color=BLUE, ha="center", z=6, color=None, fp=None):
    """Chip with fully rounded ends; pops in with overshoot as s goes 0->1. dark=True -> black highlight pill."""
    fp = fp or F_MED
    a = out_back(s); al = ease(min(1, s * 1.6))
    if al <= 0: return 0
    tw = text_w(txt, fp, size); h = size * 1.39 + 18; iw = (h - 10) if icon else 0
    w = (tw + 32 + iw + (8 if icon else 0)) * a; hh = h * a
    x0 = (x - w / 2) if ha == "center" else x; y0 = y - hh / 2
    shadow_box(ax, x0, y0, w, hh, r=hh / 2, fc=INK if dark else WHITE, ec=INK if dark else LINE, alpha=al, z=z)
    if a > 0.85:
        tx = x0 + 16
        if icon:
            ax.add_patch(Circle((tx + iw / 2, y), iw / 2, fc=WHITE if dark else "#f4f4f4", ec="none", zorder=z + .5, alpha=al))
            ax.text(tx + iw / 2, y, icon, fontproperties=F_SEMI, fontsize=size * .62, color=icon_color, ha="center",
                    va="center", zorder=z + 1, alpha=al)
            tx += iw + 8
        ax.text(tx, y - 1, txt, fontproperties=fp, fontsize=size, color=color or (WHITE if dark else INK),
                va="center", zorder=z + 1, alpha=al)
    return w

def badge(ax, x, y, r=24, label=None, s=1.0, color=BLUE, glyph="tri", z=5):
    """Round white icon chip with shadow. glyph: 'tri' | 'dot' | 'sq' | any short text."""
    a = out_back(s); al = ease(min(1, s * 1.6))
    if al <= 0: return
    rr = r * a
    for i in range(5):
        ax.add_patch(Circle((x, y - 3 - i), rr + 2 + i * 3, fc="#000", ec="none", alpha=0.018 * al, zorder=z - .1))
    ax.add_patch(Circle((x, y), rr, fc=WHITE, ec=LINE, lw=1, zorder=z, alpha=al))
    if glyph == "tri":
        tri = np.array([[0, .55], [-.5, -.35], [.5, -.35]]) * rr * .62 + [x, y + rr * .08]
        ax.add_patch(Polygon(tri, closed=True, fc=color, ec="none", zorder=z + .5, alpha=al))
    elif glyph == "dot":
        ax.add_patch(Circle((x, y), rr * .32, fc=color, ec="none", zorder=z + .5, alpha=al))
    elif glyph == "sq":
        q = rr * .55; ax.add_patch(FancyBboxPatch((x - q / 2, y - q / 2), q, q, boxstyle=f"round,pad=0,rounding_size={q*.25}",
                                                  fc=color, ec="none", zorder=z + .5, alpha=al))
    else:
        ax.text(x, y, glyph, fontproperties=F_SEMI, fontsize=rr * .55, color=color, ha="center", va="center", zorder=z + .5, alpha=al)
    if label:
        ax.text(x, y - rr - 16, label, fontproperties=F_SEMI, fontsize=11, color=INK, ha="center", va="center", alpha=al, zorder=z)

def cursor(ax, x, y, color=BLUE, scale=1.0, alpha=1.0, z=20):
    """Pointer dart; (x, y) is the tip."""
    pts = np.array([[0, 0], [7, -23], [11.5, -12.5], [23, -8]]) * scale + [x, y]
    ax.add_patch(Polygon(pts + [2, -3], closed=True, fc="#000", ec="none", alpha=.12 * alpha, zorder=z - .1))
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec=WHITE, lw=2, joinstyle="round", alpha=alpha, zorder=z))

def glide(s, keys):
    """Cursor path: keys = [(t, x, y), ...] -> (x, y) at time s (eased between keys)."""
    ts = [k[0] for k in keys]
    if s <= ts[0]: return keys[0][1:]
    for (t0, x0, y0), (t1, x1, y1) in zip(keys, keys[1:]):
        if s <= t1:
            e = ease((s - t0) / (t1 - t0)); return x0 + (x1 - x0) * e, y0 + (y1 - y0) * e
    return keys[-1][1:]

def dashed(ax, p, q, s, color=GREEN, lw=2.0, z=3, flow=None, alpha=1.0):
    """Connector that grows p->q as s: 0->1. flow = current time (seconds) to animate the dashes."""
    p = np.asarray(p, float); q = np.asarray(q, float); e = ease(s)
    if e <= 0: return
    r = p + (q - p) * e; ph = -(flow or 0) * 18
    ax.plot([p[0], r[0]], [p[1], r[1]], color=color, lw=lw, ls=(ph, (5, 4)), zorder=z, alpha=alpha, solid_capstyle="round")

def strike(ax, x0, x1, y, s, color=RED, lw=2.5, z=12):
    e = ease(s)
    if e > 0: ax.plot([x0, x0 + (x1 - x0) * e], [y, y], color=color, lw=lw, zorder=z, solid_capstyle="round")

# ---------------------------------------------------------------- data / diagram axes
def geo_axes(rect_px, xlim, ylim, zoom=1.0, pan=(0, 0), equal=True):
    """Frameless axes placed by pixel rect. zoom>1 pushes in. Equal aspect computed by hand (never adjustable='datalim')."""
    x, y, w, h = rect_px
    ax = fig.add_axes([x / W, y / H, w / W, h / H]); ax.axis("off"); ax.set_facecolor("none")
    cx, cy = (xlim[0] + xlim[1]) / 2 + pan[0], (ylim[0] + ylim[1]) / 2 + pan[1]
    if equal:
        k = max((xlim[1] - xlim[0]) / w, (ylim[1] - ylim[0]) / h) / zoom
        ax.set_xlim(cx - k * w / 2, cx + k * w / 2); ax.set_ylim(cy - k * h / 2, cy + k * h / 2)
    else:
        hx, hy = (xlim[1] - xlim[0]) / 2 / zoom, (ylim[1] - ylim[0]) / 2 / zoom
        ax.set_xlim(cx - hx, cx + hx); ax.set_ylim(cy - hy, cy + hy)
    ax.set_autoscale_on(False)
    return ax

def to_pix(ax, pts):
    """Data coords of `ax` -> pixel coords of a layer() overlay."""
    return ax.transData.transform(np.atleast_2d(pts))

# ---------------------------------------------------------------- shot templates
def hero_title(s, D, line1, line2=None, chip=None, source=None):
    fg = layer()
    for i, (x, y, c) in enumerate([(140, 560, BLUE), (1120, 590, GREEN), (230, 170, PURPLE), (1050, 150, BLUE), (640, 650, AMBER)]):
        badge(fg, x + bob(s, i, 6), y + bob(s, i + 3, 6), r=26, s=win(s, i * .1, i * .1 + .6), color=c, glyph="dot" if i % 2 else "tri")
    headline(fg, line1, W / 2, 400 if line2 else 360, s, .2, size=66)
    if line2: headline(fg, line2, W / 2, 318, s, .45, size=66)
    if chip:
        pill(fg, W / 2, 230, chip, win(s, 1.2, 1.7))
        cx, cy = glide(s, [(1.6, 1000, 120), (2.6, W / 2 + 140, 222)]); cursor(fg, cx, cy, alpha=ease(win(s, 1.5, 1.8)))
    if source: subline(fg, source, s, 1.8, y=60)

def concept(s, D, word, tagline, chips=()):
    """chips: [(x, y, text, rejected_bool)]"""
    fg = layer()
    headline(fg, word, W / 2, 400, s, 0, size=84)
    headline(fg, tagline, W / 2, 320, s, .5, size=40, color=MUTED)
    for i, (x, y, t, bad) in enumerate(chips):
        xx = x + bob(s, i)
        pill(fg, xx, y, t, win(s, .3 + i * .12, .8 + i * .12), size=16, icon="•", icon_color=RED if bad else GREEN)
        if bad:
            w = text_w(t, F_MED, 16) + 70; strike(fg, xx - w / 2 + 6, xx + w / 2 - 6, y, win(s, 1.8 + i * .1, 2.2 + i * .1))

def big_stat(s, D, number, caption, chips=(), source=None):
    """chips: [(x, y, text, icon_color)]"""
    fg = layer()
    headline(fg, number, W / 2, 360, s, 0, size=120)
    subline(fg, caption, s, .5, y=280, size=20)
    for i, (x, y, t, c) in enumerate(chips):
        pill(fg, x + bob(s, i), y, t, win(s, .6 + i * .15, 1.1 + i * .15), size=13, icon="•", icon_color=c)
    if source: pill(fg, W / 2, 80, source, win(s, 1.0, 1.5), size=12)

def leaderboard(s, D, title, rows, chosen=(), notes=(), select_at=2.5):
    """rows: [(label, value_str, active_bool)]; chosen: row indices that invert to black pills."""
    fg = layer()
    headline(fg, title, 90, 600, s, 0, size=50, ha="left")
    for j, n in enumerate(notes): subline(fg, n, s, .4 + j * .1, y=560 - j * 26, x=92, ha="left", size=16)
    w, rh = 600, 52; h = rh * len(rows) + 30; x, y = 600, max(70, (H - h) / 2 - 40)
    a, dy = card_in(s, .4); shadow_box(fg, x, y + dy, w, h, r=18, alpha=a)
    for n, (lab, val, act) in enumerate(rows):
        ra = ease(win(s, .7 + n * .1, 1.1 + n * .1))
        if ra <= 0: continue
        ry = y + dy + h - 40 - n * rh
        k = list(chosen).index(n) if n in chosen else -1
        sel = ease(win(s, select_at + k * .5, select_at + k * .5 + .4)) if k >= 0 else 0
        if sel > 0:
            pad = 8 * sel; shadow_box(fg, x + 14 - pad, ry - 20, w - 28 + 2 * pad, 40, r=12, fc=INK, ec=INK, alpha=sel, z=6)
        hl = sel > .5; col = WHITE if hl else (INK if act else "#b0b0b0")
        fg.text(x + 34, ry, str(n + 1), fontproperties=F_BODY, fontsize=14, color=WHITE if hl else MUTED, va="center", zorder=8, alpha=ra)
        fg.text(x + 70, ry, lab, fontproperties=F_MED, fontsize=16, color=col, va="center", zorder=8, alpha=ra)
        fg.text(x + w - 34, ry, val, fontproperties=F_MED, fontsize=15, color=col, va="center", ha="right", zorder=8, alpha=ra)
    if chosen:
        keys = [(select_at - .3, x + w - 60, y + h - 10)] + [(select_at + k * .5, x + w - 120, y + h - 46 - c * rh) for k, c in enumerate(chosen)]
        cx, cy = glide(s, keys); cursor(fg, cx, cy, alpha=ease(win(s, select_at - .4, select_at - .1)))

def pipeline(s, D, title, steps, colors=None, note=None):
    fg = layer()
    headline(fg, title, W / 2, 600, s, 0, size=54)
    n = len(steps); xs = np.linspace(130, W - 130, n); ys = np.array([440 if i % 2 == 0 else 330 for i in range(n)])
    colors = colors or [BLUE, GREEN, PURPLE, RED, ORANGE, GREEN, INK, BLUE][:n]
    for i in range(n - 1):
        dashed(fg, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), win(s, .6 + i * .35, .95 + i * .35), flow=s, z=2)
    for i, t in enumerate(steps):
        last = i == n - 1
        pill(fg, xs[i], ys[i], t, win(s, .4 + i * .35, .8 + i * .35), size=15, dark=last, icon=str(i + 1), icon_color=INK if last else colors[i])
    ci = min(n - 1, max(0, (s - .5) / .35))
    cursor(fg, np.interp(ci, range(n), xs) + 30, np.interp(ci, range(n), ys) - 14, alpha=ease(win(s, .6, .9)))
    if note: subline(fg, note, s, .5 + n * .35, y=180, size=17)

def end_card(s, D, title, source=None):
    fg = layer()
    badge(fg, W / 2, 400, r=30, s=win(s, 0, .5))
    headline(fg, title, W / 2, 300, s, .3, size=26, stagger=.03)
    if source: subline(fg, source, s, 1.0, y=240, size=14)

# ---------------------------------------------------------------- render
def _frame(fn, s, D, fade=True):
    fig.clf(); fig.patch.set_facecolor(WHITE)
    fn(s, D)
    if fade:
        f = min(1, s / .25, (D - s) / .25)
        if f < 1: layer().add_patch(Rectangle((0, 0), W, H, fc=WHITE, alpha=1 - f, zorder=100))

def preview(scenes, at=None, outdir="preview"):
    """One full-size PNG per shot. at: {index: seconds}; default = 75 % into the shot."""
    os.makedirs(outdir, exist_ok=True); paths = []
    for i, (fn, D) in enumerate(scenes):
        s = (at or {}).get(i, .75 * D); _frame(fn, s, D, fade=False)
        p = os.path.join(outdir, f"shot_{i:02d}.png"); fig.savefig(p, facecolor=WHITE); paths.append(p)
    return paths

def render(scenes, out="video.mp4", workdir="shots", only=None, crf=18):
    """One H.264 file per shot (tmp name, renamed on success), then concat -c copy.
    only: list of shot indices to re-render (others reused)."""
    os.makedirs(workdir, exist_ok=True); names = []
    for i, (fn, D) in enumerate(scenes):
        final = os.path.join(workdir, f"shot_{i:02d}.mp4"); names.append(final)
        if (only is not None and i not in only and os.path.exists(final)): continue
        tmp = final[:-4] + ".tmp.mp4"
        proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                                 "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
                                 "-preset", "medium", "-r", str(FPS), tmp], stdin=subprocess.PIPE)
        for f in range(int(round(D * FPS))):
            _frame(fn, f / FPS, D); fig.canvas.draw(); proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        proc.stdin.close()
        if proc.wait() != 0: sys.exit(f"shot {i} failed")
        os.replace(tmp, final); print("rendered", final, flush=True)
    lst = os.path.join(workdir, "list.txt")
    with open(lst, "w") as fh: fh.writelines(f"file '{os.path.abspath(n)}'\n" for n in names)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", out], check=True)
    return out
```

## demo.py (pattern for a video script)

```python
import sys
from motionkit import *

def s_hero(s, D): hero_title(s, D, "Motion kit", "launch-style videos", chip="Kinetic type  ·  Soft cards  ·  Cursors", source="demo")
def s_diagram(s, D):
    fg_bg = layer()
    geo = geo_axes((140, 70, 1000, 480), (0, 10), (0, 5), zoom=1 + .06 * s / D)
    fg = layer()
    headline(fg, "Data flows to the hub", W / 2, 640, s)
    hub = to_pix(geo, (5, 2.5))[0]
    nodes = [(1.5, 4), (1.5, 1), (8.5, 4), (8.5, 1)]
    for i, p in enumerate(nodes):
        P = to_pix(geo, p)[0]
        dashed(fg, P, hub, win(s, .8 + i * .15, 1.6 + i * .15), flow=s)
        badge(fg, P[0], P[1], r=24, label=f"Node {i+1}", s=win(s, .2 + i * .1, .7 + i * .1))
    pill(fg, hub[0], hub[1], "Hub", win(s, 1.6, 2.0), dark=True, size=16)
    cx, cy = glide(s, [(1.8, 900, 200), (2.6, hub[0] + 30, hub[1] - 12)]); cursor(fg, cx, cy, alpha=ease(win(s, 1.8, 2.0)))
    subline(fg, "Badges for actors, flowing dashed connectors, a black pill for the result.", s, .8)
def s_concept(s, D): concept(s, D, "Gating", "drop the bad ones early", [(200, 560, "A", False), (1080, 580, "B", True), (260, 170, "C", True), (1000, 160, "D", False)])
def s_board(s, D): leaderboard(s, D, "Pick the best", [("Option A", "0.33", True), ("Option B", "0.53", True), ("Option C", "3.04", True), ("Option D", "∞", False)], chosen=(0, 1), notes=["Lowest cost wins.", "Each item used once."])
def s_stat(s, D): big_stat(s, D, "5,832 → 84", "candidates left after gating", [(220, 560, "fewer costs", GREEN), (1060, 600, "smaller problem", BLUE)], source="Source chip")
def s_pipe(s, D): pipeline(s, D, "The whole pipeline", ["Input", "Filter", "Group", "Gate", "Assign", "Output"], note="One grey line of takeaway.")
def s_end(s, D): end_card(s, D, "Thanks for watching", "source · link")

SCENES = [(s_hero, 4), (s_diagram, 5), (s_concept, 4), (s_board, 5), (s_stat, 4), (s_pipe, 5), (s_end, 3)]
if __name__ == "__main__":
    if sys.argv[1:2] == ["preview"]: print(preview(SCENES, outdir="prev"))
    else: render(SCENES, sys.argv[1] if len(sys.argv) > 1 else "demo.mp4")
```