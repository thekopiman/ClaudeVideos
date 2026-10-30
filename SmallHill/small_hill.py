#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMALL HILL  -  a 16 s top-view map film that tilts into 3D at 8 s.
Pure Python: every frame is drawn with Pillow, the score and SFX are synthesized with numpy,
ffmpeg only encodes and muxes.

    pip install numpy pillow        (and install ffmpeg: brew install ffmpeg / apt install ffmpeg)

    python small_hill.py                      # 1280x720, 30 fps  -> small_hill_16x9.mp4
    python small_hill.py --format both        # also 720x1280     -> small_hill_9x16.mp4
    python small_hill.py --res 1080 --fps 60 --mblur 3     # final quality (slow)
    python small_hill.py --still 9.5          # one PNG of t = 9.5 s (fast way to check a look)
    python small_hill.py --sheet              # contact sheet, one frame per beat
    python small_hill.py --res 360 --fps 15   # quick preview

How it is built (same rules as the Motion Reel Kit):
  * the film is a pure function of time: render(t) paints frame t, nothing is carried between frames
  * springs only: closed-form damped springs, a value with several targets = sum of one spring per change
  * seeded randomness only (random.Random(seed)); no timers, no easing curves
  * 120 BPM = 0.5 s per beat, 2 s per bar, 8 bars. Visual events and sounds share the same times.
  * the last frame equals the first, so it loops
"""
import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ----------------------------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------------------------
DUR = 16.0
BPM = 120
BEAT = 60.0 / BPM

PAPER = (243, 239, 230)
INK = (26, 26, 26)
INK2 = (120, 116, 108)
ACC = (255, 92, 31)          # the one accent
LINE = (205, 199, 186)
GROUND = (236, 228, 212)
LOT = (246, 239, 226)
RIVER = (190, 200, 196)
HAZE = (242, 236, 224)
TINTS = [(236, 229, 214), (228, 220, 198), (219, 209, 184), (209, 197, 168)]
FACE = {"S": (236, 219, 194), "N": (248, 235, 215), "E": (222, 203, 176), "W": (243, 228, 205)}
ROOF = (252, 241, 224)
TREE = (150, 165, 122)
TREE_EDGE = (128, 144, 102)
SHADOW = (128, 116, 98)

# ----------------------------------------------------------------------------------------------
# springs (closed form, pure functions of time)
# ----------------------------------------------------------------------------------------------
PRESETS = {
    "snappy": (24.0, 0.80),   # buttons, labels, leading edges
    "default": (13.0, 0.90),  # cards, containers, camera moves
    "heavy": (7.0, 1.00),     # big moves: the 3D tilt
    "pop": (22.0, 0.62),      # visible overshoot: the pin, the callout card
}


def spring(t, t0, preset="default"):
    """Unit step response of a damped spring that starts at t0 (0 -> 1)."""
    x = t - t0
    if x <= 0.0:
        return 0.0
    w, z = PRESETS[preset]
    if z < 1.0:
        wd = w * math.sqrt(1.0 - z * z)
        return 1.0 - math.exp(-z * w * x) * (math.cos(wd * x) + (z * w / wd) * math.sin(wd * x))
    return 1.0 - math.exp(-w * x) * (1.0 + w * x)


def track(t, keys, preset="default"):
    """A value with several targets: the sum of one spring per change, each starting at its own time."""
    return sum(d * spring(t, t0, preset) for t0, d in keys)


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def lerpc(c1, c2, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))


# ----------------------------------------------------------------------------------------------
# fonts (falls back gracefully; override with --font)
# ----------------------------------------------------------------------------------------------
FONT_OVERRIDE = {"path": None}
FONT_PATHS = {
    "bold": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "C:\\Windows\\Fonts\\arialbd.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"],
    "reg": ["/System/Library/Fonts/Supplemental/Arial.ttf", "C:\\Windows\\Fonts\\arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"],
    "cond": ["/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf", "C:\\Windows\\Fonts\\ARIALNB.TTF",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"],
    "condreg": ["/System/Library/Fonts/Supplemental/Arial Narrow.ttf", "C:\\Windows\\Fonts\\ARIALN.TTF",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
}


@lru_cache(maxsize=None)
def font(size, kind="bold"):
    size = max(6, int(size))
    cands = list(FONT_PATHS[kind])
    if FONT_OVERRIDE["path"] and kind in ("bold", "cond"):
        cands.insert(0, FONT_OVERRIDE["path"])
    for p in cands:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


# ----------------------------------------------------------------------------------------------
# the scene
# ----------------------------------------------------------------------------------------------
STREAM_REL = [(130, -138), (118, -88), (150, -38), (128, 12), (60, 42), (-10, 57), (-90, 47),
              (-150, 72), (-190, 132), (-215, 185), (-205, 232)]
LABELS = ["10 m", "20 m", "30 m", "40 m"]
LABEL_IDX = [80, 18, 56, 38]


class Scene:
    def __init__(self, W, H, SS):
        self.W, self.H, self.SS = W, H, SS
        self.portrait = H > W
        self.K = W / 400.0 if self.portrait else W / 548.0         # panel units -> pixels
        self.cy2d = H * 0.46 + 4 * self.K if self.portrait else H / 2 + 4 * self.K
        rr = random.Random(7)
        self.ph = [rr.uniform(0, 6.28318) for _ in range(4)]

        sr = random.Random(11)
        rows = range(-21, 16) if self.portrait else range(-8, 7)
        cells = []
        for gx in range(-9, 9):
            for gy in rows:
                x0, y0 = gx * 58, gy * 52
                cx, cy = x0 + 29, y0 + 26
                if (cx / 150) ** 2 + (cy / 135) ** 2 < 1:
                    continue
                if 200 < cy < 275:
                    continue
                cells.append((cx, cy))
        ylim, xlim = (320, 260) if self.portrait else (190, 520)
        elig = [c for c in cells if abs(c[0]) + abs(c[1]) > 130 and abs(c[1]) < ylim and abs(c[0]) < xlim]
        sr.shuffle(elig)
        plazas = elig[:10]
        pset = set(plazas)
        self.cells = [(cx, cy, clamp(math.hypot(cx, cy) / 430.0)) for cx, cy in cells]
        self.markers = [(cx, cy, i) for i, (cx, cy) in enumerate(plazas)]

        boxes = []
        for cx, cy in cells:
            if (cx, cy) in pset or sr.random() < 0.22:
                continue
            n = 1 if sr.random() < 0.6 else 2
            for i in range(n):
                w = sr.uniform(20, 34)
                dp = sr.uniform(18, 30)
                bx = cx + (sr.uniform(-7, 7) if n == 1 else (-13 if i == 0 else 13))
                by = cy + sr.uniform(-6, 6)
                h = sr.uniform(12, 34)
                if cx > 240 and sr.random() < 0.35:
                    h = sr.uniform(60, 96)
                d = clamp(math.hypot(bx, by) / 430.0)
                boxes.append((bx - w / 2, by - dp / 2, bx + w / 2, by + dp / 2, h, d))
        self.boxes = boxes

        trees = []
        ymin, ymax, nt = (-1050, 700, 330) if self.portrait else (-380, 340, 90)
        while len(trees) < nt:
            x, y = sr.uniform(-480, 480), sr.uniform(ymin, ymax)
            if (x / 140) ** 2 + (y / 125) ** 2 < 1 or 195 < y < 280:
                continue
            trees.append((x, y, sr.uniform(4.5, 7), clamp(math.hypot(x, y) / 430.0), sr.uniform(0, 0.25)))
        self.trees = trees

        # world-space paths
        self.trail = []
        for i in range(140):
            t = i / 139
            a = -1.2 + t * 4.2 * math.pi * 0.75
            rad = 108 * (1 - t) + 8
            self.trail.append((rad * math.cos(a), rad * 0.9 * math.sin(a), rad))
        self.stream = []
        for i in range(len(STREAM_REL) - 1):
            for j in range(12):
                u = j / 12
                a, b = STREAM_REL[i], STREAM_REL[i + 1]
                self.stream.append((lerp(a[0], b[0], u), lerp(a[1], b[1], u) + 3 * math.sin(i + u * 5)))
        self.stream.append(STREAM_REL[-1])

        # depth blur / haze gradients (cached)
        w, h = W * SS, int(H * SS * 0.55)
        col = np.linspace(1.0, 0.0, h)[:, None] * np.ones((1, w))
        self.g_blur = Image.fromarray((col * 255).astype(np.uint8), "L")
        self.g_haze = Image.fromarray((col * 95).astype(np.uint8), "L")

    # ---- helpers -------------------------------------------------------------------------
    def ring_pts(self, k, s, cxk, cyk, n=96):
        R = 9 + (112 - 24 * k - 9) * s
        amp = clamp(s, 0, 1)
        ph = self.ph
        pts = []
        for i in range(n):
            a = i / n * 6.28318
            w = 1 + amp * (0.07 * math.sin(2 * a + ph[0] + k * 0.7) + 0.045 * math.sin(3 * a + ph[1] + k)
                           + 0.025 * math.sin(5 * a + ph[2]))
            pts.append((cxk + math.cos(a) * R * w, cyk + math.sin(a) * R * w))
        return pts

    @staticmethod
    def wipe(img, x, y, w, h, f, fn):
        """Draw something onto a layer and reveal it left to right (a mask wipe, never a fade)."""
        if f <= 0.01 or w < 2 or h < 2:
            return
        layer = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
        fn(ImageDraw.Draw(layer))
        cw = max(1, int(w * min(f, 1.0)))
        c = layer.crop((0, 0, cw, int(h)))
        img.paste(c, (int(x), int(y)), c)

    # ---- the frame -----------------------------------------------------------------------
    def render(self, t):
        W, H, SS, K = self.W, self.H, self.SS, self.K
        U = K * SS                                           # panel unit in canvas pixels

        # camera
        pitch = track(t, [(8.0, 55), (14.0, -55)], "heavy")
        if abs(pitch) < 0.02:
            pitch = 0.0                                     # settle exactly, so the last frame == frame 0
        u = clamp(pitch / 55.0)
        rot_deg = track(t, [(12.0, 14), (14.0, -14)], "default")
        rot = math.radians(0.0 if abs(rot_deg) < 0.02 else rot_deg)
        th = math.radians(pitch)
        ct, st = math.cos(th), math.sin(th)
        cr, sr = math.cos(rot), math.sin(rot)
        zoom = K * (1 - 0.10 * u)
        cx0, cy0 = W / 2, self.cy2d + 22 * K * u

        def PJ(x, y, z=0.0):
            xr = x * cr - y * sr
            yr = x * sr + y * cr
            pf = zoom * max(0.04, 1 + yr * st * 0.0011)
            return ((cx0 + xr * pf) * SS, (cy0 + (yr * ct - z * st) * pf) * SS)

        def depth(x, y):
            return x * sr + y * cr

        def circ(x, y, z, rad, n=22):
            return [PJ(x + rad * math.cos(i / n * 6.2832), y + rad * math.sin(i / n * 6.2832), z) for i in range(n)]

        # global state driven by springs
        g = clamp(track(t, [(9.0, 1), (14.5, -1)]))
        g = 0.0 if g < 1e-3 else g
        hfac = max(0.0, track(t, [(8.5, 1), (14.0, -1)]))
        hh0 = 15.0 * hfac
        ztop = 4 * hh0
        ring_s = [max(0.0, track(t, [(2.0 + 0.5 * k, 1), (15.1 - 0.2 * k, -1)], "snappy")) for k in range(4)]

        img = Image.new("RGB", (W * SS, H * SS), lerpc(PAPER, GROUND, g))
        d = ImageDraw.Draw(img)

        # grid (world space, so it tilts with the ground)
        gw = max(1, int(SS * max(1.0, K / 2.3)))
        gcol = lerpc(LINE, (226, 218, 200), g)
        for i in range(-26, 27):
            d.line([PJ(i * 34, -800), PJ(i * 34, 800)], fill=gcol, width=gw)
            d.line([PJ(-900, i * 34), PJ(900, i * 34)], fill=gcol, width=gw)

        # lots grow out of the grid, staggered outward from the hill
        for cx, cy, dd in self.cells:
            ls = track(t, [(9.0 + dd * 1.2, 1), (14.2 + (1 - dd) * 0.3, -1)])
            if ls <= 0.01:
                continue
            hw, hh_ = 25 * ls, 22 * ls
            d.polygon([PJ(cx - hw, cy - hh_), PJ(cx + hw, cy - hh_), PJ(cx + hw, cy + hh_), PJ(cx - hw, cy + hh_)],
                      fill=LOT)

        # river wipes in along x
        rvp = clamp(track(t, [(9.4, 1), (14.3, -1)]))
        if rvp > 0.01:
            xr_ = lerp(-560, 560, rvp)
            xs = list(range(-560, int(xr_) + 1, 40))
            if len(xs) >= 2:
                top = [PJ(x, 222 + 9 * math.sin(x / 90)) for x in xs]
                bot = [PJ(x, 262 + 9 * math.sin(x / 90)) for x in reversed(xs)]
                d.polygon(top + bot, fill=RIVER)
                if xr_ > 176:
                    d.polygon([PJ(150, 205), PJ(176, 205), PJ(176, 282), PJ(150, 282)], fill=(246, 240, 228))

        # stream draws on (2D), becomes water in 3D
        sp = clamp(track(t, [(4.5, 1), (15.0, -1)]))
        if sp > 0.005:
            n = max(2, int(sp * len(self.stream)))
            pts = [PJ(x, y) for x, y in self.stream[:n]]
            d.line(pts, fill=lerpc((90, 86, 78), RIVER, g), width=int(lerp(2.5, 5.0, u) * U), joint="curve")

        # ---- shadows (one mask, blended once, light from the NW) ----
        bh_of = lambda b: max(0.0, track(t, [(10.0 + b[5] * 1.6, 1), (14.0 + (1 - b[5]) * 0.5, -1)], "snappy"))
        tr_of = lambda tr: max(0.0, track(t, [(10.3 + tr[3] * 1.4 + tr[4], 1), (14.0 + (1 - tr[3]) * 0.4, -1)], "snappy"))
        box_state = [(b, bh_of(b)) for b in self.boxes]
        tree_state = [(tr, tr_of(tr)) for tr in self.trees]
        want_shadow = hfac > 0.05 or any(s > 0.02 for _, s in box_state)
        centers = [(k * 1.5, -k * 1.2) for k in range(4)]
        if want_shadow:
            sm = Image.new("L", img.size, 0)
            sd = ImageDraw.Draw(sm)

            def shp(pts3, h):
                ps = list(pts3) + [(x + h * 0.62, y + h * 0.5) for x, y in pts3]
                hp = _hull([(round(a, 2), round(b, 2)) for a, b in ps])
                if len(hp) > 2:
                    sd.polygon([PJ(x, y) for x, y in hp], fill=255)

            for b, s in box_state:
                if s > 0.02:
                    shp([(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])], b[4] * s)
            for tr, s in tree_state:
                if s > 0.03:
                    x, y, rr = tr[0], tr[1], tr[2]
                    shp([(x - rr * .6, y - rr * .6), (x + rr * .6, y + rr * .6), (x + rr * .6, y - rr * .6),
                         (x - rr * .6, y + rr * .6)], (rr * 1.6 + 4) * s)
            if hfac > 0.05:
                for k in range(4):
                    if ring_s[k] > 0.004:
                        shp(self.ring_pts(k, ring_s[k], *centers[k], n=40), (k + 1) * hh0)
            img = Image.composite(Image.new("RGB", img.size, SHADOW), img, sm.point(lambda v: int(v * 0.34)))
            d = ImageDraw.Draw(img)

        # ---- depth-sorted items: buildings, trees, markers, hill ----
        items = []
        for b, s in box_state:
            if s > 0.02:
                items.append((depth((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), 0, b, s))
        for tr, s in tree_state:
            if s > 0.03:
                items.append((depth(tr[0], tr[1]), 1, tr, s))
        for mx, my, i in self.markers:
            ms = max(0.0, track(t, [(12.0 + 0.15 * i, 1), (14.0 + 0.05 * i, -1)], "snappy"))
            if ms > 0.03:
                items.append((depth(mx, my), 2, (mx, my), ms))
        items.append((0.0, 3, None, 1.0))
        items.sort(key=lambda it: it[0])

        trail_p = clamp(track(t, [(6.0, 1 / 3), (6.33, 1 / 3), (6.66, 1 / 3), (14.4, -1 / 3), (14.6, -1 / 3), (14.8, -1 / 3)]))
        end_x, end_y, _ = self.trail[-1]
        hero_ring = [max(0.0, track(t, [(12.0, 30), (14.0, -30)], "snappy")),
                     max(0.0, track(t, [(12.25, 48), (14.0, -48)], "snappy"))]
        shade_off = 3.0 * (1 - clamp(u * 4))
        shade_p = clamp(track(t, [(4.0, 1)]))
        faces = [((0, 1), "S"), ((0, -1), "N"), ((1, 0), "E"), ((-1, 0), "W")]

        label_pos = []
        for _, kind, dat, s in items:
            if kind == 0:
                x0, y0, x1, y1, h, _d = dat
                h *= s
                cs = {"S": ((x0, y1), (x1, y1)), "N": ((x1, y0), (x0, y0)), "E": ((x1, y1), (x1, y0)),
                      "W": ((x0, y0), (x0, y1))}
                for (nx, ny), key in faces:
                    if nx * sr + ny * cr > 0.02 and h > 0.5:
                        a, b = cs[key]
                        d.polygon([PJ(*a, 0), PJ(*b, 0), PJ(*b, h), PJ(*a, h)], fill=FACE[key])
                d.polygon([PJ(x0, y0, h), PJ(x1, y0, h), PJ(x1, y1, h), PJ(x0, y1, h)], fill=ROOF)
            elif kind == 1:
                x, y, rr = dat[0], dat[1], dat[2]
                z = rr * 1.1 * s
                d.line([PJ(x, y, 0), PJ(x, y, z)], fill=(120, 110, 90), width=SS)
                px, py = PJ(x, y, z + rr * 0.9 * s)
                pf = rr * zoom * s * SS
                d.ellipse([px - pf, py - pf * 0.95, px + pf, py + pf * 0.95], fill=TREE, outline=TREE_EDGE)
            elif kind == 2:
                x, y = dat
                d.polygon(circ(x, y, 0, 15 * s), fill=(250, 246, 238), outline=INK, width=int(2 * SS))
                d.polygon(circ(x, y, 0, 6 * s), fill=INK)
            else:
                # tiny ring at the origin (frame 0)
                d.polygon(circ(0, 0, 0, 9, 40), outline=INK, width=int(1.8 * U))
                # contour rings / terraces
                drawn = []
                for k in range(4):
                    sk = ring_s[k]
                    if sk <= 0.004:
                        continue
                    pts = self.ring_pts(k, sk, *centers[k])
                    drawn.append((k, pts))
                    zk, zp = (k + 1) * hh0, k * hh0
                    n = len(pts)
                    if hfac > 0.05:
                        for i in range(n):
                            a, b = pts[i], pts[(i + 1) % n]
                            mx, my = (a[0] + b[0]) / 2 - centers[k][0], (a[1] + b[1]) / 2 - centers[k][1]
                            L = math.hypot(mx, my) or 1.0
                            nx, ny = mx / L, my / L
                            if nx * sr + ny * cr > 0.0:
                                sh = 0.80 + 0.16 * (nx * -0.6 + ny * -0.8)
                                col = tuple(int(c * sh) for c in TINTS[k])
                                d.polygon([PJ(*a, zp), PJ(*b, zp), PJ(*b, zk), PJ(*a, zk)], fill=col)
                    d.polygon([PJ(x, y, zk) for x, y in pts], fill=TINTS[k])
                    lw = (1.5 + (0.8 if k == 3 else 0.0)) * U
                    d.line([PJ(x, y, zk) for x, y in pts] + [PJ(pts[0][0], pts[0][1], zk)],
                           fill=INK if k % 2 == 0 else INK2, width=max(1, int(lw)), joint="curve")
                # hillshade: SE-offset contour lines revealed by a NW -> SE sweep, melts away as the camera tilts
                if shade_p > 0.01 and shade_off > 0.02:
                    thr = lerp(-250, 250, shade_p)
                    for k, pts in drawn:
                        for i in range(len(pts)):
                            a, b = pts[i], pts[(i + 1) % len(pts)]
                            if (a[0] + a[1]) < thr:
                                d.line([PJ(a[0] + shade_off, a[1] + shade_off), PJ(b[0] + shade_off, b[1] + shade_off)],
                                       fill=(150, 142, 125), width=max(1, int(1.2 * U)))
                for k, pts in drawn:
                    label_pos.append((k, pts[LABEL_IDX[k]]))
                # trail on the terraces
                if trail_p > 0.005:
                    nvis = max(2, int(trail_p * (len(self.trail) - 1)))
                    for i in range(0, nvis - 1, 3):
                        a, b = self.trail[i], self.trail[i + 1]
                        za = (min(3, max(0, int((112 - a[2]) // 24))) + 1) * hh0 + 0.5
                        d.line([PJ(a[0], a[1], za), PJ(b[0], b[1], za)], fill=INK, width=max(1, int(2.0 * U)))
                    ds = max(0.0, track(t, [(6.0, 1), (7.0, -1)], "snappy"))
                    if ds > 0.02:
                        p = self.trail[min(nvis, len(self.trail) - 1)]
                        zp_ = (min(3, max(0, int((112 - p[2]) // 24))) + 1) * hh0 + 0.5
                        px, py = PJ(p[0], p[1], zp_)
                        r_ = 7 * U * ds
                        d.ellipse([px - r_, py - r_, px + r_, py + r_], fill=ACC)
                # pulses around the summit
                for rad in hero_ring:
                    if rad > 1:
                        c_ = circ(end_x, end_y, ztop, rad, 40)
                        d.line(c_ + [c_[0]], fill=(255, 150, 110) if rad < 40 else (255, 190, 160), width=max(1, int(1.6 * U)))

        # ---- depth blur + haze toward the far edge ----
        if u > 0.02:
            h55 = self.g_blur.size[1]
            region = img.crop((0, 0, img.size[0], h55))
            blurred = region.filter(ImageFilter.GaussianBlur(2.4 * U * u))
            region = Image.composite(blurred, region, self.g_blur.point(lambda v: int(v * u)))
            region = Image.composite(Image.new("RGB", region.size, HAZE), region, self.g_haze.point(lambda v: int(v * u)))
            img.paste(region, (0, 0))
            d = ImageDraw.Draw(img)

        # ================= screen-space chrome =================
        ax, ay = PJ(end_x, end_y, ztop)
        c0x, c0y = PJ(0, 0, 0)

        # elevation labels (2D only): wipe in with each ring, wipe out before the tilt
        for k, p in label_pos:
            f = track(t, [(2.15 + 0.5 * k, 1), (7.55 + 0.06 * k, -1)], "snappy")
            if f > 0.01 and u < 0.05:
                lx, ly = PJ(p[0], p[1], 0)
                w_, h_ = (len(LABELS[k]) * 6.6 + 10) * U, 18 * U

                def fn(dd, w_=w_, h_=h_, k=k):
                    dd.rectangle([0, 0, w_, h_], fill=PAPER)
                    dd.text((w_ / 2, h_ / 2), LABELS[k], font=font(10 * U), fill=INK, anchor="mm")
                self.wipe(img, lx - w_ / 2, ly - h_ / 2, w_, h_, f, fn)

        # "+0 m" label and orange crosshair (frame 0 state)
        f0 = track(t, [(2.0, -1), (15.0, 1)], "snappy") + 1.0
        lab_w, lab_h = 46 * U, 18 * U

        def fn0(dd):
            dd.text((0, lab_h / 2), "+0 m", font=font(11 * U), fill=INK, anchor="lm")
        self.wipe(img, c0x + 16 * U, c0y - 40 * U, lab_w, lab_h, f0, fn0)
        arm = max(0.0, 22 + track(t, [(1.0, 26), (2.0, -38), (7.7, -10), (15.0, 22)], "snappy"))
        if arm > 0.5:
            d.line([(c0x - arm * U, c0y), (c0x + arm * U, c0y)], fill=ACC, width=max(1, int(1.5 * U)))
            d.line([(c0x, c0y - arm * U), (c0x, c0y + arm * U)], fill=ACC, width=max(1, int(1.5 * U)))

        # summit pin: drops with a little overshoot, then sinks into the summit marker as the camera tilts
        drop = spring(t, 7.0, "pop") - spring(t, 14.6, "snappy")
        if drop > 0.02:
            e = 1 - spring(t, 8.2, "snappy")
            sc = clamp(drop, 0, 1.25)
            yoff = -80 * U * (1 - min(1.0, spring(t, 7.0, "pop")))
            hx, hy = ax, ay + yoff - 26 * U * e * sc
            if e > 0.03:
                d.polygon([(ax - 9 * U * sc, hy), (ax + 9 * U * sc, hy), (ax, ay + yoff - 2 * U * e * sc)], fill=ACC)
            r1 = 9 * U * sc
            d.ellipse([hx - r1, hy - r1, hx + r1, hy + r1], fill=ACC)
            r2 = 3.5 * U * sc
            d.ellipse([hx - r2, hy - r2, hx + r2, hy + r2], fill=PAPER)

            # badge counts 0 -> 42 m (2D)
            wb = track(t, [(7.2, 1), (7.95, -1)], "snappy")
            if wb > 0.01:
                val = int(round(42 * clamp(spring(t, 7.2, "default"))))
                bw, bh = 70 * U, 24 * U

                def fnb(dd):
                    dd.rounded_rectangle([0, 0, bw, bh], radius=10 * U, fill=INK)
                    dd.text((bw / 2, bh / 2), "%d m" % val, font=font(13 * U), fill=PAPER, anchor="mm")
                self.wipe(img, ax + 16 * U, ay + yoff - 52 * U, bw, bh, wb, fnb)

        # scale bar + north arrow (2D chrome), draw in at 7.25, leave before the tilt
        fs = track(t, [(7.25, 1), (7.95, -1)], "snappy")
        if fs > 0.01:
            sbw, sbh = 96 * U, 34 * U

            def fns(dd):
                dd.line([(0, sbh - 4 * U), (80 * U, sbh - 4 * U)], fill=INK, width=int(2.5 * U))
                dd.line([(0, sbh - 10 * U), (0, sbh + 2 * U)], fill=INK, width=int(2 * U))
                dd.line([(80 * U, sbh - 10 * U), (80 * U, sbh + 2 * U)], fill=INK, width=int(2 * U))
                dd.text((40 * U, sbh - 18 * U), "500 m", font=font(9 * U, "reg"), fill=INK, anchor="mm")
            self.wipe(img, 36 * U, H * SS - 64 * U, sbw, sbh, fs, fns)
            nx_, ny_ = (W - 44 * K) * SS, 34 * U
            hgt = 28 * U * clamp(fs)
            d.polygon([(nx_, ny_), (nx_ - 10 * U, ny_ + hgt), (nx_, ny_ + hgt * 0.8), (nx_ + 10 * U, ny_ + hgt)], fill=INK)
            d.text((nx_, ny_ - 10 * U), "N", font=font(11 * U), fill=INK, anchor="mm")

        # info card (3D half), like the "6:15 am" card in the reference
        fc = track(t, [(8.5, 1), (14.0, -1)], "snappy")
        if fc > 0.01:
            cw_, ch_ = 136 * U, 44 * U

            def fnc(dd):
                dd.rectangle([0, 0, cw_, ch_], fill=INK)
                dd.text((10 * U, 17 * U), "Small Hill", font=font(22 * U / 1.1, "cond"), fill=PAPER, anchor="lm")
                dd.text((10 * U, 34 * U), "Elevation 42 m", font=font(8.5 * U, "condreg"), fill=(190, 186, 178), anchor="lm")
            self.wipe(img, 14 * U, 14 * U, cw_, ch_, fc, fnc)

        # callout card (white, dark border, condensed bold) springs open above the summit
        cs = track(t, [(12.5, 1), (13.8, -1)], "pop")
        if cs > 0.05:
            bw, bh = 140 * U * cs, 60 * U * cs
            bx, by = ax, ay - 70 * U * cs
            d.line([(ax, ay - 10 * U), (bx, by)], fill=INK, width=max(1, int(2 * U)))
            d.rectangle([bx - bw / 2, by - bh, bx + bw / 2, by], fill=(255, 255, 255), outline=INK, width=max(1, int(2 * U * cs)))
            if cs > 0.3:
                d.text((bx, by - bh * 0.68), "Summit", font=font(13 * U * cs, "cond"), fill=INK, anchor="mm")
                d.text((bx, by - bh * 0.28), "42 m", font=font(10 * U * cs, "condreg"), fill=INK, anchor="mm")

        return img


def _hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


# ----------------------------------------------------------------------------------------------
# audio: 120 BPM score + SFX, synthesized with numpy, all hits on the beat grid
# ----------------------------------------------------------------------------------------------
SR = 44100


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)


def synth_audio(path):
    n = int(DUR * SR)
    music = np.zeros(n)
    sfx = np.zeros(n)
    rng = np.random.default_rng(4242)

    def add(buf, start, sig, gain=1.0):
        i0 = int(round(start * SR))
        if i0 >= n or i0 < 0:
            return
        i1 = min(n, i0 + len(sig))
        buf[i0:i1] += gain * sig[:i1 - i0]

    def tt(dur):
        return np.arange(int(dur * SR)) / SR

    def kick():
        x = tt(0.38)
        ph = 2 * np.pi * (45 * x + (95 / 30.0) * (1 - np.exp(-30 * x)))
        return np.sin(ph) * np.exp(-x * 9)

    def bass(f):
        x = tt(0.95)
        return (np.sin(2 * np.pi * f * x) + 0.28 * np.sin(4 * np.pi * f * x) * np.exp(-x * 5)) * np.exp(-x * 3.2)

    def pluck(f):
        x = tt(0.7)
        s = sum((1.0 / h) * np.sin(2 * np.pi * f * h * x) * np.exp(-x * (5 + 3 * h)) for h in range(1, 5))
        return s * np.minimum(1, x / 0.003)

    def hat():
        x = tt(0.09)
        z = rng.standard_normal(len(x))
        return np.diff(z, prepend=0) * np.exp(-x * 55)

    def tick(f=1500.0):
        x = tt(0.07)
        return np.sin(2 * np.pi * f * x) * np.exp(-x * 60)

    def click():
        x = tt(0.03)
        return (np.sin(2 * np.pi * 3200 * x) + 0.5 * rng.standard_normal(len(x))) * np.exp(-x * 160)

    def pop():
        x = tt(0.14)
        ph = 2 * np.pi * (420 * x + (600 / 2.0) * x * x)
        return np.sin(ph) * np.exp(-x * 24)

    def thump():
        x = tt(0.6)
        ph = 2 * np.pi * (52 * x + (70 / 14.0) * (1 - np.exp(-14 * x)))
        return np.sin(ph) * np.exp(-x * 6.5)

    def whoosh(dur, peak):
        m = int(dur * SR)
        z = rng.standard_normal(m)
        x = np.arange(m) / SR
        env = np.sin(np.pi * np.clip(x / dur, 0, 1) ** (math.log(0.5) / math.log(peak / dur))) ** 2
        cutoff = 300 + 5200 * env                          # brighter toward the peak
        a = 1 - np.exp(-2 * np.pi * cutoff / SR)
        y = np.empty(m)
        s = 0.0
        for i in range(m):
            s += a[i] * (z[i] - s)
            y[i] = s
        return y * env * 3.0

    # ---- music -----------------------------------------------------------------------------
    roots = [33, 29, 36, 31]                                   # A1 F1 C2 G1
    chords = [(69, 72, 76), (65, 69, 72), (72, 76, 79), (67, 71, 74)]   # Am F C G
    pattern = [0, 1, 2, 1, 0, 2, 1, 2]
    for bar in range(8):
        b0 = bar * 2.0
        r = midi(roots[bar % 4])
        if bar < 7:
            add(music, b0, bass(r), 0.55)
            add(music, b0 + 1.0, bass(r), 0.45)
            if bar >= 4:
                add(music, b0 + 1.5, bass(r * 1.5), 0.25)
        else:
            add(music, b0, bass(r), 0.5)
        if 1 <= bar <= 7:
            for j in range(8):
                if bar == 7 and j >= 4:
                    break
                note = chords[bar % 4][pattern[j]] + (12 if (j % 4 == 3 and bar >= 4) else 0)
                add(music, b0 + 0.25 * j, pluck(midi(note)), 0.20 if bar < 4 else 0.25)
        if 2 <= bar <= 6:
            for k in range(4):
                add(music, b0 + 0.25 + 0.5 * k, hat(), 0.16)
    for k in range(29):                                        # kicks on every beat, 0 .. 14 s
        add(music, k * BEAT, kick(), 0.85)

    # ---- SFX cues (the same times the picture uses) ------------------------------------------
    cues = [(0.0, "thump", 0.9)]
    cues += [(2.0 + 0.5 * k, "tick", 0.5) for k in range(4)]
    cues += [(4.0, "whoosh1", 0.5)]
    cues += [(6.0, "tick", 0.4), (6.33, "tick2", 0.4), (6.66, "tick3", 0.4)]
    cues += [(7.0, "thump", 0.8), (7.2, "click", 0.5)]
    cues += [(8.0, "thump", 1.0), (8.0, "whoosh2", 0.7)]
    cues += [(10.0 + 0.5 * k, "tick", 0.35) for k in range(4)]
    cues += [(12.0 + 0.15 * i, "click", 0.45) for i in range(10)]
    cues += [(12.5, "pop", 0.8)]
    cues += [(14.0, "whoosh3", 0.7), (15.5, "tick", 0.5)]
    cache = {}
    for tm, kind, gain in cues:
        if kind not in cache:
            cache[kind] = {
                "thump": thump, "click": click, "pop": pop,
                "tick": lambda: tick(1500), "tick2": lambda: tick(1900), "tick3": lambda: tick(2400),
                "whoosh1": lambda: whoosh(1.3, 0.55), "whoosh2": lambda: whoosh(1.6, 0.55),
                "whoosh3": lambda: whoosh(1.4, 0.5),
            }[kind]()
        add(sfx, tm, cache[kind], gain)

    mix = 0.55 * music + 0.75 * sfx
    mix = np.tanh(mix * 1.1)                                    # soft limiter
    mix /= max(1e-6, np.abs(mix).max()) / 0.9
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ----------------------------------------------------------------------------------------------
# rendering / encoding
# ----------------------------------------------------------------------------------------------
_SC = None
_OPT = {}


def _init(W, H, SS, font_path, mblur, shutter, fps):
    global _SC
    FONT_OVERRIDE["path"] = font_path
    _SC = Scene(W, H, SS)
    _OPT.update(mblur=mblur, shutter=shutter, fps=fps, size=(W, H))


def _frame(i):
    fps = _OPT["fps"]
    n = _OPT["mblur"]
    t = i / fps
    if n <= 1:
        im = _SC.render(t)
        return im.resize(_OPT["size"], Image.LANCZOS).tobytes()
    acc = None
    for j in range(n):
        tj = t + ((j + 0.5) / n - 0.5) * _OPT["shutter"] / fps
        a = np.asarray(_SC.render(max(0.0, tj)).resize(_OPT["size"], Image.LANCZOS), dtype=np.float32)
        acc = a if acc is None else acc + a
    return (acc / n).astype(np.uint8).tobytes()


def dims(fmt, res):
    short = int(res) // 2 * 2
    long_ = int(round(short * 16 / 9)) // 2 * 2
    return (long_, short) if fmt == "16x9" else (short, long_)


def encode(fmt, args, audio_path):
    W, H = dims(fmt, args.res)
    N = int(round(DUR * args.fps))
    out = os.path.join(args.out, "small_hill_%s.mp4" % fmt)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(args.fps), "-i", "-"]
    if audio_path:
        cmd += ["-i", audio_path]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]
    if audio_path:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ac", "2",
                "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,apad", "-t", str(DUR)]
    cmd += ["-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    workers = args.workers or max(1, cpu_count() - 1)
    t0 = time.time()
    print("[%s] %dx%d, %d frames, %d workers, supersample %dx, motion blur %d" %
          (fmt, W, H, N, workers, args.ss, args.mblur))
    with Pool(workers, initializer=_init,
              initargs=(W, H, args.ss, args.font, args.mblur, args.shutter, args.fps)) as pool:
        for i, buf in enumerate(pool.imap(_frame, range(N), chunksize=2)):
            proc.stdin.write(buf)
            if i % 10 == 0 or i == N - 1:
                el = time.time() - t0
                eta = el / (i + 1) * (N - i - 1)
                sys.stdout.write("\r   frame %d/%d   elapsed %ds   eta %ds   " % (i + 1, N, el, eta))
                sys.stdout.flush()
    proc.stdin.close()
    proc.wait()
    print("\n   -> %s" % out)
    return out


def main():
    ap = argparse.ArgumentParser(description="Render the Small Hill film (pure Python + ffmpeg).")
    ap.add_argument("--format", choices=["16x9", "9x16", "both"], default="16x9")
    ap.add_argument("--res", type=int, default=720, help="short side in px (720 = 1280x720, 1080 = 1920x1080)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ss", type=int, default=2, help="supersampling for anti-aliasing (1 = fast, 2 = good, 3 = best)")
    ap.add_argument("--mblur", type=int, default=1, help="motion-blur subframes per frame (1 = off, 3-4 = final)")
    ap.add_argument("--shutter", type=float, default=0.5, help="shutter as a fraction of the frame time")
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (default: CPU count - 1)")
    ap.add_argument("--out", default=".", help="output folder")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--font", default=None, help="path to a .ttf to use for display type")
    ap.add_argument("--still", type=float, default=None, help="render one PNG at this time (seconds) and exit")
    ap.add_argument("--sheet", action="store_true", help="write a contact sheet (one frame per beat) and exit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fmts = ["16x9", "9x16"] if args.format == "both" else [args.format]

    if args.still is not None or args.sheet:
        for fmt in fmts:
            W, H = dims(fmt, args.res)
            _init(W, H, args.ss, args.font, 1, 0.5, args.fps)
            if args.still is not None:
                p = os.path.join(args.out, "still_%s_%05.2fs.png" % (fmt, args.still))
                _SC.render(args.still).resize((W, H), Image.LANCZOS).save(p)
                print(p)
            else:
                ts = [i * BEAT for i in range(32)]
                cols, tw = 8, 320 if fmt == "16x9" else 180
                th_ = int(tw * H / W)
                sheet = Image.new("RGB", (cols * tw, (len(ts) // cols) * th_), (14, 14, 14))
                for i, tm in enumerate(ts):
                    im = _SC.render(tm).resize((tw, th_), Image.LANCZOS)
                    sheet.paste(im, ((i % cols) * tw, (i // cols) * th_))
                p = os.path.join(args.out, "contact_sheet_%s.png" % fmt)
                sheet.save(p)
                print(p)
        return

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg was not found. Install it first (macOS: brew install ffmpeg, Ubuntu: sudo apt install ffmpeg).")

    audio = None
    tmp = None
    if not args.no_audio:
        tmp = tempfile.mkdtemp()
        audio = os.path.join(tmp, "score.wav")
        print("synthesizing score + SFX ...")
        synth_audio(audio)
    for fmt in fmts:
        encode(fmt, args, audio)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print("done.")


if __name__ == "__main__":
    main()
