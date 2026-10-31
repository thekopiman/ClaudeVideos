#!/usr/bin/env python3
"""
anim3d.py - a small 3D-looking animation engine in pure Python (Pillow + numpy; ffmpeg only encodes).

What it gives you
  * closed-form springs (spring, track) so every value is a pure function of time t
  * Camera: heading-up perspective tilt (pitch 0 = top view, 55 = tilted), orbit, zoom, pan
  * Frame: painter's-algorithm scene with extruded boxes (flat / gable / hip roofs), domes, onion domes, spires,
    cylinders with cone roofs, trees, people, oriented boxes (cars), shaded height-field TERRAIN with contour
    lines and forest, soft shadows, depth blur + haze, and overlays (drone, dotted routes, tags, cards, wipes)
  * run(): multiprocess renderer -> ffmpeg pipe -> MP4, with --still, --sheet, --check-loop, optional audio

Minimal use (see example_city.py for a complete film):

    from anim3d import *
    class Scene:
        def __init__(self, W, H, SS): self.cam = Camera(W, H, SS)
        def render(self, t):
            cam = self.cam
            cam.set(pitch=track(t, [(3.0, 52)], "heavy"), rot=180, target=(0, 0), zoom=1.2, y_shift=4)
            f = Frame(cam, bg=PAPER)
            f.grid(34)
            f.box(-20, -20, 20, 20, 30, WALLS[0], "gable", rh=10, ridge="x", hf=spring(t, 4, "snappy"))
            f.flush()
            return f.finish()
    def make_scene(W, H, SS): return Scene(W, H, SS)
    if __name__ == "__main__": run(make_scene, duration=8, name="demo")

Conventions
  * world units: x east, y south (screen down), z up. A 720-px-wide frame shows about 720/zoom world units.
  * panel units for UI: 1 unit = frame_width/720 px (cam.K). Multiply by cam.U for supersampled canvas pixels.
  * everything is drawn on a supersampled canvas (SS x); run() downsamples with LANCZOS.
"""
import argparse
import bisect
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ----------------------------------------------------------------------------------------------
# palette (warm paper + one accent). Override freely; keep ONE accent.
# ----------------------------------------------------------------------------------------------
PAPER = (243, 239, 230); INK = (26, 26, 26); INK2 = (120, 116, 108); ACC = (255, 92, 31); LINE = (205, 199, 186)
GROUND = (236, 228, 212); ROAD = (219, 211, 195); LOT = (246, 239, 226); RIVER = (178, 194, 196); BANK = (228, 226, 212)
SHADOW = (128, 116, 98); COPPER = (146, 176, 150); STONE = (216, 205, 188); ROOFC = (226, 204, 184); HAZE = (242, 236, 224)
WALLS = [(244, 232, 212), (240, 222, 184), (238, 212, 198), (222, 226, 204), (232, 228, 220), (246, 236, 200)]
TREE = (150, 165, 122); TREE_EDGE = (128, 144, 102)
LOWC = (238, 232, 214); HIGHC = (224, 216, 186); ROCKC = (234, 226, 212)       # terrain
CAR_COLORS = [(58, 58, 58), (246, 244, 238), (168, 163, 154), (206, 186, 150), (90, 90, 92)]
PERSON_COLORS = [(58, 58, 58), (120, 116, 108), (88, 84, 76), (150, 112, 84), (70, 76, 86)]

# ----------------------------------------------------------------------------------------------
# springs: closed-form step responses, pure functions of time
# ----------------------------------------------------------------------------------------------
PRESETS = {                      # (omega, zeta)
    "snappy": (24.0, 0.80),      # labels, pins, presence, leading edges (tiny overshoot)
    "default": (13.0, 0.90),     # growth, containers
    "heavy": (7.0, 1.00),        # big camera moves (critically damped, no overshoot)
    "pop": (22.0, 0.62),         # visible overshoot: pins, callout cards
    "glide": (6.0, 1.00),        # camera pans and flight legs; overlapping legs make a smooth path
    "return": (11.0, 1.00),      # settling back to frame 0 fast enough to snap exactly
}


def spring(t, t0, preset="default"):
    """0 -> 1 step response of a damped spring that starts at t0."""
    x = t - t0
    if x <= 0.0:
        return 0.0
    w, z = PRESETS[preset]
    if z < 1.0:
        wd = w * math.sqrt(1.0 - z * z)
        return 1.0 - math.exp(-z * w * x) * (math.cos(wd * x) + (z * w / wd) * math.sin(wd * x))
    return 1.0 - math.exp(-w * x) * (1.0 + w * x)


def track(t, keys, preset="default"):
    """A value with several targets: the sum of one spring per change, [(t0, delta), ...]. Never restart a spring."""
    return sum(d * spring(t, t0, preset) for t0, d in keys)


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def lerpc(c1, c2, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))


def smooth(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def shade(c, m):
    return tuple(max(0, min(255, int(v * m))) for v in c)


# ----------------------------------------------------------------------------------------------
# fonts
# ----------------------------------------------------------------------------------------------
FONT_OVERRIDE = {"path": None}
FONT_PATHS = {
    "bold": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "C:\\Windows\\Fonts\\arialbd.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "reg": ["/System/Library/Fonts/Supplemental/Arial.ttf", "C:\\Windows\\Fonts\\arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "cond": ["/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf", "C:\\Windows\\Fonts\\ARIALNB.TTF",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"],
    "condreg": ["/System/Library/Fonts/Supplemental/Arial Narrow.ttf", "C:\\Windows\\Fonts\\ARIALN.TTF",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
    "mono": ["/System/Library/Fonts/Menlo.ttc", "C:\\Windows\\Fonts\\consola.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"],
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


def hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo = []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    up = []
    for p in reversed(pts):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


# ----------------------------------------------------------------------------------------------
# camera
# ----------------------------------------------------------------------------------------------
class Camera:
    """Heading-up perspective camera. pitch 0 = straight down (a map), 55 = tilted. rot 180 puts world -y (north) at
    the bottom of the screen, which makes the route run bottom -> top ("heading up")."""

    def __init__(self, W, H, SS=2):
        self.W, self.H, self.SS = W, H, SS
        self.K = W / 720.0            # px per panel unit
        self.U = self.K * SS          # supersampled px per panel unit
        self.set()

    def set(self, pitch=0.0, rot=180.0, target=(0.0, 0.0), zoom=1.0, y_shift=None, persp=0.0011):
        """pitch in degrees; target = world point at screen centre; zoom in panel units per world unit;
        y_shift (panel units) lowers the centre a little as the camera tilts; default 4 + 20 * tilt, computed AFTER the pitch snap."""
        if abs(pitch) < 0.02:
            pitch = 0.0                 # snap tails so the last frame can equal frame 0
        self.pitch = pitch
        self.u = clamp(pitch / 55.0)    # tilt fraction 0..1
        th = math.radians(pitch)
        self.ct, self.st = math.cos(th), math.sin(th)
        r = math.radians(rot)
        self.cr, self.sr = math.cos(r), math.sin(r)
        self.tx, self.ty = target
        self.zoom = zoom * self.K
        if y_shift is None:
            y_shift = 4.0 + 20.0 * self.u
        self.cy0 = self.H / 2 + self.K * y_shift
        self.persp = persp
        return self

    def PJ(self, x, y, z=0.0):
        """world -> supersampled canvas pixels."""
        x -= self.tx
        y -= self.ty
        xr = x * self.cr - y * self.sr
        yr = x * self.sr + y * self.cr
        pf = self.zoom * max(0.05, 1 + yr * self.st * self.persp)
        return ((self.W / 2 + xr * pf) * self.SS, (self.cy0 + (yr * self.ct - z * self.st) * pf) * self.SS)

    def depth(self, x, y):
        """larger = nearer the camera. Sort items by this, far first."""
        x -= self.tx
        y -= self.ty
        return x * self.sr + y * self.cr

    def rotn(self, nx, ny):
        return (nx * self.cr - ny * self.sr, nx * self.sr + ny * self.cr)

    def vis(self, n):
        """is a face with world normal n=(nx,ny,nz) facing the camera?"""
        nxr, nyr = self.rotn(n[0], n[1])
        return nyr * self.st + n[2] * self.ct > 0.002

    def tone(self, col, n):
        """face colour from its normal: light from the top-left of the screen."""
        nxr, nyr = self.rotn(n[0], n[1])
        lit = nxr * -0.6 + nyr * -0.8
        return shade(col, min(1.06, 0.99 + 0.13 * lit + 0.05 * n[2]))

    def onscreen(self, x, y, margin=0.3):
        sx, sy = self.PJ(x, y, 0)
        W, H = self.W * self.SS, self.H * self.SS
        return -margin * W < sx < (1 + margin) * W and -margin * H < sy < (1.4) * H + 200 * self.U


# ----------------------------------------------------------------------------------------------
# terrain: shaded height field with contour lines
# ----------------------------------------------------------------------------------------------
def make_hill(cx, cy, rx, ry, A=60.0, plateau=0.45, asym=1.0, rot=0.0, seed=1.0):
    """Height function h(x, y): a smooth hill with a flat top (plateau) and a steeper side where v < 0 (cliff).
    Returns h with .bbox and .A attached, ready for Heightfield(h)."""
    ca, sa = math.cos(math.radians(-rot)), math.sin(math.radians(-rot))

    def h(x, y):
        dx, dy = x - cx, y - cy
        u, v = (dx * ca - dy * sa) / rx, (dx * sa + dy * ca) / ry
        if v < 0:
            v *= asym
        r = math.hypot(u, v)
        f = 1 - smooth(plateau, 1.0, r)
        n = 0.07 * math.sin(4.3 * u + seed) * math.sin(3.7 * v + 1.7 * seed) + 0.04 * math.sin(8.1 * u + 2.0 * v)
        return A * max(0.0, min(1.0, f ** 0.9 + n * f * (1 - f) * 4))
    h.bbox = (cx - rx * 1.15, cy - ry * 1.15, cx + rx * 1.15, cy + ry * 1.15)
    h.A = A
    h.center = (cx, cy)
    h.radii = (rx, ry)
    return h


class Heightfield:
    """Precomputed cells (with contour segments) and tree positions for one height function. Build once at import."""

    def __init__(self, fn, cell=3, levels=8, trees=40, seed=1, tree_range=(0.10, 0.93)):
        self.fn, self.A, self.levels = fn, fn.A, levels
        self.center, self.radii = getattr(fn, "center", (0, 0)), getattr(fn, "radii", (50, 50))
        x0, y0, x1, y1 = fn.bbox
        xs = np.arange(x0, x1 + cell, cell)
        ys = np.arange(y0, y1 + cell, cell)
        Z = np.array([[fn(x, y) for x in xs] for y in ys])
        gy, gx = np.gradient(Z, cell, cell)
        step = self.A / (levels + 1)
        lv = [step * k for k in range(1, levels + 1)]
        self.cells = []
        for j in range(len(ys) - 1):
            for i in range(len(xs) - 1):
                z4 = (float(Z[j, i]), float(Z[j, i + 1]), float(Z[j + 1, i + 1]), float(Z[j + 1, i]))
                if max(z4) < 0.15:
                    continue
                X0, X1, Y0, Y1 = float(xs[i]), float(xs[i + 1]), float(ys[j]), float(ys[j + 1])
                P = [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)]
                segs = []
                for li, L in enumerate(lv):
                    pts = []
                    for e in range(4):
                        za, zb = z4[e], z4[(e + 1) % 4]
                        if (za - L) * (zb - L) < 0:
                            t = (L - za) / (zb - za)
                            pa, pb = P[e], P[(e + 1) % 4]
                            pts.append((pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t))
                    for k in range(0, len(pts) - 1, 2):
                        segs.append((li, pts[k], pts[k + 1], L))
                self.cells.append((X0, Y0, X1, Y1, z4, float((gx[j, i] + gx[j + 1, i + 1]) / 2), float((gy[j, i] + gy[j + 1, i + 1]) / 2),
                                   (X0 + X1) / 2, (Y0 + Y1) / 2, sum(z4) / 4 / self.A, segs))
        import random
        rnd = random.Random(int(seed * 10))
        self.trees = []
        cx, cy = self.center
        rx, ry = self.radii
        tries = 0
        while len(self.trees) < trees and tries < 4000:
            tries += 1
            a, rr = rnd.uniform(0, 6.2832), math.sqrt(rnd.uniform(0, 1))
            x, y = cx + math.cos(a) * rr * rx, cy + math.sin(a) * rr * ry
            z = fn(x, y)
            if tree_range[0] < z / self.A < tree_range[1]:
                self.trees.append((x, y, rnd.uniform(3.0, 4.4), z))

    def height(self, x, y):
        return self.fn(x, y)

    def base_polygon(self, n=60):
        cx, cy = self.center
        rx, ry = self.radii
        return [(cx + math.cos(i / n * 6.2832) * rx, cy + math.sin(i / n * 6.2832) * ry) for i in range(n)]


# ----------------------------------------------------------------------------------------------
# frame: one painted image
# ----------------------------------------------------------------------------------------------
class Frame:
    def __init__(self, cam, bg=PAPER):
        self.cam = cam
        self.bg = bg
        self.img = Image.new("RGB", (cam.W * cam.SS, cam.H * cam.SS), bg)
        self.d = ImageDraw.Draw(self.img)
        self.items = []               # (depth, draw_fn, shadow_shape)

    # ---- immediate drawing on the ground ------------------------------------------------------
    def grid(self, spacing=34, color=LINE, extent=700):
        c, w = self.cam, max(1, int(self.cam.SS * max(1.0, self.cam.K / 2.3)))
        for i in range(-int(extent / spacing), int(extent / spacing) + 1):
            self.d.line([c.PJ(i * spacing, -extent), c.PJ(i * spacing, extent)], fill=color, width=w)
            self.d.line([c.PJ(-extent, i * spacing), c.PJ(extent, i * spacing)], fill=color, width=w)

    def poly(self, pts, fill=None, z=0.0, outline=None, w=1.0):
        c = self.cam
        self.d.polygon([c.PJ(x, y, z) for x, y in pts], fill=fill, outline=outline, width=max(1, int(w * c.U / 2)) if outline else 0)

    def path(self, pts, color, w=2.0, z=0.0):
        c = self.cam
        self.d.line([c.PJ(p[0], p[1], p[2] if len(p) > 2 and z is None else z) for p in pts], fill=color, width=max(1, int(w * c.U / 1.3)), joint="curve")

    def ribbon(self, center_pts, width, fill, n=None):
        """a river / road: polygon along a polyline (draw progress by passing a prefix of the points)."""
        L, R = [], []
        for i, p in enumerate(center_pts):
            a, b = center_pts[max(0, i - 1)], center_pts[min(len(center_pts) - 1, i + 1)]
            dx, dy = b[0] - a[0], b[1] - a[1]
            l = math.hypot(dx, dy) or 1
            nx, ny = -dy / l, dx / l
            L.append((p[0] + nx * width / 2, p[1] + ny * width / 2))
            R.append((p[0] - nx * width / 2, p[1] - ny * width / 2))
        if len(L) >= 2:
            self.poly(L + R[::-1], fill=fill)

    # ---- deferred items (painter's algorithm) -----------------------------------------------------------------
    def add(self, depth, fn, shape=None):
        self.items.append((depth, fn, shape))

    def box(self, x0, y0, x1, y1, h, col, roof="flat", rh=0.0, ridge="x", rc=ROOFC, z0=0.0, hf=1.0, pr=1.0, outline=False):
        """Extruded box. h = wall height, roof = flat|gable|hip with rise rh and ridge along 'x' or 'y'.
        hf scales heights (growth springs), pr scales the footprint about its centre (presence)."""
        c = self.cam
        if pr < 0.02:
            return
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not c.onscreen(cx, cy, 0.35):
            return
        hx, hy = (x1 - x0) / 2 * pr, (y1 - y0) / 2 * pr
        x0, x1, y0, y1 = cx - hx, cx + hx, cy - hy, cy + hy
        Hh, RH = h * hf, rh * hf
        d = self

        def draw():
            D = d.d
            faces = [((0, 1, 0), [(x0, y1), (x1, y1)]), ((0, -1, 0), [(x1, y0), (x0, y0)]),
                     ((1, 0, 0), [(x1, y1), (x1, y0)]), ((-1, 0, 0), [(x0, y0), (x0, y1)])]
            for n, (a, b) in faces:
                if c.vis(n) and Hh > 0.3:
                    D.polygon([c.PJ(*a, z0), c.PJ(*b, z0), c.PJ(*b, z0 + Hh), c.PJ(*a, z0 + Hh)], fill=c.tone(col, n))
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            zt = z0 + Hh
            if roof == "flat" or RH < 0.3:
                D.polygon([c.PJ(x0, y0, zt), c.PJ(x1, y0, zt), c.PJ(x1, y1, zt), c.PJ(x0, y1, zt)], fill=c.tone(STONE if roof == "flat" else rc, (0, 0, 1)))
            elif roof == "gable":
                if ridge == "x":
                    sl = [((0, RH, (y1 - y0) / 2), [(x0, y1, zt), (x1, y1, zt), (x1, ym, zt + RH), (x0, ym, zt + RH)]),
                          ((0, -RH, (y1 - y0) / 2), [(x0, y0, zt), (x1, y0, zt), (x1, ym, zt + RH), (x0, ym, zt + RH)])]
                    ends = [((1, 0, 0), [(x1, y0, zt), (x1, y1, zt), (x1, ym, zt + RH)]), ((-1, 0, 0), [(x0, y0, zt), (x0, y1, zt), (x0, ym, zt + RH)])]
                else:
                    sl = [((RH, 0, (x1 - x0) / 2), [(x1, y0, zt), (x1, y1, zt), (xm, y1, zt + RH), (xm, y0, zt + RH)]),
                          ((-RH, 0, (x1 - x0) / 2), [(x0, y0, zt), (x0, y1, zt), (xm, y1, zt + RH), (xm, y0, zt + RH)])]
                    ends = [((0, 1, 0), [(x0, y1, zt), (x1, y1, zt), (xm, y1, zt + RH)]), ((0, -1, 0), [(x0, y0, zt), (x1, y0, zt), (xm, y0, zt + RH)])]
                for n, p in ends:
                    if c.vis(n):
                        D.polygon([c.PJ(*q) for q in p], fill=c.tone(col, n))
                for n, p in sl:
                    l = math.sqrt(sum(v * v for v in n))
                    n = tuple(v / l for v in n)
                    if c.vis(n):
                        D.polygon([c.PJ(*q) for q in p], fill=c.tone(rc, n))
            else:  # hip
                ins = min(x1 - x0, y1 - y0) * 0.42
                a, b = ([(x0 + ins, ym), (x1 - ins, ym)] if ridge == "x" else [(xm, y0 + ins), (xm, y1 - ins)])
                zr = zt + RH
                if ridge == "x":
                    sl = [((0, RH, (y1 - y0) / 2), [(x0, y1, zt), (x1, y1, zt), (b[0], b[1], zr), (a[0], a[1], zr)]),
                          ((0, -RH, (y1 - y0) / 2), [(x1, y0, zt), (x0, y0, zt), (a[0], a[1], zr), (b[0], b[1], zr)]),
                          ((RH, 0, (x1 - x0) / 2), [(x1, y1, zt), (x1, y0, zt), (b[0], b[1], zr)]),
                          ((-RH, 0, (x1 - x0) / 2), [(x0, y0, zt), (x0, y1, zt), (a[0], a[1], zr)])]
                else:
                    sl = [((0, RH, (y1 - y0) / 2), [(x0, y1, zt), (x1, y1, zt), (b[0], b[1], zr)]),
                          ((0, -RH, (y1 - y0) / 2), [(x1, y0, zt), (x0, y0, zt), (a[0], a[1], zr)]),
                          ((RH, 0, (x1 - x0) / 2), [(x1, y1, zt), (x1, y0, zt), (a[0], a[1], zr), (b[0], b[1], zr)]),
                          ((-RH, 0, (x1 - x0) / 2), [(x0, y0, zt), (x0, y1, zt), (b[0], b[1], zr), (a[0], a[1], zr)])]
                for n, p in sl:
                    l = math.sqrt(sum(v * v for v in n)) or 1
                    n = tuple(v / l for v in n)
                    if c.vis(n):
                        D.polygon([c.PJ(*q) for q in p], fill=c.tone(rc, n))
            if outline and c.u < 0.08:
                D.line([c.PJ(x0, y0, zt), c.PJ(x1, y0, zt), c.PJ(x1, y1, zt), c.PJ(x0, y1, zt), c.PJ(x0, y0, zt)], fill=INK, width=max(1, int(1.3 * c.U)))
        self.add(c.depth(cx, cy), draw, ("box", x0, y0, x1, y1, Hh + RH * 0.6))

    def _slices(self, cx, cy, z0, Hh, rfun, col, n=12):
        c = self.cam
        if Hh < 0.2 or not c.onscreen(cx, cy, 0.35):
            return
        d = self

        def draw():
            for i in range(n):
                zf = (i + 0.5) / n
                R = rfun(zf)
                if R <= 0.2:
                    continue
                d.d.polygon([c.PJ(cx + R * math.cos(a / 16 * 6.2832), cy + R * math.sin(a / 16 * 6.2832), z0 + Hh * zf) for a in range(16)], fill=shade(col, 0.94 + 0.14 * zf))
        self.add(c.depth(cx, cy) + 0.01, draw, ("box", cx - 6, cy - 6, cx + 6, cy + 6, Hh))

    def dome(self, cx, cy, z0, R, hf=1.0, pr=1.0, col=COPPER):
        self._slices(cx, cy, z0, R * 0.9 * hf, lambda zf: R * pr * math.sqrt(max(0, 1 - zf ** 2)), col, 10)

    def onion(self, cx, cy, z0, R, H, hf=1.0, pr=1.0, col=COPPER):
        self._slices(cx, cy, z0, H * hf, lambda zf: R * pr * (0.55 + 0.75 * math.sin(math.pi * zf * 0.82)) * (1 - zf ** 2.4) * (1 if zf < 0.97 else 0.1), col, 12)

    def spire(self, cx, cy, hw, z0, H, hf=1.0, pr=1.0, col=STONE):
        self._slices(cx, cy, z0, H * hf, lambda zf: hw * 1.15 * pr * (1 - zf), col, 10)

    def cylinder(self, cx, cy, R, z0, H, col=STONE, cone=0.0, cone_col=ROOFC, hf=1.0, pr=1.0):
        self._slices(cx, cy, z0, H * hf, lambda zf: R * pr, col, 12)
        if cone:
            self._slices(cx, cy, z0 + H * hf, cone * hf, lambda zf: R * pr * 1.12 * (1 - zf), cone_col, 12)

    def tree(self, x, y, r=5.0, z=0.0, s=1.0):
        c = self.cam
        if s < 0.03 or not c.onscreen(x, y, 0.3):
            return
        d = self

        def draw():
            d.d.line([c.PJ(x, y, z), c.PJ(x, y, z + r * 1.1 * s)], fill=(120, 110, 90), width=c.SS)
            px, py = c.PJ(x, y, z + r * 2.0 * s)
            pf = r * c.zoom * c.SS * s
            d.d.ellipse([px - pf, py - pf * 0.95, px + pf, py + pf * 0.95], fill=TREE, outline=TREE_EDGE)
        self.add(c.depth(x, y), draw, ("box", x - 3, y - 3, x + 3, y + 3, r * 1.6 * s + 3 * s))

    def person(self, x, y, z=0.0, col=PERSON_COLORS[0], s=1.0, t=0.0, phase=0.0):
        c = self.cam
        if s < 0.03 or not c.onscreen(x, y):
            return
        d = self

        def draw():
            zz = z + 0.5 * abs(math.sin(2 * math.pi * 1.9 * t + phase))
            d.d.line([c.PJ(x, y, zz), c.PJ(x, y, zz + 9.5 * s)], fill=col, width=max(2, int(3.6 * c.zoom * c.SS)))
            hx, hy = c.PJ(x, y, zz + 11.8 * s)
            r = 2.7 * c.zoom * c.SS * s
            d.d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=(222, 186, 160))
        self.add(c.depth(x, y) + 0.2, draw)

    def obox(self, cx, cy, length, width, ang, h, col, streak=0.0):
        """Oriented box (a car) heading `ang` radians in world space; streak draws a motion line behind it."""
        c = self.cam
        if h < 0.05 or not c.onscreen(cx, cy):
            return
        ca, sa = math.cos(ang), math.sin(ang)
        P = [(cx + ca * a - sa * b, cy + sa * a + ca * b) for a, b in ((length / 2, width / 2), (length / 2, -width / 2), (-length / 2, -width / 2), (-length / 2, width / 2))]
        d = self

        def draw():
            if streak:
                d.d.line([c.PJ(cx - ca * length / 2, cy - sa * length / 2, 0.2), c.PJ(cx - ca * (length / 2 + streak), cy - sa * (length / 2 + streak), 0.2)],
                         fill=(204, 195, 178), width=max(1, int(1.6 * c.U)))
            for i in range(4):
                a, b = P[i], P[(i + 1) % 4]
                ex, ey = b[0] - a[0], b[1] - a[1]
                l = math.hypot(ex, ey) or 1
                n = (ey / l, -ex / l, 0)
                if c.vis(n):
                    d.d.polygon([c.PJ(*a, 0), c.PJ(*b, 0), c.PJ(*b, h), c.PJ(*a, h)], fill=c.tone(col, n))
            d.d.polygon([c.PJ(p[0], p[1], h) for p in P], fill=shade(col, 1.12))
        self.add(c.depth(cx, cy), draw, ("box", cx - length / 2, cy - length / 2, cx + length / 2, cy + length / 2, h))

    def terrain(self, hf, presence=1.0, height=1.0, bg=None, contour_u=0.3):
        """Draw a Heightfield. presence 0..1 grows the hill out of the ground AND reveals contour levels from the
        outside in (level k shows when presence >= k/levels); height 0..1 extrudes it into 3D. Trees ride on top."""
        c = self.cam
        if presence < 0.02:
            return
        bgc = bg or self.bg
        d = self

        def draw():
            ordered = sorted(hf.cells, key=lambda k: c.depth(k[7], k[8]))
            hs = presence
            for (x0, y0, x1, y1, z4, gxx, gyy, cx_, cy_, f, segs) in ordered:
                fe = f * min(presence, 1.0)
                if fe < 0.004 or not c.onscreen(cx_, cy_, 0.4):
                    continue
                gx_, gy_ = gxx * hs, gyy * hs
                nrm = math.sqrt(gx_ * gx_ + gy_ * gy_ + 1)
                nxr, nyr = c.rotn(-gx_ / nrm, -gy_ / nrm)
                lit = nxr * -0.55 + nyr * -0.7 + 0.45 / nrm
                shd = max(0.74, min(1.08, 0.86 + 0.30 * lit))
                base = tuple(lerp(a, b, fe) for a, b in zip(LOWC, HIGHC))
                rock = smooth(0.85, 1.5, math.hypot(gx_, gy_)) * 0.85
                base = tuple(lerp(a, b, rock) for a, b in zip(base, ROCKC))
                col = shade(tuple(int(v) for v in base), shd)
                bl = smooth(0.0, 0.35, fe)
                col = tuple(int(lerp(a, b, bl)) for a, b in zip(bgc, col))
                poly = [c.PJ(x0, y0, z4[0] * height), c.PJ(x1, y0, z4[1] * height), c.PJ(x1, y1, z4[2] * height), c.PJ(x0, y1, z4[3] * height)]
                d.d.polygon(poly, fill=col, outline=col)
                for (li, pa, pb, L) in segs:
                    if (li + 1) / (hf.levels + 1) * 0.95 > presence:
                        continue
                    idx = (li + 1) % 3 == 0
                    if c.u > contour_u and not idx:
                        continue
                    d.d.line([c.PJ(pa[0], pa[1], L * height), c.PJ(pb[0], pb[1], L * height)], fill=INK2 if idx and c.u <= contour_u else (150, 142, 126),
                             width=max(1, int((1.3 if idx else 0.8) * c.U / 1.3)))
        self.add(c.depth(*hf.center) - 0.5, draw, ("poly", hf.base_polygon(), hf.A * height * min(presence, 1.0)))

    def forest(self, hf, scale=1.0, height=1.0):
        for (x, y, r, z) in hf.trees:
            self.tree(x, y, r, z * height, scale)

    # ---- finishing -------------------------------------------------------------------------------------------------
    def flush(self, shadows=True, strength=0.30, sun=(-0.55, -0.45), min_height=2.0):
        """Draw all deferred items far to near, with one soft shadow mask (light from the top-left of the screen when
        rot = 180; change `sun` to move it). Call once, after adding everything."""
        c = self.cam
        if shadows and self.items:
            sm = Image.new("L", self.img.size, 0)
            sd = ImageDraw.Draw(sm)
            any_s = False
            for dep, fn, shp in self.items:
                if not shp:
                    continue
                if shp[0] == "box":
                    _, x0, y0, x1, y1, hgt = shp
                    if hgt < min_height:
                        continue
                    ps = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                else:
                    ps, hgt = shp[1], shp[2]
                    if hgt < min_height:
                        continue
                ps2 = ps + [(x + hgt * sun[0], y + hgt * sun[1]) for x, y in ps]
                hp = hull([(round(a, 2), round(b, 2)) for a, b in ps2])
                if len(hp) > 2:
                    sd.polygon([c.PJ(x, y) for x, y in hp], fill=255)
                    any_s = True
            if any_s:
                self.img = Image.composite(Image.new("RGB", self.img.size, SHADOW), self.img, sm.point(lambda v: int(v * strength)))
                self.d = ImageDraw.Draw(self.img)
        self.items.sort(key=lambda it: it[0])
        for dep, fn, shp in self.items:
            fn()
        self.items = []

    def depth_blur(self, amount, frac=0.36, haze=70):
        """blur + haze toward the far (top) edge; amount 0..1 (use the camera's tilt fraction)."""
        c = self.cam
        if amount <= 0.02:
            return
        W, Hh = self.img.size
        h = int(Hh * frac)
        col = np.linspace(1.0, 0.0, h)[:, None] * np.ones((1, W))
        gb = Image.fromarray((col * 255).astype(np.uint8), "L")
        gh = Image.fromarray((col * haze).astype(np.uint8), "L")
        region = self.img.crop((0, 0, W, h))
        blurred = region.filter(ImageFilter.GaussianBlur(2.2 * c.U * amount))
        region = Image.composite(blurred, region, gb.point(lambda v: int(v * amount)))
        region = Image.composite(Image.new("RGB", region.size, HAZE), region, gh.point(lambda v: int(v * amount)))
        self.img.paste(region, (0, 0))
        self.d = ImageDraw.Draw(self.img)

    def finish(self):
        return self.img

    # ---- overlays (call after flush / depth_blur so they stay sharp) --------------------------------------------------
    def wipe(self, x, y, w, h, f, fn):
        """Reveal a layer left to right: fn(draw) paints a w x h RGBA layer; f 0..1 is how much is shown. A mask wipe,
        never a fade."""
        if f <= 0.01 or w < 2 or h < 2:
            return
        layer = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
        fn(ImageDraw.Draw(layer))
        cw = int(w) if f > 0.995 else max(1, int(w * min(f, 1.0)))
        crop = layer.crop((0, 0, cw, int(h)))
        self.img.paste(crop, (int(x), int(y)), crop)

    def tag(self, text, anchor_world, offset, f, size=10.0):
        """Dark pill with an accent leader that grows from a world point. offset in panel units."""
        c = self.cam
        if f <= 0.02:
            return
        ax, ay = c.PJ(*anchor_world)
        px, py = ax + offset[0] * c.U, ay + offset[1] * c.U
        fnt = font(size * c.U, "mono")
        tw = self.d.textlength(text, font=fnt)
        fp = min(f, 1.0)
        self.d.line([(ax, ay), (lerp(ax, px, fp), lerp(ay, py, fp))], fill=ACC, width=max(1, int(2 * c.U / 1.3)))
        w, h = tw + 20 * c.U, 24 * c.U

        def fn(dd):
            dd.rounded_rectangle([0, 0, w - 1, h - 1], radius=11 * c.U, fill=(20, 20, 20), outline=ACC, width=max(1, int(c.U)))
            dd.text((10 * c.U, h / 2), text, font=fnt, fill=(250, 250, 250), anchor="lm")
        self.wipe(px - 9 * c.U, py - h / 2, w, h, fp, fn)

    def info_card(self, title, sub, f, x=14, y=14, w=196, h=46):
        c = self.cam

        def fn(dd):
            dd.rectangle([0, 0, w * c.U, h * c.U], fill=INK)
            dd.text((10 * c.U, h * 0.43 * c.U), title, font=font(13.5 * c.U, "cond"), fill=PAPER, anchor="lm")
            dd.text((10 * c.U, h * 0.80 * c.U), sub, font=font(6.5 * c.U, "condreg"), fill=(190, 186, 178), anchor="lm")
        self.wipe(x * c.U, y * c.U, w * c.U, h * c.U, f, fn)

    def callout(self, title, sub, anchor_px, box_center_panel, s):
        """White card with a dark border and a leader to anchor_px (canvas px); s is a 'pop' spring (can overshoot)."""
        c = self.cam
        if s < 0.05:
            return
        bx, by = box_center_panel[0] * c.U, box_center_panel[1] * c.U
        bw, bh = 190 * c.U * s, 62 * c.U * s
        self.d.line([(bx, by), anchor_px], fill=INK, width=max(1, int(2 * c.U / 1.3)))
        self.d.rectangle([bx - bw / 2, by - bh, bx + bw / 2, by], fill=(255, 255, 255), outline=INK, width=max(1, int(2 * c.U * s / 1.3)))
        if s > 0.3:
            self.d.text((bx, by - bh * 0.68), title, font=font(13.5 * c.U * s, "cond"), fill=INK, anchor="mm")
            self.d.text((bx, by - bh * 0.28), sub, font=font(9.5 * c.U * s, "condreg"), fill=INK, anchor="mm")

    def dotted(self, pts3, color=ACC, w=2.2, lift=1.0, every=2):
        """dashed polyline through world points (x, y, z); lift 0..1 scales z (route rises as the camera tilts)."""
        c = self.cam
        for i in range(0, len(pts3) - 1, every):
            a, b = pts3[i], pts3[i + 1]
            self.d.line([c.PJ(a[0], a[1], a[2] * lift), c.PJ(b[0], b[1], b[2] * lift)], fill=color, width=max(1, int(w * c.U / 1.3)))

    def drone(self, pos, heading, scale=1.0, trail=None):
        """quad-rotor overlay with ground shadow (offset = height), an accent LED and an accent trail (list of 3D points)."""
        c = self.cam
        if scale < 0.03:
            return
        x, y, z = pos
        shp = [c.PJ(x - z * 0.25 + 7 * scale * math.cos(i / 16 * 6.2832), y - z * 0.20 + 7 * scale * math.sin(i / 16 * 6.2832), 0) for i in range(16)]
        self.d.polygon(shp, fill=(196, 188, 172))
        if trail and len(trail) > 1:
            self.d.line([c.PJ(*p) for p in trail], fill=ACC, width=max(1, int(2 * c.U / 1.3)))
        arm = 15.0 * scale
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            ox = (sx * math.cos(heading) - sy * math.sin(heading)) * arm * 0.75
            oy = (sx * math.sin(heading) + sy * math.cos(heading)) * arm * 0.75
            self.d.line([c.PJ(x, y, z), c.PJ(x + ox, y + oy, z)], fill=INK, width=max(1, int(2.4 * c.U / 1.3)))
            self.d.polygon([c.PJ(x + ox + 8 * scale * math.cos(i / 14 * 6.2832), y + oy + 8 * scale * math.sin(i / 14 * 6.2832), z) for i in range(14)],
                           fill=(240, 236, 226), outline=INK, width=max(1, int(1.4 * c.U / 1.3)))
        self.d.polygon([c.PJ(x + 5.5 * scale * math.cos(i / 14 * 6.2832), y + 5.5 * scale * math.sin(i / 14 * 6.2832), z) for i in range(14)], fill=INK)
        fx, fy = c.PJ(x + math.cos(heading) * 5.5 * scale, y + math.sin(heading) * 5.5 * scale, z)
        r = 2.4 * c.U * scale
        self.d.ellipse([fx - r, fy - r, fx + r, fy + r], fill=ACC)


def polyline_at(pts, cum, s):
    """point + unit tangent at arc length s along a polyline (cum = cumulative lengths). For cars on roads/rivers."""
    s = max(0.0, min(cum[-1], s))
    i = min(bisect.bisect_right(cum, s) - 1, len(pts) - 2)
    t = (s - cum[i]) / (cum[i + 1] - cum[i] + 1e-9)
    a, b = pts[i], pts[i + 1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    l = math.hypot(dx, dy) or 1.0
    return a[0] + dx * t, a[1] + dy * t, dx / l, dy / l


def cumulative(pts):
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    return cum


# ----------------------------------------------------------------------------------------------
# rendering and encoding
# ----------------------------------------------------------------------------------------------
_SC = None
_OPT = {}


def _init(make_scene, W, H, SS, font_path, mblur, shutter, fps):
    global _SC
    FONT_OVERRIDE["path"] = font_path
    _SC = make_scene(W, H, SS)
    _OPT.update(mblur=mblur, shutter=shutter, fps=fps, size=(W, H))


def _frame(i):
    fps, n = _OPT["fps"], _OPT["mblur"]
    t = i / fps
    if n <= 1:
        return _SC.render(t).resize(_OPT["size"], Image.LANCZOS).tobytes()
    acc = None
    for j in range(n):
        tj = t + ((j + 0.5) / n - 0.5) * _OPT["shutter"] / fps
        a = np.asarray(_SC.render(max(0.0, tj)).resize(_OPT["size"], Image.LANCZOS), dtype=np.float32)
        acc = a if acc is None else acc + a
    return (acc / n).astype(np.uint8).tobytes()


def dims(res):
    short = int(res) // 2 * 2
    return int(round(short * 16 / 9)) // 2 * 2, short


def run(make_scene, duration, name="film", fps=30, audio_fn=None, argv=None, default_res=720, beat=0.5):
    """CLI + renderer. make_scene(W, H, SS) must be a module-level function returning an object with render(t) -> PIL image
    (supersampled). audio_fn(path) may write a WAV (see the code-audio-synth skill). Flags: --res --fps --ss --mblur
    --workers --out --no-audio --audio file.wav --font --still T --sheet --check-loop"""
    ap = argparse.ArgumentParser(description="Render %s (pure Python + ffmpeg)." % name)
    ap.add_argument("--res", type=int, default=default_res, help="short side in px (720 = 1280x720)")
    ap.add_argument("--fps", type=int, default=fps)
    ap.add_argument("--ss", type=int, default=2, help="supersampling (1 fast, 2 good)")
    ap.add_argument("--mblur", type=int, default=1, help="motion-blur subframes (1 off, 3-4 final)")
    ap.add_argument("--shutter", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=".")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--audio", default=None, help="use this WAV instead of audio_fn")
    ap.add_argument("--font", default=None)
    ap.add_argument("--still", type=float, default=None, help="write one PNG at this time and exit")
    ap.add_argument("--sheet", action="store_true", help="contact sheet, one frame per beat, and exit")
    ap.add_argument("--check-loop", action="store_true", help="compare frame 0 with the last frame and exit")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    W, H = dims(args.res)
    if args.still is not None or args.sheet or args.check_loop:
        _init(make_scene, W, H, args.ss, args.font, 1, 0.5, args.fps)
        if args.check_loop:
            a = np.asarray(_SC.render(0.0), dtype=float)
            b = np.asarray(_SC.render(duration - 1.0 / args.fps), dtype=float)
            diff = np.abs(a - b).sum(2)
            print("loop seam: max diff %.1f, pixels off by >30: %d  %s" % (diff.max(), int((diff > 30).sum()), "OK (exact)" if diff.max() == 0 else "NOT exact: check settle snaps"))
            return
        if args.still is not None:
            p = os.path.join(args.out, "still_%05.2fs.png" % args.still)
            _SC.render(args.still).resize((W, H), Image.LANCZOS).save(p)
            print(p)
            return
        ts = [i * beat for i in range(int(duration / beat))]
        cols, tw = 8, 320
        th = int(tw * H / W)
        sheet = Image.new("RGB", (cols * tw, ((len(ts) + cols - 1) // cols) * th), (14, 14, 14))
        for i, tm in enumerate(ts):
            sheet.paste(_SC.render(tm).resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
        p = os.path.join(args.out, "contact_sheet.png")
        sheet.save(p)
        print(p)
        return
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg was not found. Install it first (macOS: brew install ffmpeg, Ubuntu: sudo apt install ffmpeg).")
    audio = tmp = None
    if args.audio:
        audio = args.audio
    elif audio_fn and not args.no_audio:
        tmp = tempfile.mkdtemp()
        audio = os.path.join(tmp, "score.wav")
        print("synthesizing audio ...")
        audio_fn(audio)
    N = int(round(duration * args.fps))
    out = os.path.join(args.out, "%s_%dx%d.mp4" % (name, W, H))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(args.fps), "-i", "-"]
    if audio:
        cmd += ["-i", audio]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ac", "2", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,apad", "-t", str(duration)]
    cmd += ["-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    workers = args.workers or max(1, cpu_count() - 1)
    t0 = time.time()
    print("%dx%d, %d frames, %d workers, supersample %dx, motion blur %d" % (W, H, N, workers, args.ss, args.mblur))
    with Pool(workers, initializer=_init, initargs=(make_scene, W, H, args.ss, args.font, args.mblur, args.shutter, args.fps)) as pool:
        for i, buf in enumerate(pool.imap(_frame, range(N), chunksize=2)):
            proc.stdin.write(buf)
            if i % 10 == 0 or i == N - 1:
                el = time.time() - t0
                sys.stdout.write("\r   frame %d/%d   elapsed %ds   eta %ds   " % (i + 1, N, el, el / (i + 1) * (N - i - 1)))
                sys.stdout.flush()
    proc.stdin.close()
    proc.wait()
    print("\n   -> %s" % out)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
