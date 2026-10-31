#!/usr/bin/env python3
"""
berlin_film.py - "Berlin: build-up and fall", a 135 s film (a 90 s design played 1.5x slower, so every card can be read) made entirely in Python (Pillow + numpy, ffmpeg only encodes).

Medieval Cölln to today, with extensive coverage of WWII and the Cold War. A stylized 3D Berlin (not to scale) is the
background; a launch-style info card (max 1/3 of the frame) alternates sides by era. The camera sways slowly for the whole
film and makes ONE fast 360-degree spin during the fall of the Wall (74-78 s). The score and sound effects are
synthesized in code on the same bar grid (80 BPM, bar = 3 s). The film is a pure function of time and loops seamlessly.

Needs: python3, numpy, Pillow, ffmpeg.  Files next to this one: anim3d.py, synth.py, fonts/ (Inter).

    python berlin_film.py                       # full film with audio -> berlin_1280x720.mp4  (use all CPU cores)
    python berlin_film.py --res 540 --ss 1      # quick draft
    python berlin_film.py --still 75.0          # one frame as PNG
    python berlin_film.py --sheet               # contact sheet, one frame per beat
    python berlin_film.py --check-loop          # frame 0 vs last frame
    python berlin_film.py --segment 0 20        # render 0-20 s only (single-core machines: do 5 segments, then concat)
    python berlin_film.py --audio-only score.wav
"""
import os, sys, math, random, time
from functools import lru_cache
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from anim3d import *

TS = 1.5                      # time stretch: the 90 s design is played 1.5x slower
DUR, FPS = 90.0 * TS, 30


def catmull(pts, n=10):
    out = []
    P = [pts[0]] + list(pts) + [pts[-1]]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for j in range(n):
            t = j / n
            out.append(tuple(0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t * t
                                    + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t ** 3) for k in (0, 1)))
    out.append(tuple(pts[-1]))
    return out


# =====================================================================================================
# WORLD (stylized, not to scale; seeded; identical in every process)
# =====================================================================================================
GLASS = (203, 216, 226); NEWC = (236, 234, 228); SLAB = (226, 224, 218)
GREEN_L = (31, 157, 85); BLUE_L = (47, 107, 255); RED_L = (229, 56, 59); AMBER_L = (245, 165, 36)


# ------------------------------------------------------------------------------ geometry helpers
def dseg(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay; L = dx * dx + dy * dy
    t = 0 if L == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def dpoly(p, pts):
    return min(dseg(p, a, b) for a, b in zip(pts, pts[1:]))


def densify(pts, step=5.0):
    out = []
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        for k in range(n):
            out.append((lerp(a[0], b[0], k / n), lerp(a[1], b[1], k / n)))
    out.append(pts[-1])
    return out


# ------------------------------------------------------------------------------ the world
rr = random.Random(11)
CO = (-36, 0); BE = (95, -22); PAL = (6, -11); GATE = (-262, 0); REICH = (-302, -54); MEM = (-254, 50)
TV = (170, -15); POTS = (-317, 122); TEMP = (-110, 300); KWC = (-440, 100); GRUN = (-520, 220); STAD = (-560, 10)
CORE = (-30, 0)

RIV_E = catmull([(300, 75), (200, 52), (130, 28), (88, 14)], 8)
RIV_N = catmull([(88, 14), (58, -40), (0, -52), (-62, -46), (-102, -14)], 8)
RIV_S = catmull([(88, 14), (58, 44), (0, 54), (-62, 48), (-102, 16)], 8)
RIV_W = catmull([(-102, 0), (-170, -18), (-240, -68), (-330, -130), (-430, -205)], 8)
RIVERS = [(RIV_E, 26), (RIV_N, 16), (RIV_S, 16), (RIV_W, 24)]
WALL_PTS = [(-250, -215), (-272, -140), (-272, -40), (-272, 60), (-282, 118), (-190, 140), (-60, 148), (60, 165), (160, 215)]
WALL_N = [(-250, -215), (-120, -245)]
WALL_D = densify(WALL_PTS, 5) + densify(WALL_N, 5)
ROUTE = densify([(-520, 220), (-400, 150), (-300, 85), (-150, 22), (20, 6), (140, 0), (290, -12)], 12)


def in_river(p, m=4):
    return any(dpoly(p, pts) < w / 2 + m for pts, w in RIVERS)


HOUSES = []


def hs(b, h, rh, c, roof, ridge, year, kind='house'):
    HOUSES.append(dict(b=b, h=h, rh=rh, c=c, roof=roof, ridge=ridge, year=year, kind=kind, r=rr.random(), r2=rr.random()))


# medieval Colln (island) and Berlin (north-east bank)
for x in range(-64, 36, 12):
    for y in range(-30, 32, 11):
        cx, cy = x + 6 + rr.uniform(-1, 1), y + 5 + rr.uniform(-1, 1)
        if -20 < cx < 32 and -34 < cy < 12:
            continue                                  # palace site
        if abs(cx + 48) < 6 and abs(cy + 6) < 6:
            continue                                  # church
        if in_river((cx, cy), 5) or rr.random() < 0.05:
            continue
        w, d = rr.uniform(8, 11), rr.uniform(8, 11)
        yr = 1237 if math.hypot(cx - CO[0], cy - CO[1]) < 34 else 1307
        hs((cx - w / 2, cy - d / 2, cx + w / 2, cy + d / 2), rr.uniform(9, 14), 5, rr.randrange(6), 'gable', rr.choice('xy'), yr)
for x in range(62, 150, 12):
    for y in range(-62, 14, 11):
        cx, cy = x + 6 + rr.uniform(-1, 1), y + 5 + rr.uniform(-1, 1)
        if in_river((cx, cy), 5) or rr.random() < 0.12:
            continue
        w, d = rr.uniform(7, 10), rr.uniform(7, 10)
        dd = math.hypot(cx - BE[0], cy - BE[1])
        yr = 1244 if dd < 30 else (1307 if dd < 52 else 1400)
        hs((cx - w / 2, cy - d / 2, cx + w / 2, cy + d / 2), rr.uniform(9, 14), 5, rr.randrange(6), 'gable', rr.choice('xy'), yr)
# baroque Unter den Linden
for x in range(-112, -250, -15):
    for side in (-1, 1):
        for row in (0, 1):
            cy = side * (24 + row * 20) + rr.uniform(-1.5, 1.5)
            if rr.random() < 0.1:
                continue
            yr = 1701 + int(abs(x + 60) / 200 * 90)
            hs((x - 6, cy - 7, x + 6, cy + 7), rr.uniform(15, 22), 6, rr.randrange(6), 'hip', 'x', yr, 'baroque')
# tenement ring 1871-1916
for gx in range(-250, 340, 24):
    for gy in range(-250, 262, 24):
        cx, cy = gx + rr.uniform(-2, 2), gy + rr.uniform(-2, 2)
        if (-75 < cx < 65 and abs(cy) < 55) or in_river((cx, cy), 14) or (cx < -280):
            continue
        if (60 < cx < 155 and -66 < cy < 16) or (-270 < cx < -100 and abs(cy) < 52):
            continue
        if abs(cx + 272) < 14 or (-325 < cx < -280 and -80 < cy < -30) or (-280 < cx < -228 and 20 < cy < 80):
            continue
        d = math.hypot(cx - CORE[0], cy - CORE[1])
        if d > 340 or rr.random() < 0.18:
            continue
        w, dd = rr.uniform(15, 20), rr.uniform(15, 20)
        hs((cx - w / 2, cy - dd / 2, cx + w / 2, cy + dd / 2), rr.uniform(20, 30), 7, rr.randrange(6),
           rr.choice(['flat', 'gable', 'hip']), rr.choice('xy'), 1871 + int(min(d / 330, 1) * 45), 'tenement')
# 1920s outskirts
for gx in range(-250, 430, 30):
    for gy in range(-310, 350, 30):
        cx, cy = gx + rr.uniform(-3, 3), gy + rr.uniform(-3, 3)
        d = math.hypot(cx - CORE[0], cy - CORE[1])
        if d < 335 or d > 470 or in_river((cx, cy), 14) or cx < -285 or rr.random() < 0.3:
            continue
        w, dd = rr.uniform(18, 26), rr.uniform(14, 20)
        hs((cx - w / 2, cy - dd / 2, cx + w / 2, cy + dd / 2), rr.uniform(16, 24), 0, 4, 'flat', 'x', 1920 + int((d - 335) / 135 * 10), 'weimar')
# East slabs 1952-1975
for i, x in enumerate(range(205, 345, 46)):
    for row, y in enumerate((-36, 22)):
        hs((x, y, x + 40, y + 10), 30, 0, 4, 'flat', 'x', 1955 + i * 6 + row * 3, 'slab')
# Potsdamer Platz towers (glass, 1990s)
for i, (x, y, h) in enumerate([(-338, 104, 62), (-318, 104, 50), (-338, 128, 46), (-316, 130, 70)]):
    hs((x, y, x + 15, y + 14), h, 0, 4, 'flat', 'x', 1996, 'glass')
# Tiergarten trees
TREES = []
tr = random.Random(5)
while len(TREES) < 150:
    x, y = tr.uniform(-545, -290), tr.uniform(-125, 130)
    if abs(y) < 10 or (-280 < x < -228 and 20 < y < 82) or dpoly((x, y), RIV_W) < 18 or (-325 < x < -280 and -80 < y < -30):
        continue
    TREES.append((x, y, tr.uniform(3.5, 6)))
# Memorial: stylized 9 x 9 grid (real: 2,711 stelae)
STELAE = []
for i in range(9):
    for j in range(9):
        STELAE.append((-270 + i * 4.3, 30 + j * 4.7, 3.0 + 7.0 * (0.5 + 0.5 * math.sin(i * 1.3 + j * 0.9))))
# crowds / people
CROWD = [(-262 + tr.uniform(-22, 22), tr.uniform(-34, 34)) for _ in range(46)]
SMOKE_SPOTS = [(tr.uniform(-330, 60), tr.uniform(-160, 160)) for _ in range(40)]


# ------------------------------------------------------------------------------ helpers
def tint(col, dim, dark=1.0):
    g = sum(col) // 3
    return shade(lerpc(col, (g, g, g), dim), dark)


def plane(f, x, y, hd, size=9, col=INK, z=0.0):
    c = f.cam; u = c.U
    tip = c.PJ(x + math.cos(hd) * size, y + math.sin(hd) * size, z)
    tail = c.PJ(x - math.cos(hd) * size, y - math.sin(hd) * size, z)
    wl = c.PJ(x + math.cos(hd + 1.57) * size * 0.8, y + math.sin(hd + 1.57) * size * 0.8, z)
    wr = c.PJ(x - math.cos(hd + 1.57) * size * 0.8, y - math.sin(hd + 1.57) * size * 0.8, z)
    mid = c.PJ(x, y, z)
    w = max(1, int(1.7 * u))
    f.d.line([tail, tip], fill=col, width=w + 1)
    f.d.line([wl, wr], fill=col, width=w + 1)
    f.d.ellipse([tip[0] - 1.6 * u, tip[1] - 1.6 * u, tip[0] + 1.6 * u, tip[1] + 1.6 * u], fill=ACC)


def starburst(f, x, y, r=6, z=0.0, col=ACC):
    c = f.cam; u = c.U
    sx, sy = c.PJ(x, y, z)
    for k in range(8):
        a = k * math.pi / 4
        f.d.line([(sx + math.cos(a) * r * 0.4 * u, sy + math.sin(a) * r * 0.4 * u), (sx + math.cos(a) * r * u, sy + math.sin(a) * r * u)], fill=col, width=max(1, int(1.6 * u)))
    f.d.ellipse([sx - r * 0.3 * u, sy - r * 0.3 * u, sx + r * 0.3 * u, sy + r * 0.3 * u], fill=(255, 255, 255))


def arrow(f, a, b, col=ACC, w=2.2):
    c = f.cam; u = c.U
    pa, pb = c.PJ(a[0], a[1], 0), c.PJ(b[0], b[1], 0)
    f.d.line([pa, pb], fill=col, width=max(1, int(w * u)))
    ang = math.atan2(pb[1] - pa[1], pb[0] - pa[0]); L = 8 * u
    f.d.polygon([pb, (pb[0] - math.cos(ang - 0.45) * L, pb[1] - math.sin(ang - 0.45) * L), (pb[0] - math.cos(ang + 0.45) * L, pb[1] - math.sin(ang + 0.45) * L)], fill=col)




# =====================================================================================================
# TIMELINE DATA: 45 bars (2 s each): camera target, year, damage, overlays, tags and the card for every bar
# =====================================================================================================
PANELS = [{'P': {'dim': 0.0,
        'hero': (-36, 0),
        'pitch': 0,
        'ring_at': (-48, -6),
        'side': 'L',
        'tags': [("St. Peter's · 1237 deed", (-48, -6, 0), (70, 34))],
        'year': 1237,
        'zoom': 3.0},
  'card': {'chips': [('28 Oct 1237 · first mention', (47, 107, 255), False), ('On the Fischerinsel, Spree', (31, 157, 85), False)],
           'era': 'MEDIEVAL',
           'h': 'Cölln enters the record',
           'k': '1237',
           's': "A priest's deed names the island town",
           'stat': None},
  'n': 1,
  't0': 0,
  't1': 2,
  'title': 'Cölln'},
 {'P': {'dim': 0.0,
        'hero': (30, -8),
        'pitch': 0,
        'ring_at': (95, -22),
        'side': 'L',
        'tags': [('Berlin · 29 Apr 1244', (95, -22, 0), (60, -34))],
        'year': 1244,
        'zoom': 2.6},
  'card': {'chips': [('29 Apr 1244 · first mention', (47, 107, 255), False), ('City rights: 1251 and 1261', (31, 157, 85), False)],
           'era': 'MEDIEVAL',
           'h': 'Berlin appears across the river',
           'k': '1244',
           's': 'The second settlement is documented',
           'stat': None},
  'n': 2,
  't0': 2,
  't1': 4,
  'title': 'Berlin'},
 {'P': {'dim': 0.0,
        'hero': (30, -8),
        'pitch': 0,
        'ring_at': (88, 14),
        'side': 'L',
        'tags': [('Union · 20 Mar 1307', (88, 14, 0), (60, 36))],
        'year': 1307,
        'zoom': 2.6},
  'card': {'chips': [('20 Mar 1307 · union', (47, 107, 255), False), ('Hanseatic League from 1360', (31, 157, 85), False),
                     ('~8,000 people around 1400', (245, 165, 36), True)],
           'era': 'MEDIEVAL',
           'h': 'Two towns, one council',
           'k': '1307',
           's': 'A shared town hall and shared walls',
           'stat': None},
  'n': 3,
  't0': 4,
  't1': 6,
  'title': 'Twin city'},
 {'P': {'dim': 0.0,
        'hero': (6, -11),
        'pal_pr': 0.8,
        'pitch': 0,
        'ring_at': (6, -11),
        'side': 'L',
        'tags': [('Foundation stone · 1443', (6, -11, 0), (70, 40))],
        'year': 1443,
        'zoom': 3.0},
  'card': {'chips': [("Site of today's Humboldt Forum", (47, 107, 255), False), ('Residence of the electors, 1486', (31, 157, 85), True)],
           'era': 'MEDIEVAL',
           'h': 'The Hohenzollern palace begins',
           'k': '1443',
           's': 'Frederick II lays the foundation stone',
           'stat': None},
  'n': 4,
  't0': 6,
  't1': 8,
  'title': 'The palace'},
 {'P': {'dim': 0.0, 'hero': (6, -11), 'hf': 0.6, 'pitch': 30, 'ring_at': (6, -11), 'side': 'R', 'year': 1701, 'zoom': 2.0},
  'card': {'chips': [('18 Jan 1701 · coronation', (47, 107, 255), False), ('Königsberg crowned him, Berlin rules', (31, 157, 85), True)],
           'era': 'PRUSSIA',
           'h': 'Capital of a new kingdom',
           'k': '1701',
           's': 'Berlin becomes the royal residence',
           'stat': None},
  'n': 5,
  't0': 8,
  't1': 10,
  'title': 'The tilt'},
 {'P': {'dim': 0.0,
        'hero': (-215, 0),
        'pitch': 52,
        'ring_at': (-262, 0),
        'ring_z': 24,
        'side': 'R',
        'tags': [('Brandenburg Gate · 1788-91', (-262, 0, 26), (80, 40))],
        'year': 1791,
        'zoom': 1.7},
  'card': {'chips': [('Built 1788 to 1791', (47, 107, 255), False), ('Quadriga added in 1793', (31, 157, 85), False)],
           'era': 'PRUSSIA',
           'h': 'The Brandenburg Gate rises',
           'k': '1791',
           's': 'Built by Carl Gotthard Langhans',
           'stat': None},
  'n': 6,
  't0': 10,
  't1': 12,
  'title': 'The Gate'},
 {'P': {'dim': 0.0,
        'hero': (-90, 0),
        'pitch': 52,
        'ring_at': (-30, 0),
        'side': 'R',
        'tags': [('Imperial capital · 1871', (-30, 0, 0), (70, 40))],
        'year': 1871,
        'zoom': 1.1},
  'card': {'chips': [('Empire proclaimed in 1871', (47, 107, 255), False), ('826,815 residents in 1871', (245, 165, 36), True)],
           'era': 'EMPIRE',
           'h': 'Capital of the German Empire',
           'k': '1871',
           's': 'Industrial growth fills the ring',
           'stat': None},
  'n': 7,
  't0': 12,
  't1': 14,
  'title': 'Empire'},
 {'P': {'dim': 0.0,
        'hero': (-250, -40),
        'pitch': 52,
        'ring_at': (-302, -54),
        'ring_z': 14,
        'side': 'R',
        'tags': [('Reichstag · 1884', (-302, -54, 26), (80, 40))],
        'year': 1890,
        'zoom': 1.6},
  'card': {'chips': [('Completed in 1884', (47, 107, 255), False), ('Weimar Republic from 1918', (31, 157, 85), False)],
           'era': 'EMPIRE',
           'h': 'The Reichstag is completed',
           'k': '1884',
           's': 'Hohenzollern rule lasts until 1918',
           'stat': None},
  'n': 8,
  't0': 14,
  't1': 16,
  'title': 'Reichstag'},
 {'P': {'dim': 0.0, 'hero': (-30, 0), 'pitch': 52, 'ring': False, 'side': 'L', 'year': 1920, 'zoom': 0.85},
  'card': {'chips': [('8 cities, 59 communities merged', (47, 107, 255), False), ('13 times the area', (31, 157, 85), True)],
           'era': 'WEIMAR',
           'h': 'Berlin becomes a metropolis',
           'k': '1920',
           's': 'The Greater Berlin Act takes effect on 1 Oct',
           'stat': '1.9M → 3.8M'},
  'n': 9,
  't0': 16,
  't1': 18,
  'title': 'Greater Berlin'},
 {'P': {'dim': 0.0, 'hero': (-80, 0), 'pitch': 52, 'ring': True, 'ring_at': (6, -11), 'side': 'L', 'year': 1930, 'zoom': 1.1},
  'card': {'chips': [('Capital of the Weimar Republic', (47, 107, 255), False), ('3.8 million people in 1920', (245, 165, 36), False)],
           'era': 'WEIMAR',
           'h': 'Third-largest city on Earth',
           'k': '1920s',
           's': 'Only London and New York were bigger',
           'stat': None},
  'n': 10,
  't0': 18,
  't1': 20,
  'title': 'Weimar capital'},
 {'P': {'dim': 0.4, 'hero': (-150, 0), 'pitch': 52, 'ring_at': (-262, 0), 'ring_z': 24, 'side': 'L', 'year': 1933, 'zoom': 1.4},
  'card': {'chips': [('160,000 Jewish Berliners', (229, 56, 59), False), ('A third of all German Jews', (229, 56, 59), False)],
           'era': 'NAZI GERMANY',
           'h': 'The Nazis take power',
           'k': '1933',
           's': 'Berlin becomes the capital of Nazi Germany',
           'stat': None},
  'n': 11,
  't0': 20,
  't1': 22,
  'title': '1933'},
 {'P': {'dim': 0.25,
        'hero': (-380, 5),
        'pitch': 52,
        'ring_at': (-560, 10),
        'ring_z': 16,
        'side': 'L',
        'tags': [('Olympic Games · 1936', (-560, 10, 16), (60, 40))],
        'year': 1936,
        'zoom': 0.9},
  'card': {'chips': [('Summer Games held in Berlin', (47, 107, 255), False), ('"No Jews" signs briefly removed', (229, 56, 59), True)],
           'era': 'NAZI GERMANY',
           'h': 'Olympics as a showcase',
           'k': '1936',
           's': 'Visitors saw a staged, temporary calm',
           'stat': None},
  'n': 12,
  't0': 22,
  't1': 24,
  'title': 'Olympics'},
 {'P': {'dim': 0.55,
        'hero': (130, -30),
        'overlays': ['dots'],
        'pitch': 52,
        'ring_at': (134, -42),
        'side': 'R',
        'tags': [('Scheunenviertel', (134, -42, 0), (60, 36))],
        'year': 1938,
        'zoom': 2.2},
  'card': {'chips': [('Thousands of Berlin Jews jailed', (229, 56, 59), False), ('Old quarter near Alexanderplatz', (31, 157, 85), False)],
           'era': 'NAZI GERMANY',
           'h': 'The Kristallnacht pogrom',
           'k': 'Nov 1938',
           's': 'Persecution turns into organised violence',
           'stat': None},
  'n': 13,
  't0': 24,
  't1': 26,
  'title': 'Pogrom'},
 {'P': {'bombers': 9,
        'dim': 0.55,
        'hero': (-100, 0),
        'overlays': ['bombers'],
        'pitch': 52,
        'ring': False,
        'side': 'R',
        'year': 1940,
        'zoom': 1.1},
  'card': {'chips': [('RAF Bomber Command, 1940 on', (229, 56, 59), False), ('Only 9 alerts in all of 1942', (31, 157, 85), False)],
           'era': 'WORLD WAR II',
           'h': '363 air raids on Berlin',
           'k': '1940-45',
           's': 'RAF from 1940, US Air Force from 1943',
           'stat': None},
  'n': 14,
  't0': 26,
  't1': 28,
  'title': 'Sirens'},
 {'P': {'dim': 0.6,
        'hero': (-230, 110),
        'overlays': ['route'],
        'pitch': 52,
        'ring_at': (-520, 220),
        'side': 'R',
        'tags': [('Grunewald station', (-520, 220, 0), (60, 36))],
        'year': 1941,
        'zoom': 0.8},
  'card': {'chips': [('1,013 people on the first train', (229, 56, 59), False), ('Bound for the Łódź ghetto', (229, 56, 59), False)],
           'era': 'THE HOLOCAUST',
           'h': 'First deportation train leaves Grunewald',
           'k': '18 Oct 1941',
           's': 'Berlin Jews are sent east',
           'stat': None},
  'n': 15,
  't0': 28,
  't1': 30,
  'title': 'Gleis 17'},
 {'P': {'dim': 0.7,
        'hero': (-254, 50),
        'overlays': ['stelae'],
        'pitch': 52,
        'ring_at': (-254, 50),
        'ring_z': 10,
        'side': 'R',
        'tags': [('Memorial · opened 2005', (-254, 50, 12), (60, 40))],
        'year': 1941,
        'zoom': 3.2},
  'card': {'chips': [('2,711 stelae in the real memorial', (31, 157, 85), False), ('Opened 10 May 2005', (47, 107, 255), False)],
           'era': 'THE HOLOCAUST',
           'h': 'Over 50,000 Berlin Jews deported',
           'k': '1941-45',
           's': 'Remembered at the Memorial to the Murdered Jews of Europe',
           'stat': None},
  'n': 16,
  't0': 30,
  't1': 32,
  'title': 'Memorial'},
 {'P': {'blasts': 10,
        'bombers': 16,
        'dim': 0.6,
        'dmg': 0.15,
        'hero': (-120, 0),
        'overlays': ['smoke', 'bombers', 'blasts'],
        'pitch': 52,
        'ring': False,
        'side': 'R',
        'smoke_n': 8,
        'year': 1943,
        'zoom': 1.2},
  'card': {'chips': [('16 massed RAF raids by Mar 1944', (229, 56, 59), False),
                     ('22 Nov: 2,000 dead, 175,000 homeless', (229, 56, 59), True)],
           'era': 'WORLD WAR II',
           'h': 'The air battle for Berlin',
           'k': '18 Nov 1943',
           's': 'RAF Bomber Command opens its campaign',
           'stat': None},
  'n': 17,
  't0': 32,
  't1': 34,
  'title': 'Air battle'},
 {'P': {'bombers': 10,
        'dim': 0.65,
        'dmg': 0.4,
        'hero': (-40, 0),
        'overlays': ['smoke', 'bombers'],
        'pitch': 52,
        'ring': False,
        'side': 'R',
        'smoke_n': 14,
        'year': 1944,
        'zoom': 1.1},
  'card': {'chips': [('1.2 million evacuated by 1944', (47, 107, 255), False),
                     ('790,000 of them women and children', (229, 56, 59), False)],
           'era': 'WORLD WAR II',
           'h': 'A quarter of the city leaves',
           'k': '1944',
           's': 'Women and children go to the countryside',
           'stat': None},
  'n': 18,
  't0': 34,
  't1': 36,
  'title': 'Evacuation'},
 {'P': {'blasts': 24,
        'bombers': 28,
        'bseed': 11,
        'dim': 0.7,
        'dmg': 0.6,
        'hero': (-150, -10),
        'overlays': ['smoke', 'bombers', 'blasts'],
        'pitch': 52,
        'ring_at': (-150, 0),
        'side': 'R',
        'smoke_n': 22,
        'tags': [('Government district', (-150, 0, 0), (60, 40))],
        'year': 1945,
        'zoom': 1.5},
  'card': {'chips': [('Nearly 1,000 B-17 bombers', (229, 56, 59), False), ('2,500 dead, 100,000 homeless', (229, 56, 59), True)],
           'era': 'WORLD WAR II',
           'h': 'The heaviest US raid on Berlin',
           'k': '3 Feb 1945',
           's': 'Aimed at the government district',
           'stat': None},
  'n': 19,
  't0': 36,
  't1': 38,
  'title': '3 Feb 1945'},
 {'P': {'dim': 0.7,
        'dmg': 0.7,
        'hero': (110, 20),
        'overlays': ['artillery', 'smoke'],
        'pitch': 52,
        'ring': False,
        'side': 'R',
        'smoke_n': 14,
        'tags': [('Soviet guns · 20,000', (330, 0, 0), (60, -30))],
        'year': 1945,
        'zoom': 0.9},
  'card': {'chips': [('20,000 guns open fire', (229, 56, 59), False), ('Seelow Heights, ~70 km east', (31, 157, 85), False)],
           'era': 'WORLD WAR II',
           'h': 'The Soviet offensive opens',
           'k': '16 Apr 1945',
           's': 'The final push on the Oder front',
           'stat': None},
  'n': 20,
  't0': 38,
  't1': 40,
  'title': 'Barrage'},
 {'P': {'dim': 0.72,
        'dmg': 0.75,
        'hero': (30, 0),
        'overlays': ['advance', 'smoke'],
        'pitch': 52,
        'ring_at': (60, 0),
        'side': 'R',
        'smoke_n': 16,
        'tags': [('Soviet units enter', (60, 0, 0), (60, 40))],
        'year': 1945,
        'zoom': 0.9},
  'card': {'chips': [('City cut off by 27 April', (47, 107, 255), False), ('Fighting goes on until 2 May', (229, 56, 59), True)],
           'era': 'WORLD WAR II',
           'h': 'Soviet troops enter Berlin',
           'k': '21 Apr 1945',
           's': 'House-to-house fighting begins',
           'stat': None},
  'n': 21,
  't0': 40,
  't1': 42,
  'title': 'Street fighting'},
 {'P': {'dim': 0.78,
        'dmg': 0.85,
        'hero': (-220, -10),
        'overlays': ['smoke'],
        'pitch': 52,
        'ring_at': (-302, -54),
        'ring_z': 14,
        'side': 'R',
        'smoke_n': 12,
        'tags': [('Reichstag district', (-302, -54, 12), (60, 40))],
        'year': 1945,
        'zoom': 1.6},
  'card': {'chips': [('30 April 1945', (229, 56, 59), False), ('Street fighting continues to 2 May', (31, 157, 85), False)],
           'era': 'WORLD WAR II',
           'h': 'The regime collapses',
           'k': '30 Apr 1945',
           's': 'Hitler dies by suicide',
           'stat': None},
  'n': 22,
  't0': 42,
  't1': 44,
  'title': 'Last days'},
 {'P': {'dim': 0.8,
        'dmg': 0.9,
        'hero': (-302, -54),
        'overlays': ['flag', 'smoke'],
        'pitch': 52,
        'ring_at': (-302, -54),
        'ring_z': 32,
        'side': 'R',
        'smoke_n': 6,
        'tags': [('Flag on the Reichstag', (-302, -54, 40), (60, 40))],
        'year': 1945,
        'zoom': 1.8},
  'card': {'chips': [('Flag raised on the Reichstag roof', (47, 107, 255), False), ('Soviet dead: over 80,000', (229, 56, 59), True)],
           'era': 'WORLD WAR II',
           'h': 'Berlin surrenders',
           'k': '2 May 1945',
           's': 'The garrison lays down its arms',
           'stat': None},
  'n': 23,
  't0': 44,
  't1': 46,
  'title': 'Surrender'},
 {'P': {'dim': 0.85,
        'dmg': 1.0,
        'hero': (-100, 0),
        'overlays': ['smoke'],
        'pitch': 52,
        'ring': False,
        'side': 'R',
        'smoke_n': 3,
        'year': 1945,
        'zoom': 1.0},
  'card': {'chips': [('Battle of Berlin: 16 Apr to 2 May', (47, 107, 255), False), ('Surrender signed on 7 May', (31, 157, 85), False)],
           'era': 'WORLD WAR II',
           'h': 'The war in Europe ends',
           'k': '7-8 May 1945',
           's': 'Berlin lies in ruins',
           'stat': None},
  'n': 24,
  't0': 46,
  't1': 48,
  'title': 'Rubble'},
 {'P': {'dim': 0.55,
        'dmg': 1.0,
        'hero': (-130, 30),
        'overlays': ['sectors'],
        'pitch': 40,
        'rebuild': 0.1,
        'ring': False,
        'side': 'R',
        'year': 1945,
        'zoom': 0.8},
  'card': {'chips': [('Occupation lasts until 1990', (47, 107, 255), False), ('Gate and Mitte: Soviet sector', (31, 157, 85), False)],
           'era': 'OCCUPIED BERLIN',
           'h': 'Four Allied sectors',
           'k': '1945',
           's': 'Soviet, British, American, French',
           'stat': None},
  'n': 25,
  't0': 48,
  't1': 50,
  'title': 'Four sectors'},
 {'P': {'dim': 0.7,
        'dmg': 1.0,
        'hero': (-440, 100),
        'pitch': 52,
        'rebuild': 0.1,
        'ring_at': (-440, 100),
        'ring_z': 22,
        'side': 'R',
        'tags': [('Kaiser Wilhelm Memorial Church', (-440, 100, 18), (60, 40))],
        'year': 1945,
        'zoom': 3.0},
  'card': {'chips': [('Kaiser Wilhelm Memorial Church', (47, 107, 255), False), ('Preserved on purpose', (31, 157, 85), True)],
           'era': 'OCCUPIED BERLIN',
           'h': 'A ruin kept as a memorial',
           'k': '1945',
           's': 'Left standing after Allied bombing',
           'stat': None},
  'n': 26,
  't0': 50,
  't1': 52,
  'title': 'Ruin as memory'},
 {'P': {'dim': 0.7, 'dmg': 1.0, 'hero': (-130, 30), 'pitch': 25, 'rebuild': 0.15, 'ring': False, 'side': 'R', 'year': 1945, 'zoom': 0.8},
  'card': {'chips': [('RAF 1940 to 1945', (47, 107, 255), False), ('US Air Force 1943 to 1945', (47, 107, 255), False)],
           'era': 'WORLD WAR II',
           'h': 'Air raids on one city',
           'k': '1940-45',
           's': 'Between 1940 and 1945',
           'stat': '363'},
  'n': 27,
  't0': 52,
  't1': 54,
  'title': 'Toll'},
 {'P': {'dim': 0.4,
        'dmg': 1.0,
        'hero': (-170, 60),
        'overlays': ['blockade'],
        'pitch': 45,
        'rebuild': 0.3,
        'ring': False,
        'side': 'L',
        'tags': [('West Berlin (stylized)', (-290, 60, 0), (60, 40))],
        'year': 1948,
        'zoom': 0.8},
  'card': {'chips': [('Rail, road and canal closed', (229, 56, 59), False), ('Two million people cut off', (245, 165, 36), True)],
           'era': 'COLD WAR',
           'h': 'The Soviets blockade West Berlin',
           'k': '24 Jun 1948',
           's': 'All land and water routes are cut',
           'stat': None},
  'n': 28,
  't0': 54,
  't1': 56,
  'title': 'Blockade'},
 {'P': {'dim': 0.35,
        'dmg': 1.0,
        'hero': (-190, 160),
        'overlays': ['planes', 'tempelhof'],
        'pitch': 45,
        'planes_n': 9,
        'rebuild': 0.35,
        'ring_at': (-110, 294),
        'side': 'L',
        'tags': [('Tempelhof airport', (-110, 294, 10), (60, -40))],
        'year': 1948,
        'zoom': 0.75},
  'card': {'chips': [('A landing every minute by 1949', (47, 107, 255), False), ('Operation Vittles', (31, 157, 85), True)],
           'era': 'COLD WAR',
           'h': 'The Airlift feeds a city',
           'k': 'Jun 1948',
           's': 'Day and night, in all weather',
           'stat': None},
  'n': 29,
  't0': 56,
  't1': 58,
  'title': 'Airlift'},
 {'P': {'dim': 0.35,
        'dmg': 1.0,
        'hero': (-190, 170),
        'overlays': ['planes', 'tempelhof'],
        'pitch': 45,
        'planes_n': 16,
        'rebuild': 0.4,
        'ring': False,
        'side': 'L',
        'year': 1948,
        'zoom': 0.75},
  'card': {'chips': [('Easter Parade, 16 Apr 1949', (47, 107, 255), False), ('12,940 tons in 24 hours', (31, 157, 85), True)],
           'era': 'COLD WAR',
           'h': 'Supplies flown into Berlin',
           'k': 'Jun 1948 - May 1949',
           's': 'In roughly 278,000 flights',
           'stat': '2.3M tons'},
  'n': 30,
  't0': 58,
  't1': 60,
  'title': 'Tons'},
 {'P': {'dim': 0.3,
        'dmg': 1.0,
        'hero': (-130, 30),
        'overlays': ['sectors'],
        'pitch': 40,
        'rebuild': 0.45,
        'ring': False,
        'side': 'L',
        'year': 1949,
        'zoom': 0.8},
  'card': {'chips': [('Blockade lifted 12 May 1949', (47, 107, 255), False), ('West Germany: May 1949', (31, 157, 85), False),
                     ('East Germany: 7 Oct 1949', (229, 56, 59), True)],
           'era': 'COLD WAR',
           'h': 'The blockade ends, two states form',
           'k': '1949',
           's': 'West Germany and East Germany',
           'stat': None},
  'n': 31,
  't0': 60,
  't1': 62,
  'title': 'Two states'},
 {'P': {'crowd_dx': 382,
        'crowd_dy': 0,
        'crowd_n': 40,
        'dim': 0.25,
        'dmg': 1.0,
        'hero': (120, 0),
        'overlays': ['crowd'],
        'pitch': 52,
        'rebuild': 0.5,
        'ring': False,
        'side': 'L',
        'year': 1953,
        'zoom': 2.0},
  'card': {'chips': [('Uprising in East Berlin', (229, 56, 59), False), ('Date is a street name today', (31, 157, 85), False)],
           'era': 'COLD WAR',
           'h': 'East Berlin workers rise up',
           'k': '17 Jun 1953',
           's': 'A protest against the government',
           'stat': None},
  'n': 32,
  't0': 62,
  't1': 64,
  'title': '17 June 1953'},
 {'P': {'dim': 0.2,
        'dmg': 1.0,
        'hero': (-170, 60),
        'pitch': 40,
        'rebuild': 0.6,
        'ring_at': (-272, 60),
        'side': 'L',
        'tags': [('Wall · 13 Aug 1961', (-272, 60, 0), (60, 40))],
        'wall': 0.55,
        'year': 1961,
        'zoom': 1.0},
  'card': {'chips': [('Barriers built overnight', (229, 56, 59), False), ('155 km in total', (245, 165, 36), True)],
           'era': 'COLD WAR',
           'h': 'The Wall goes up',
           'k': '13 Aug 1961',
           's': 'East Germany closes the border',
           'stat': None},
  'n': 33,
  't0': 64,
  't1': 66,
  'title': 'The Wall'},
 {'P': {'dim': 0.15,
        'dmg': 1.0,
        'hero': (-272, 10),
        'pitch': 52,
        'rebuild': 0.8,
        'ring_at': (-272, 30),
        'side': 'L',
        'tags': [('Wall · 155 km', (-272, -40, 11), (60, -40))],
        'wall': 1.0,
        'year': 1965,
        'zoom': 1.7},
  'card': {'chips': [('At least 140 died at the Wall', (229, 56, 59), False), ('Over 5,000 escaped over it', (31, 157, 85), False)],
           'era': 'COLD WAR',
           'h': 'A city divided by concrete',
           'k': '28 years',
           's': 'From 1961 to 1989',
           'stat': None},
  'n': 34,
  't0': 66,
  't1': 68,
  'title': 'Death strip'},
 {'P': {'dim': 0.1,
        'dmg': 1.0,
        'hero': (170, -15),
        'pitch': 52,
        'rebuild': 1.0,
        'ring_at': (170, -15),
        'ring_z': 88,
        'side': 'L',
        'tags': [('Fernsehturm · 368 m', (170, -15, 90), (70, 40))],
        'wall': 1.0,
        'year': 1969,
        'zoom': 1.8},
  'card': {'chips': [('Opened 3 Oct 1969', (47, 107, 255), False), ('368 m tall today', (31, 157, 85), True)],
           'era': 'COLD WAR',
           'h': 'A tower for the east',
           'k': '1969',
           's': 'The East German government built it',
           'stat': None},
  'n': 35,
  't0': 68,
  't1': 70,
  'title': 'TV tower'},
 {'P': {'dim': 0.1,
        'dmg': 1.0,
        'hero': (-135, 60),
        'pitch': 45,
        'rebuild': 1.0,
        'ring': False,
        'side': 'L',
        'tags': [('East: GDR capital', (170, -15, 80), (60, -20)), ('West: ruin as memorial', (-440, 100, 22), (60, 30))],
        'wall': 1.0,
        'year': 1975,
        'zoom': 0.66},
  'card': {'chips': [('East Berlin: capital of the GDR', (47, 107, 255), False), ('West: a ruined church stays', (31, 157, 85), False)],
           'era': 'COLD WAR',
           'h': 'Two Berlins, one skyline',
           'k': '1975',
           's': 'The Wall runs between them',
           'stat': None},
  'n': 36,
  't0': 70,
  't1': 72,
  'title': 'Two Berlins'},
 {'P': {'crowd_dx': 30,
        'crowd_n': 26,
        'dim': 0.1,
        'dmg': 1.0,
        'hero': (-150, 0),
        'overlays': ['crowd'],
        'pitch': 52,
        'rebuild': 1.0,
        'ring_at': (-262, 0),
        'ring_z': 24,
        'side': 'L',
        'wall': 1.0,
        'year': 1989,
        'zoom': 1.3},
  'card': {'chips': [('Protests grow in East Germany', (47, 107, 255), False), ('Pan-European Picnic, Aug 1989', (31, 157, 85), False)],
           'era': 'COLD WAR',
           'h': 'Pressure builds across the east',
           'k': 'Summer 1989',
           's': 'Revolutions in Poland and Hungary',
           'stat': None},
  'n': 37,
  't0': 72,
  't1': 74,
  'title': 'Pressure'},
 {'P': {'crowd_n': 46,
        'dim': 0.05,
        'dmg': 1.0,
        'hero': (-262, 0),
        'overlays': ['crowd'],
        'pitch': 52,
        'rebuild': 1.0,
        'ring_at': (-272, 0),
        'side': 'L',
        'tags': [('Border crossings open', (-272, 0, 0), (60, -40))],
        'wall': 1.0,
        'wall_open': True,
        'year': 1989,
        'zoom': 2.4},
  'card': {'chips': [('GDR lifts travel limits', (47, 107, 255), False), ('Crowds gather at the crossings', (31, 157, 85), False)],
           'era': 'COLD WAR',
           'h': 'The border opens',
           'k': '9 Nov 1989',
           's': 'Crowds climb onto the Wall',
           'stat': None},
  'n': 38,
  't0': 74,
  't1': 76,
  'title': '9 Nov 1989'},
 {'P': {'crowd_n': 20,
        'dim': 0.0,
        'dmg': 1.0,
        'hero': (-272, 0),
        'overlays': ['crowd'],
        'pitch': 52,
        'rebuild': 1.0,
        'ring': False,
        'side': 'L',
        'wall': 1.0,
        'wall_cut': (25, 110),
        'year': 1990,
        'zoom': 1.4},
  'card': {'chips': [('Demolition began 13 Jun 1990', (47, 107, 255), False), ('Completed in 1994', (31, 157, 85), False)],
           'era': 'COLD WAR',
           'h': 'The Wall is taken down',
           'k': '1990-94',
           's': 'Demolition begins in June 1990',
           'stat': None},
  'n': 39,
  't0': 76,
  't1': 78,
  'title': 'Tearing down'},
 {'P': {'dim': 0.0,
        'dmg': 1.0,
        'hero': (-80, 0),
        'pitch': 52,
        'rebuild': 1.0,
        'ring': False,
        'scar': True,
        'side': 'L',
        'year': 1990,
        'zoom': 0.95},
  'card': {'chips': [('East and West unite', (47, 107, 255), False), ('The scar line stays visible', (31, 157, 85), False)],
           'era': 'COLD WAR',
           'h': 'Germany is reunified',
           'k': '3 Oct 1990',
           's': 'Berlin is one city again',
           'stat': None},
  'n': 40,
  't0': 78,
  't1': 80,
  'title': 'Reunified'},
 {'P': {'dim': 0.0,
        'dmg': 1.0,
        'hero': (-250, -40),
        'pitch': 52,
        'rebuild': 1.0,
        'ring_at': (-302, -54),
        'ring_z': 14,
        'scar': True,
        'side': 'R',
        'tags': [('Bundestag vote · 20 Jun 1991', (-302, -54, 26), (80, 40))],
        'year': 1991,
        'zoom': 1.8},
  'card': {'chips': [('A very close vote', (47, 107, 255), False), ('The move came in 1999', (31, 157, 85), False)],
           'era': 'REUNIFIED',
           'h': 'Parliament votes for Berlin',
           'k': '20 Jun 1991',
           's': 'The seat of government leaves Bonn',
           'stat': None},
  'n': 41,
  't0': 80,
  't1': 82,
  'title': 'Capital vote'},
 {'P': {'dim': 0.0,
        'dmg': 1.0,
        'hero': (-302, -54),
        'pitch': 52,
        'rebuild': 1.0,
        'ring_at': (-302, -54),
        'ring_z': 34,
        'scar': True,
        'side': 'R',
        'tags': [('Rebuilt Reichstag · 1999', (-302, -54, 36), (80, 40))],
        'year': 1999,
        'zoom': 2.0},
  'card': {'chips': [('Bundestag moved in April 1999', (47, 107, 255), False), ('Glass dome on the roof', (31, 157, 85), True)],
           'era': 'REUNIFIED',
           'h': 'Parliament moves into the Reichstag',
           'k': '1999',
           's': 'The rebuilt building opens its doors',
           'stat': None},
  'n': 42,
  't0': 82,
  't1': 84,
  'title': 'Reichstag'},
 {'P': {'dim': 0.0,
        'dmg': 1.0,
        'hero': (-254, 50),
        'pitch': 52,
        'rebuild': 1.0,
        'ring_at': (-254, 50),
        'ring_z': 10,
        'scar': True,
        'side': 'R',
        'tags': [('Memorial · 10 May 2005', (-254, 50, 12), (80, 40))],
        'year': 2005,
        'zoom': 2.6},
  'card': {'chips': [('2,711 concrete stelae', (47, 107, 255), False), ('Design by Peter Eisenman', (31, 157, 85), False)],
           'era': 'REUNIFIED',
           'h': 'A memorial at the Gate',
           'k': '10 May 2005',
           's': 'On the former Wall border strip',
           'stat': None},
  'n': 43,
  't0': 84,
  't1': 86,
  'title': 'The memorial'},
 {'P': {'dim': 0.0,
        'dmg': 1.0,
        'hero': (-120, 0),
        'pitch': 52,
        'rebuild': 1.0,
        'ring': False,
        'scar': True,
        'side': 'R',
        'year': 2024,
        'zoom': 0.95},
  'card': {'chips': [('Largest city in the EU', (47, 107, 255), False), ('Memorials mark the old line', (31, 157, 85), False)],
           'era': 'BERLIN',
           'h': 'A capital, whole again',
           'k': 'Today',
           's': 'People living in the city',
           'stat': '3.9M'},
  'n': 44,
  't0': 86,
  't1': 88,
  'title': 'Today'},
 {'P': {'dim': 0.0,
        'hero': (-36, 0),
        'pitch': 0,
        'ring_at': (-48, -6),
        'side': 'L',
        'tags': [("St. Peter's · 1237 deed", (-48, -6, 0), (70, 34))],
        'year': 1237,
        'zoom': 3.0},
  'card': {'chips': [('28 Oct 1237 · first mention', (47, 107, 255), False), ('On the Fischerinsel, Spree', (31, 157, 85), False)],
           'era': 'MEDIEVAL',
           'h': 'Cölln enters the record',
           'k': '1237',
           's': "A priest's deed names the island town",
           'stat': None},
  'n': 45,
  't0': 88,
  't1': 90,
  'title': 'Loop'}]

# =====================================================================================================
# TIMELINE: every parameter is a sum of springs (one per panel change). Nothing is stored between frames.
# =====================================================================================================
PRESETS['slow'] = (4.5, 1.0)
SPIN0, SPIN1 = 74.0, 78.0


def rot_at(t):
    """heading: slow sway (+-30 deg, tapered to 0 before the end so the loop closes) + one eased 360 spin 74-78 s."""
    x = max(0.0, min(1.0, (t - SPIN0) / (SPIN1 - SPIN0)))
    spin = x * x * (3 - 2 * x)
    sway = 30 * math.sin(2 * math.pi * t / 90) * (1 - smooth(88.0, 89.5, t))
    return 90 + sway + (360 * spin if spin < 1.0 else 0.0)   # a finished spin is the same heading


def sp(t, t0, pre='default'):
    v = spring(t, t0, pre)
    return 1.0 if 0.9995 < v < 1.0005 else (0.0 if abs(v) < 1e-4 else v)


def _track(getter, preset='glide'):
    vals = [getter(p) for p in PANELS]
    keys = []; prev = vals[0]
    for i in range(1, len(PANELS)):
        d = vals[i] - prev; prev = vals[i]
        if abs(d) > 1e-9:
            pr = preset(PANELS[i]) if callable(preset) else preset
            keys.append((PANELS[i]['t0'], d, 'return' if i == len(PANELS) - 1 else pr))
    return (vals[0], keys, vals[-1])


def tv(tr, t, tol=0.0):
    v0, keys, vl = tr
    x = v0
    for t0, d, pr in keys:
        if t > t0:
            x += d * spring(t, t0, pr)
    if t > 88.2 and abs(x - vl) < max(tol, 1e-3):
        x = vl
    return x


def _ov(p, name):
    return name in p['P'].get('overlays', [])


TR = {}
TR['zoom'] = _track(lambda p: p['P']['zoom'])
TR['hx'] = _track(lambda p: p['P']['hero'][0]); TR['hy'] = _track(lambda p: p['P']['hero'][1])
TR['pitch'] = _track(lambda p: 52 if p['n'] == 5 else p['P'].get('pitch', 52), lambda p: 'heavy' if p['n'] == 5 else 'glide')
TR['dim'] = _track(lambda p: p['P'].get('dim', 0.0)); TR['dmg'] = _track(lambda p: p['P'].get('dmg', 0.0))
TR['reb'] = _track(lambda p: p['P'].get('rebuild', 0.0))
TR['wall'] = _track(lambda p: 1.0 if 40 <= p['n'] <= 44 else p['P'].get('wall', 0.0))
TR['scar'] = _track(lambda p: 1.0 if p['P'].get('scar') else 0.0)
TR['year'] = _track(lambda p: p['P']['year'], 'slow')
TR['rx'] = _track(lambda p: p['P'].get('ring_at', p['P']['hero'])[0], 'default')
TR['ry'] = _track(lambda p: p['P'].get('ring_at', p['P']['hero'])[1], 'default')
TR['rz'] = _track(lambda p: p['P'].get('ring_z', 0), 'default')
TR['rs'] = _track(lambda p: 1.0 if p['P'].get('ring', True) else 0.0, 'pop')
TR['sf'] = _track(lambda p: 1.0 if p['P']['side'] == 'L' else -1.0)
TR['smoke_n'] = _track(lambda p: p['P'].get('smoke_n', 12) if _ov(p, 'smoke') else 0, 'default')
TR['bombers'] = _track(lambda p: p['P'].get('bombers', 18) if _ov(p, 'bombers') else 0, 'default')
TR['blasts'] = _track(lambda p: p['P'].get('blasts', 14) if _ov(p, 'blasts') else 0, 'default')
TR['planes_n'] = _track(lambda p: p['P'].get('planes_n', 9) if _ov(p, 'planes') else 0, 'default')
TR['crowd_n'] = _track(lambda p: p['P'].get('crowd_n', 46) if _ov(p, 'crowd') else 0, 'default')
TR['crowd_dx'] = _track(lambda p: p['P'].get('crowd_dx', 0), 'default')
for _nm in ('artillery', 'advance', 'tempelhof', 'sectors', 'blockade', 'route', 'flag', 'dots', 'stelae'):
    TR['ov_' + _nm] = _track(lambda p, _nm=_nm: 1.0 if _ov(p, _nm) else 0.0, 'default')

_kc = min(range(len(WALL_D)), key=lambda k: math.hypot(WALL_D[k][0] + 272, WALL_D[k][1]))
_MAXD = max(_kc, len(WALL_D) - _kc)
DEMOL_KEYS = [(74.3, 45, 'glide'), (75.6, 60, 'glide'), (77.0, _MAXD - 105 + 14, 'glide')]


def demol(t):
    return sum(d * spring(t, t0, pr) for t0, d, pr in DEMOL_KEYS if t > t0)


# =====================================================================================================
# CARD (launch-motion-video look): white soft-shadow card, kicker pill, headline, grey line, chips, cursor
# =====================================================================================================
@lru_cache(maxsize=None)
def FT(name, size):
    return ImageFont.truetype(os.path.join(HERE, 'fonts', name), max(6, int(size)))


class Cards:
    CW = 226; PAD = 15

    def __init__(self, U):
        self.U = U; self.lay = {}; self.shadow = {}
        self.probe = ImageDraw.Draw(Image.new('RGB', (10, 10)))

    def _wrap(self, text, fnt, width):
        out, cur = [], ''
        for w in text.split():
            t = (cur + ' ' + w).strip()
            if self.probe.textlength(t, font=fnt) > width * self.U and cur:
                out.append(cur); cur = w
            else:
                cur = t
        if cur:
            out.append(cur)
        return out or ['']

    def layout(self, i):
        if i in self.lay:
            return self.lay[i]
        U = self.U; c = PANELS[i]['card']; inner = self.CW - 2 * self.PAD
        f_k, f_h, f_s, f_c, f_st = FT('Inter-SemiBold.ttf', 9.5 * U), FT('InterDisplay-Medium.ttf', 19.5 * U), FT('Inter-Regular.ttf', 9.5 * U), FT('Inter-Medium.ttf', 8.7 * U), FT('InterDisplay-Medium.ttf', 33 * U)
        y = self.PAD; L = dict(kw=self.probe.textlength(c['k'], font=f_k) / U + 20, ky=y)
        y += 34
        if c.get('stat'):
            L['sy'] = y; y += 40
        L['h'] = []
        for ln in self._wrap(c['h'], f_h, inner):
            L['h'].append((y, ln)); y += 22
        y += 4; L['s'] = []
        for ln in self._wrap(c['s'], f_s, inner):
            L['s'].append((y, ln)); y += 13
        y += 10; L['chips'] = []
        for (txt, col, dark) in c.get('chips', []):
            lines = self._wrap(txt, f_c, inner - 30); h = 21 if len(lines) == 1 else 33
            tw = max(self.probe.textlength(l, font=f_c) for l in lines) / U
            L['chips'].append(dict(y=y, lines=lines, col=col, dark=dark, h=h, w=tw + 36)); y += h + 6
        L['CH'] = y + self.PAD - 4
        sh = Image.new('RGBA', (int((self.CW + 70) * U), int((L['CH'] + 70) * U)), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle([35 * U, 41 * U, (35 + self.CW) * U, (41 + L['CH']) * U], radius=18 * U, fill=(0, 0, 0, 62))
        self.shadow[i] = sh.filter(ImageFilter.GaussianBlur(11 * U))
        self.lay[i] = L
        return L

    def draw(self, img, i, side, pr, anchor):
        """pr: dict of progress values 0..1 (slide, out, kick, head, sub, chips[], cursor, conn)."""
        U = self.U; L = self.layout(i); c = PANELS[i]['card']; CW = self.CW; CH = L['CH']; PAD = self.PAD
        dirn = -1 if side == 'L' else 1
        off = ((1 - pr['slide']) + pr['out']) * (CW + 46) * dirn
        x0 = (14 if side == 'L' else 720 - 14 - CW) + off; y0 = 14.0
        img.paste(self.shadow[i], (int((x0 - 35) * U), int((y0 - 35) * U)), self.shadow[i])
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([x0 * U, y0 * U, (x0 + CW) * U, (y0 + CH) * U], radius=16 * U, fill=(255, 255, 255), outline=(232, 232, 232), width=max(1, int(U)))
        X = x0 + PAD
        f_k, f_h, f_s, f_c, f_st, f_e = (FT('Inter-SemiBold.ttf', 9.5 * U), FT('InterDisplay-Medium.ttf', 19.5 * U), FT('Inter-Regular.ttf', 9.5 * U),
                                        FT('Inter-Medium.ttf', 8.7 * U), FT('InterDisplay-Medium.ttf', 33 * U), FT('Inter-Medium.ttf', 8.5 * U))
        # kicker pill pops
        k = max(0.0, pr['kick']); kw = L['kw'] * k; ya = y0 + L['ky']
        if k > 0.02:
            d.rounded_rectangle([X * U, (ya + 11 * (1 - k)) * U, (X + kw) * U, (ya + 22 - 11 * (1 - k)) * U], radius=11 * U * k, fill=(10, 10, 10))
            if k > 0.85:
                d.text(((X + kw / 2) * U, (ya + 11) * U), c['k'], font=f_k, fill=(255, 255, 255), anchor='mm')
                d.text(((X + kw + 9) * U, (ya + 11) * U), c.get('era', ''), font=f_e, fill=(107, 107, 107), anchor='lm')
        if c.get('stat') and pr['head'] > 0.05:
            s_ = min(1.0, pr['head'])
            d.text((X * U, (y0 + L['sy'] + 2 * (1 - s_)) * U), c['stat'], font=FT('InterDisplay-Medium.ttf', 33 * U * (0.6 + 0.4 * s_)), fill=(10, 10, 10), anchor='la')
        # headline + sub: left-to-right wipe with a small rise
        for (key, fnt, col, rows) in (('head', f_h, (10, 10, 10), L['h']), ('sub', f_s, (107, 107, 107), L['s'])):
            p = max(0.0, min(1.0, pr[key]))
            if p < 0.02:
                continue
            for (yy, ln) in rows:
                w = int((CW - 2 * PAD) * U); hh = int(26 * U)
                layer = Image.new('RGBA', (w, hh), (0, 0, 0, 0))
                ImageDraw.Draw(layer).text((0, 0), ln, font=fnt, fill=col + (255,), anchor='la')
                cw = w if p > 0.995 else max(1, int(w * p))
                crop = layer.crop((0, 0, cw, hh))
                img.paste(crop, (int(X * U), int((y0 + yy + 7 * (1 - p)) * U)), crop)
        d = ImageDraw.Draw(img)
        for j, ch in enumerate(L['chips']):
            s_ = pr['chips'][j]
            if s_ < 0.03:
                continue
            ya = y0 + ch['y']; w = ch['w'] * s_; hh = ch['h'] * s_; cy = ya + ch['h'] / 2
            d.rounded_rectangle([X * U, (cy - hh / 2) * U, (X + w) * U, (cy + hh / 2) * U], radius=min(10.5, hh / 2) * U,
                                fill=(10, 10, 10) if ch['dark'] else (255, 255, 255), outline=(10, 10, 10) if ch['dark'] else (226, 226, 226), width=max(1, int(U)))
            if s_ > 0.8:
                d.ellipse([(X + 6) * U, (cy - 6) * U, (X + 18) * U, (cy + 6) * U], fill=(255, 255, 255) if ch['dark'] else (244, 244, 244))
                d.ellipse([(X + 9.5) * U, (cy - 2.5) * U, (X + 14.5) * U, (cy + 2.5) * U], fill=ch['col'])
                for q, ln in enumerate(ch['lines']):
                    yl = ya + (ch['h'] / 2 if len(ch['lines']) == 1 else 10 + q * 12.5)
                    d.text(((X + 25) * U, yl * U), ln, font=f_c, fill=(255, 255, 255) if ch['dark'] else (10, 10, 10), anchor='lm')
        # blue cursor glides to the last chip
        if L['chips'] and pr['cursor'] > 0.02:
            ch = L['chips'][-1]; tx_, ty_ = X + ch['w'] - 6, y0 + ch['y'] + ch['h'] - 4
            sx_, sy_ = x0 + CW - 20, y0 + 10; q = pr['cursor']
            cx_, cy_ = lerp(sx_, tx_, q), lerp(sy_, ty_, q)
            d.polygon([((cx_ + a) * U, (cy_ + b) * U) for a, b in [(0, 0), (3.8, 12.6), (6.3, 6.9), (12.6, 4.4)]], fill=BLUE_L, outline=(255, 255, 255))
        # green dashed connector to the hero ring
        if anchor is not None and L['chips'] and pr['conn'] > 0.02:
            sx = (x0 + CW if side == 'L' else x0) * U; sy = (y0 + L['chips'][-1]['y'] + L['chips'][-1]['h'] / 2) * U
            ax, ay = anchor; Ld = math.hypot(ax - sx, ay - sy)
            if Ld > 30 * U:
                ux, uy = (ax - sx) / Ld, (ay - sy) / Ld; end = (Ld - 13 * U) * min(1.0, pr['conn']); pos = 0.0
                while pos < end:
                    b = min(end, pos + 5 * U)
                    d.line([(sx + ux * pos, sy + uy * pos), (sx + ux * b, sy + uy * b)], fill=GREEN_L, width=max(1, int(1.7 * U)))
                    pos += 9 * U
                d.ellipse([sx - 2.6 * U, sy - 2.6 * U, sx + 2.6 * U, sy + 2.6 * U], fill=GREEN_L)


def card_progress(i, t):
    p = PANELS[i]; t0, t1 = p['t0'], p['t1']; last = (i == len(PANELS) - 1); first = (i == 0)
    pr = dict(slide=1.0 if first else sp(t, t0 + 0.02, 'default'), out=0.0 if last else sp(t, t1 - 0.4, 'snappy'))
    if first:
        pr.update(kick=1.0, head=1.0, sub=1.0, chips=[1.0] * len(p['card']['chips']), cursor=1.0, conn=1.0 - sp(t, t1 - 0.4, 'default'))
    else:
        pr.update(kick=sp(t, t0 + 0.12, 'pop'), head=sp(t, t0 + 0.18, 'default'), sub=sp(t, t0 + 0.32, 'default'),
                  chips=[sp(t, t0 + 0.34 + 0.12 * j, 'pop') for j in range(len(p['card']['chips']))], cursor=sp(t, t0 + 0.55, 'default'),
                  conn=sp(t, t0 + 0.45, 'default') - (0.0 if last else sp(t, t1 - 0.4, 'default')))
    return pr


# =====================================================================================================
# SCENE
# =====================================================================================================
def active_panel(t):
    for i, p in enumerate(PANELS):
        if p['t0'] <= t < p['t1']:
            return i
    return len(PANELS) - 1


class Scene:
    def __init__(self, W, H, SS):
        self.cam = Camera(W, H, SS); self.cards = Cards(self.cam.U)

    def render(self, t):
        t = t / TS                                     # internal (design) time
        rate = abs(rot_at(t + 0.01) - rot_at(t - 0.01)) / 0.02
        if rate > 40:                                   # motion blur only while the camera spins
            n = 3; acc = None
            for j in range(n):
                a = np.asarray(self._render(max(0.0, t + ((j + 0.5) / n - 0.5) * 0.9 / FPS / TS), blur=True), dtype=np.float32)
                acc = a if acc is None else acc + a
            return Image.fromarray((acc / n).astype(np.uint8))
        return self._render(t)

    def _render(self, t, blur=False):
        cam = self.cam; u = cam.U
        zoom = tv(TR['zoom'], t, 5e-4); hx = tv(TR['hx'], t, 0.05); hy = tv(TR['hy'], t, 0.05)
        pitch = tv(TR['pitch'], t, 0.02); sf = tv(TR['sf'], t, 1e-3); rot = rot_at(t)
        dim = tv(TR['dim'], t); dmg = tv(TR['dmg'], t); reb = tv(TR['reb'], t); Y = tv(TR['year'], t, 0.4)
        cr, sr = math.cos(math.radians(rot)), math.sin(math.radians(rot)); k_ = (122 / zoom) * sf
        cam.set(pitch=pitch, rot=rot, target=(hx - k_ * cr, hy + k_ * sr), zoom=zoom)
        hfall = 0.07 + 0.93 * clamp(pitch / 45.0)
        f = Frame(cam, bg=lerpc(PAPER, (198, 192, 184), dim * 0.8))
        f.grid(34, color=lerpc(LINE, (186, 180, 172), dim * 0.6))
        gd = lambda c: tint(c, dim)
        vis = lambda x, y: cam.onscreen(x, y, 0.45)

        e = tv(TR['ov_sectors'], t)
        if e > 0.01:
            for q, (pts, col) in enumerate([([(-272, -320), (400, -320), (400, 380), (-272, 380)], (233, 225, 213)),
                                            ([(-700, -90), (-272, -90), (-272, 95), (-700, 95)], (226, 229, 232)),
                                            ([(-700, -320), (-272, -320), (-272, -90), (-700, -90)], (231, 227, 235)),
                                            ([(-700, 95), (-272, 95), (-272, 380), (-700, 380)], (229, 233, 226))]):
                s_ = clamp(e * 1.15 - 0.1 * q)
                if s_ > 0.01:
                    f.poly([(-272 + (x + 272) * s_, y) for x, y in pts], fill=col)
        for pts, w in RIVERS:
            f.ribbon(pts, w + 7, gd(BANK)); f.ribbon(pts, w, gd(RIVER))
        pu = smooth(1437, 1443, Y)
        if pu > 0.01:
            f.ribbon([(-60, 0), (-60 + (-202) * pu, 0)], 11 if Y >= 1701 else 6, gd(ROAD))
        if Y >= 1936:
            f.ribbon([(-262, 0), (-262 - 308 * smooth(1930, 1936, Y), 10)], 12, gd(ROAD))
        # death strip / Wall line
        wp = max(tv(TR['wall'], t, 1e-3), tv(TR['scar'], t, 1e-3)); dm = demol(t); dfrac = clamp(dm / (_MAXD + 12))
        if wp > 0.01:
            n_ = max(2, int(len(WALL_D) * min(1.0, wp)))
            f.ribbon(WALL_D[:n_], 16, (236, 232, 220))
            if dfrac < 0.97:
                f.ribbon(WALL_D[:max(2, int(n_ * min(1.0, tv(TR['wall'], t))))], 3.6 * (1 - dfrac), (40, 40, 40))
        if tv(TR['ov_tempelhof'], t) > 0.05:
            f.ribbon([(-190, 330), (-190 + 160 * clamp(tv(TR['ov_tempelhof'], t)), 330)], 9, gd(ROAD))

        # houses
        for h in HOUSES:
            pr = smooth(h['year'] - 6, h['year'], Y)
            if pr < 0.003:
                continue
            cx_, cy_ = (h['b'][0] + h['b'][2]) / 2, (h['b'][1] + h['b'][3]) / 2
            if not vis(cx_, cy_):
                continue
            kind = h['kind']; col = WALLS[h['c']]; rc = ROOFC; roof = h['roof']; rh = h['rh']
            if kind == 'glass': col = GLASS
            elif kind == 'slab': col = SLAB
            elif kind == 'weimar': col = (240, 238, 232)
            hf = hfall * pr
            if dmg > 0 and kind != 'glass':
                d1 = smooth(h['r'] * 0.55, h['r'] * 0.55 + 0.12, dmg); d2 = smooth(h['r'] * 1.05, h['r'] * 1.05 + 0.12, dmg)
                hf *= lerp(lerp(1.0, 0.5, d1), 0.10, d2)
                col = lerpc(lerpc(col, (150, 144, 136), d1), (170, 164, 156), d2); rc = lerpc(ROOFC, col, max(d1, d2)); rh = rh * (1 - max(d1, d2))
                if reb > 0:
                    rb = smooth(h['r2'], h['r2'] + 0.1, reb) * d1
                    if rb > 0:
                        col = lerpc(col, NEWC, rb); rc = lerpc(rc, NEWC, rb); hf = lerp(hf, hfall * 0.85 * pr, rb); rh = rh * (1 - rb)
                        roof = 'flat' if rb > 0.5 else roof
            f.box(*h['b'], h['h'], tint(col, dim), roof, rh=rh, ridge=h['ridge'], rc=tint(rc, dim), hf=hf, pr=min(1.0, pr * 1.0))
        tp = smooth(1694, 1700, Y)
        if tp > 0.01:
            for (x, y, r) in TREES:
                if vis(x, y):
                    f.tree(x, y, r, 0, tp * (1.0 if dmg < 0.85 else 0.6))

        # landmarks (each one grows when the year passes its date)
        st = tint(STONE, dim)
        g = lambda y0: smooth(y0 - 6, y0, Y)
        q = g(1237)
        if q > 0.01:
            f.box(-52, -10, -44, -2, 12, st, 'gable', rh=4, ridge='x', hf=hfall * q, pr=q)
            f.spire(-48, -6, 3, 12 * hfall * q, 16 * q * (hfall if hfall > 0.5 else 0.3), hf=hfall * q, pr=q)
        q = g(1443)
        if q > 0.01:
            f.box(-18, -32, 30, 10, 18 + 4 * g(1701), tint((236, 226, 208), dim), 'hip', rh=7, ridge='x', rc=tint(ROOFC, dim), hf=hfall * q * (0.5 + 0.5 * g(1701)), pr=q)
        q = g(1791)
        if q > 0.01:
            for yy in (-24, -14, -4, 4, 14, 24):
                f.box(-268, yy - 2, -256, yy + 2, 22, st, 'flat', hf=hfall * q, pr=q)
            f.box(-268, -28, -256, 28, 4, st, 'flat', z0=22 * hfall * q, hf=hfall * q, pr=q)
            q2 = g(1793)
            if q2 > 0.01:
                f.box(-264, -4, -260, 4, 6, tint(COPPER, dim), 'flat', z0=26 * hfall * q, hf=hfall * q2, pr=q2)
        q = g(1884)
        if q > 0.01:
            rd = 1 - smooth(0.45, 0.55, dmg) * (1 - smooth(1992, 1999, Y)) * 0.65
            f.box(-318, -72, -286, -36, 24, st, 'flat', hf=hfall * q * rd, pr=q, outline=True)
            for (cx, cy) in ((-318, -72), (-286, -72), (-318, -36), (-286, -36)):
                f.box(cx - 3, cy - 3, cx + 3, cy + 3, 32, st, 'flat', hf=hfall * q * rd, pr=q)
            q2 = g(1999)
            if q2 > 0.01:
                f.dome(-302, -54, 24 * hfall, 10, hf=hfall * q2, pr=q2, col=GLASS)
        q = g(1936)
        if q > 0.01:
            f.cylinder(STAD[0], STAD[1], 42, 0, 16, col=st, hf=hfall * q, pr=q)
        q = g(1895)
        if q > 0.01:
            kh = lerp(1.0, 0.5, smooth(0.25, 0.35, dmg))
            f.box(KWC[0] - 5, KWC[1] - 5, KWC[0] + 5, KWC[1] + 5, 32, st, 'flat', hf=hfall * kh * q, pr=q, outline=True)
            if kh > 0.9:
                f.spire(KWC[0], KWC[1], 3, 32 * hfall * q, 22 * q, hf=hfall * q, pr=q)
        q = g(1923)
        if q > 0.01:
            f.box(-150, 288, -70, 300, 10, tint((228, 226, 220), dim), 'flat', hf=hfall * q, pr=q)
        q = g(1969)
        if q > 0.01:
            f.cylinder(TV[0], TV[1], 2.4, 0, 80, col=tint((214, 210, 202), dim), hf=hfall * q, pr=q)
            f.dome(TV[0], TV[1], 78 * hfall * q, 8, hf=hfall * q, pr=q, col=tint((196, 196, 198), dim))
            f.spire(TV[0], TV[1], 0.8, 86 * hfall * q, 24 * q, hf=hfall * q, pr=q)
        # the Wall: blocks draw on with the line, then are removed outward from the Gate
        if wp > 0.01 and hfall > 0.3 and dfrac < 0.995:
            n_ = int(len(WALL_D) * min(1.0, tv(TR['wall'], t)))
            for kk, (x, y) in enumerate(WALL_D[:n_]):
                hk = 1.0 - clamp((dm - abs(kk - _kc)) / 8.0)
                if hk > 0.02 and vis(x, y):
                    f.box(x - 2.4, y - 2.4, x + 2.4, y + 2.4, 11, (150, 148, 144), 'flat', hf=hk)
            for kk in range(18, n_, 38):
                x, y = WALL_D[kk]; hk = 1.0 - clamp((dm - abs(kk - _kc)) / 8.0)
                if hk > 0.02:
                    f.cylinder(x + 6, y, 2.2, 0, 15, col=(120, 118, 114), cone=4, cone_col=(70, 68, 66), hf=hk)
        # memorial: flash-forward at 30-32 s and for real from 2005
        sk = max(tv(TR['ov_stelae'], t), smooth(1999, 2005, Y))
        if sk > 0.01:
            for ii, (x, y, hh) in enumerate(STELAE):
                s_ = clamp(sk * 1.4 - (ii % 9) * 0.04 - (ii // 9) * 0.02)
                if s_ > 0.01:
                    f.box(x, y, x + 2.8, y + 3.4, hh * s_, (176, 174, 168), 'flat', hf=1.0)
        # smoke columns
        n = tv(TR['smoke_n'], t)
        for i in range(min(len(SMOKE_SPOTS), int(math.ceil(max(0.0, n))))):
            s_ = clamp(n - i); x, y = SMOKE_SPOTS[i]
            f.cylinder(x, y, 3 + (i % 3), 0, 30 + (i * 7) % 34, col=(158, 152, 146), cone=16, cone_col=(176, 170, 164), hf=s_)
        # crowds
        n = tv(TR['crowd_n'], t); cdx = tv(TR['crowd_dx'], t)
        for i in range(min(len(CROWD), int(math.ceil(max(0.0, n))))):
            s_ = clamp(n - i); x, y = CROWD[i]
            f.person(x + cdx + 5 * math.sin(0.6 * t + i), y + 4 * math.cos(0.5 * t + i * 1.3), 11 if (i % 4 == 0 and dm > 10) else 0, PERSON_COLORS[i % 5], 1.3 * s_, t, i * 0.7)
        f.flush(shadows=(pitch > 12), strength=0.28)
        f.depth_blur(cam.u * 0.8)

        # ---------------- overlays (sharp) ----------------
        er = tv(TR['ov_route'], t)
        if er > 0.02:
            m = max(2, int(clamp(er) * len(ROUTE)))
            f.dotted([(x, y, 0) for (x, y) in ROUTE[:m]], color=ACC, w=2.6, lift=0.0, every=2)
        n = tv(TR['bombers'], t)
        for i in range(int(math.ceil(max(0.0, n)))):
            s_ = clamp(n - i); rb = random.Random(i * 977 + 3)
            x = hx - 220 + ((rb.uniform(0, 320) + 14 * t) % 320); y = hy + rb.uniform(-130, 130)
            plane(f, x, y, 0.0, size=5.5 * s_, col=(40, 40, 40), z=120)
            a, b = cam.PJ(x - 6, y, 112), cam.PJ(x - 6, y, 92)
            f.d.line([a, b], fill=(120, 116, 110), width=max(1, int(1.1 * u)))
        n = tv(TR['blasts'], t)
        for i in range(int(math.ceil(max(0.0, n)))):
            ph = t * 0.9 + i * 0.37; cyc = int(math.floor(ph)); fr = ph - cyc; rb = random.Random(i * 131 + cyc * 7)
            gr = smooth(0, 0.12, fr) * (1 - smooth(0.12, 0.8, fr)) * clamp(n - i)
            if gr > 0.02:
                starburst(f, hx - 60 + rb.uniform(-120, 60), hy + rb.uniform(-110, 110), r=rb.uniform(5, 9) * gr)
        ea = tv(TR['ov_artillery'], t)
        if ea > 0.02:
            for i in range(14):
                rb = random.Random(i * 53 + 9); fx, fy = 300 + rb.uniform(0, 70), hy + rb.uniform(-150, 150)
                ph = (t * 1.5 + i * 0.23) % 1.0; ex, ey = rb.uniform(-120, 60), fy + rb.uniform(-60, 60)
                if ph < 0.15:
                    starburst(f, fx, fy, r=7 * smooth(0, 0.07, ph) * (1 - smooth(0.07, 0.15, ph)) * ea)
                m = max(2, int(ph * 11))
                pts = [(lerp(fx, ex, k / 10), lerp(fy, ey, k / 10), math.sin(math.pi * k / 10) * 70) for k in range(min(11, m + 1))]
                f.dotted(pts, color=ACC, w=1.4, lift=1.0, every=2)
                if ph > 0.9:
                    starburst(f, ex, ey, r=6 * ea)
        ed = tv(TR['ov_advance'], t)
        if ed > 0.02:
            for q_, yy in enumerate((-90, -40, 10, 60, 110)):
                s_ = clamp(ed * 1.2 - 0.08 * q_)
                end = (60 + abs(yy) * 0.3, yy * 0.45)
                if s_ > 0.03:
                    arrow(f, (330, yy), (lerp(330, end[0], s_), lerp(yy, end[1], s_)))
        n = tv(TR['planes_n'], t)
        for k in range(int(math.ceil(max(0.0, n)))):
            s_ = clamp(n - k); u_ = (t * 0.22 + k / max(1.0, n)) % 1.0
            x = lerp(-440, -120, u_); y = lerp(40, 285, u_) + math.sin(k * 2.1) * 14
            plane(f, x, y, math.atan2(245, 320), size=7 * s_, col=(40, 40, 40), z=40 + 30 * (1 - u_))
        eb = tv(TR['ov_blockade'], t)
        if eb > 0.02:
            for q_, (x, y) in enumerate([(-300, 30), (-300, 380), (-300, -250), (330, 20), (60, 320), (60, -300), (-420, -60), (-430, 150)]):
                s_ = clamp(sp(t, 54.0 + 0.18 * q_, 'pop')) * clamp(eb * 1.2)
                if s_ > 0.03:
                    c_ = cam.PJ(x, y, 0); r_ = 8 * u * s_
                    f.d.line([(c_[0] - r_, c_[1] - r_), (c_[0] + r_, c_[1] + r_)], fill=ACC, width=max(1, int(2.6 * u)))
                    f.d.line([(c_[0] - r_, c_[1] + r_), (c_[0] + r_, c_[1] - r_)], fill=ACC, width=max(1, int(2.6 * u)))
            m = int(25 * clamp(eb))
            if m > 1:
                f.dotted([(-290 + 330 * math.cos(a / 24 * 2 * math.pi), 60 + 270 * math.sin(a / 24 * 2 * math.pi), 0) for a in range(m)], color=ACC, w=2.0, lift=0.0, every=2)
        if e > 0.02:
            for (txt, x, y) in [('SOVIET', 40, 0), ('BRITISH', -470, 0), ('FRENCH', -470, -200), ('AMERICAN', -470, 230)]:
                c_ = cam.PJ(x, y, 0); fnt = font(9.5 * u * clamp(e), 'mono'); tw = f.d.textlength(txt, font=fnt)
                if e > 0.5:
                    f.d.rounded_rectangle([c_[0] - tw / 2 - 8 * u, c_[1] - 9 * u, c_[0] + tw / 2 + 8 * u, c_[1] + 9 * u], radius=9 * u, fill=(20, 20, 20))
                    f.d.text(c_, txt, font=fnt, fill=(250, 250, 250), anchor='mm')
            m = int(len(densify([(-272, -320), (-272, 380)], 8)) * clamp(e))
            if m > 1:
                f.dotted([(x, y, 0) for x, y in densify([(-272, -320), (-272, 380)], 8)][:m], color=ACC, w=2.2, lift=0.0, every=2)
        ef = tv(TR['ov_flag'], t)
        if ef > 0.02:
            a, b = cam.PJ(-302, -54, 24 * hfall), cam.PJ(-302, -54, (24 + 24 * clamp(ef)) * hfall)
            f.d.line([a, b], fill=INK, width=max(1, int(1.6 * u)))
            if ef > 0.4:
                f.d.polygon([b, (b[0] + 16 * u * clamp(ef), b[1] + 4 * u), (b[0], b[1] + 9 * u)], fill=ACC)
        eds = tv(TR['ov_dots'], t)
        if eds > 0.02:
            for q_, (x, y) in enumerate([(118, -46), (126, -38), (134, -50), (142, -42), (150, -34)]):
                s_ = sp(t, 24.0 + 0.5 * q_, 'pop') * clamp(eds * 1.2); dark = sp(t, 24.5 + 0.5 * q_, 'default')
                if s_ > 0.03:
                    c_ = cam.PJ(x, y, 0); r_ = 4 * u * s_
                    f.d.ellipse([c_[0] - r_, c_[1] - r_, c_[0] + r_, c_[1] + r_], fill=lerpc((255, 255, 255), INK, dark), outline=ACC, width=max(1, int(1.4 * u)))
        # hero ring + crosshair
        rs = tv(TR['rs'], t, 1e-3)
        rx, ry, rz = tv(TR['rx'], t, 0.05), tv(TR['ry'], t, 0.05), tv(TR['rz'], t, 0.05)
        anchor = None
        if rs > 0.03:
            hxp, hyp = cam.PJ(rx, ry, rz); anchor = (hxp, hyp); r = 9 * u * rs
            f.d.ellipse([hxp - r, hyp - r, hxp + r, hyp + r], outline=INK, width=max(1, int(1.8 * u / 1.5)))
            arm = 24 * u * rs
            f.d.line([(hxp - arm, hyp), (hxp + arm, hyp)], fill=ACC, width=max(1, int(1.6 * u / 1.5)))
            f.d.line([(hxp, hyp - arm), (hxp, hyp + arm)], fill=ACC, width=max(1, int(1.6 * u / 1.5)))
        # tags wipe in after the card lands and leave before the next one
        if not blur:
            for i, p in enumerate(PANELS):
                if not (p['t0'] - 0.1 <= t < p['t1'] + 0.05) or p['P'].get('spin') or SPIN0 <= p['t0'] < SPIN1:
                    continue
                up = 1.0 if i == 0 else sp(t, p['t0'] + 0.35, 'snappy'); dn = 0.0 if i == len(PANELS) - 1 else sp(t, p['t1'] - 0.3, 'snappy')
                fl = clamp(up - dn)
                for (txt, pt, off) in p['P'].get('tags', []):
                    if fl < 0.02:
                        continue
                    ax_, ay_ = cam.PJ(*pt); fnt = font(10 * u, 'mono'); tw = f.d.textlength(txt, font=fnt) / u
                    px = ax_ / u + abs(off[0]) if p['P']['side'] == 'L' else ax_ / u - abs(off[0]) - tw - 10
                    px = max(24, min(720 - 24 - tw - 9, px)); py = max(24, min(405 - 24, ay_ / u + off[1]))
                    f.tag(txt, pt, (px - ax_ / u, py - ay_ / u), fl)
        # card
        ci = active_panel(t)
        if not blur or True:
            self.cards.draw(f.img, ci, PANELS[ci]['P']['side'], card_progress(ci, t), anchor)
        return f.img


def make_scene(W, H, SS):
    return Scene(W, H, SS)


# =====================================================================================================
# AUDIO: original score + sound effects, synthesized in code on the same bar grid (120 BPM, bar = 2 s)
# =====================================================================================================
def _synthx():
    import synth as sy

    class SynthX(sy.Synth):
        def siren(self, dur=2.4, lo=420.0, hi=820.0, **_):
            x = self.t(dur); f = lo + (hi - lo) * (0.5 - 0.5 * np.cos(2 * np.pi * x / (dur * 0.9)))
            env = np.minimum(1, x / 0.4) * np.minimum(1, (dur - x) / 0.5)
            ph = 2 * np.pi * np.cumsum(f) / self.sr
            return (np.sin(ph) + 0.35 * np.sin(2 * ph)) * env * 0.6

        def whistle(self, dur=0.85, hi=2600.0, lo=700.0, **_):
            x = self.t(dur); f = hi + (lo - hi) * (x / dur) ** 0.8
            ph = 2 * np.pi * np.cumsum(f) / self.sr
            env = np.minimum(1, x / 0.1) * np.minimum(1, (dur - x) / 0.05)
            return (np.sin(ph) * 0.5 + 0.12 * self.rng.standard_normal(len(x))) * env * 0.5

        def blast(self, dur=1.0, **_):
            x = self.t(dur); z = self.rng.standard_normal(len(x))
            lp = np.zeros_like(z); s = 0.0
            for i in range(len(z)):
                s += 0.12 * (z[i] - s); lp[i] = s
            return (lp * 6.0 * np.exp(-x * 5.0) + np.sin(2 * np.pi * (42 + 70 * np.exp(-x * 9)) * x) * np.exp(-x * 4.5)) * np.minimum(1, x / 0.004)

        def boom(self, dur=0.45, f0=38.0, **_):
            x = self.t(dur); z = self.rng.standard_normal(len(x))
            return (np.sin(2 * np.pi * (f0 + 60 * np.exp(-x * 25)) * x) * np.exp(-x * 8) + 0.25 * np.diff(z, prepend=0) * np.exp(-x * 40)) * np.minimum(1, x / 0.002)

        def flak(self, dur=0.14, **_):
            x = self.t(dur)
            return np.diff(self.rng.standard_normal(len(x)), prepend=0) * np.exp(-x * 45) * 0.8

        def clack(self, dur=0.25, **_):
            x = self.t(dur); y = np.zeros(len(x))
            for off in (0.0, 0.09):
                i0 = int(off * self.sr); xx = x[:len(x) - i0]
                y[i0:] += (np.sin(2 * np.pi * 900 * xx) + 0.6 * self.rng.standard_normal(len(xx))) * np.exp(-xx * 90)
            return y * 0.7

        def hammer(self, dur=0.12, **_):
            x = self.t(dur)
            return (np.sin(2 * np.pi * 620 * x) * 0.6 + 0.8 * self.rng.standard_normal(len(x))) * np.exp(-x * 70)

        def creak(self, dur=0.9, **_):
            x = self.t(dur); f = 180 - 90 * (x / dur); ph = 2 * np.pi * np.cumsum(f) / self.sr
            saw = 2 * ((ph / (2 * np.pi)) % 1) - 1
            return saw * (0.5 + 0.5 * np.sin(2 * np.pi * 14 * x)) * np.minimum(1, x / 0.1) * np.minimum(1, (dur - x) / 0.2) * 0.35

        def drone(self, dur=4.0, freq=55.0, **_):
            x = self.t(dur); env = np.minimum(1, x / 0.8) * np.minimum(1, (dur - x) / 1.0)
            return (np.sin(2 * np.pi * freq * x) + 0.8 * np.sin(2 * np.pi * freq * 1.4142 * x) + 0.3 * np.sin(2 * np.pi * freq * 2 * x)) * env * 0.5
    return sy, SynthX


def audio(path):
    sy, SynthX = _synthx(); sy.Synth = SynthX
    dur = DUR
    roots = [38] * 4 + [36, 29, 31, 36] + [33, 29] + [33] * 17 + [33] * 11 + [36, 31] + [36, 31, 33, 29, 36]
    chords = [[62, 65, 69]] * 4 + [[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]] + [[69, 72, 76], [65, 69, 72]] + [[57, 60, 64]] * 28 + \
             [[60, 64, 67], [67, 71, 74]] + [[72, 76, 79], [67, 71, 74], [69, 72, 76], [65, 69, 72], [72, 76, 79]]
    roots = roots[:45]; chords = chords[:45]
    layers = [
        dict(type='arp', bars=[0, 3], step_beats=2, pattern=[0, 2, 1, 2], gain=0.30),
        dict(type='kick', **{'from': 8.0, 'to': 19.5}, every_beats=1, gain=0.85),
        dict(type='bass', bars=[4, 9], beats=[0, 2], gain=0.5), dict(type='arp', bars=[5, 9], gain=0.22, octave_every=4, octave_from_bar=8),
        dict(type='hat', bars=[6, 9], gain=0.25),
        dict(type='kick', **{'from': 32.0, 'to': 43.5}, every_beats=2, gain=0.45),
        dict(type='bass', bars=[27, 37], beats=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], gain=0.30),
        dict(type='kick', **{'from': 56.0, 'to': 61.5}, every_beats=2, gain=0.40),
        dict(type='kick', **{'from': 76.0, 'to': 87.5}, every_beats=1, gain=0.85),
        dict(type='bass', bars=[38, 43], beats=[0, 2], gain=0.5), dict(type='arp', bars=[38, 43], to=87.5, gain=0.22, octave_every=4, octave_from_bar=38),
        dict(type='hat', bars=[40, 43], to=87.5, gain=0.25)]
    cues = []

    def c(t, kind, gain=0.5, **kw):
        for k_ in ('dur', 'peak'):
            if k_ in kw and kind in ('whoosh', 'riser', 'siren'):
                kw[k_] = kw[k_] * TS
        d = dict(t=float(t) * TS, kind=kind, gain=gain); d.update(kw); cues.append(d)
    # 0-8 medieval: bells on the dates, soft ticks, riser into the tilt
    c(0.0, 'thump', 0.8)
    for t_, fr in zip((0, 2, 4, 6), (294, 349, 440, 587)): c(t_, 'chime', 0.30, freq=fr)
    for t_ in (2.0, 2.5, 3.0, 3.5): c(t_, 'tick', 0.25)
    c(2.6, 'click', 0.2)
    for t_, fr in zip((4.0, 4.5, 5.0, 5.5), (1500, 1700, 1900, 2100)): c(t_, 'tick', 0.3, freq=fr)
    c(4.0, 'pop', 0.4); c(6.0, 'thump', 0.6); c(7.0, 'riser', 0.5, dur=1.0)
    # 8-24 Prussia / Empire / 1900-33
    c(8.0, 'thump', 1.0); c(8.0, 'whoosh', 0.7, dur=1.6, peak=1.2)
    for k in range(6): c(10.0 + 0.25 * k, 'tick', 0.3)
    c(11.5, 'pop', 0.5)
    for k in range(6): c(12.0 + 0.5 * k, 'tick', 0.25, freq=1500 + 80 * k)
    c(14.0, 'pop', 0.5)
    c(16.0, 'thump', 0.9); c(16.0, 'whoosh', 0.6, dur=1.6, peak=1.0)
    for k, fr in enumerate((1500, 1700, 1900, 2100, 2400)): c(16.5 + 0.25 * k, 'tick', 0.3, freq=fr)
    c(20.0, 'thump', 0.8, f0=40.0, decay=5.0)
    for fr in (523, 659, 784, 1047): c(22.0, 'pluck', 0.35, freq=fr)
    # 24-54 WWII
    for k in range(5): c(24.0 + 0.5 * k, 'tick', 0.25, freq=3400 - 250 * k)
    c(26.0, 'siren', 0.35, dur=2.4); c(27.0, 'boom', 0.3); c(27.5, 'boom', 0.3)
    for k in range(7): c(28.0 + 0.55 * k, 'clack', 0.25)
    c(28.0, 'drone', 0.25, dur=3.6, freq=55.0); c(30.0, 'chime', 0.5, freq=220)
    for T_ in (32.0, 33.0, 33.5, 34.5, 35.5): c(T_ - 0.85, 'whistle', 0.35); c(T_, 'blast', 0.55)
    for T_ in (32.25, 33.25, 34.0, 35.0): c(T_, 'flak', 0.4)
    c(34.0, 'siren', 0.2, dur=2.0)
    c(36.0, 'riser', 0.6, dur=2.0)
    for T_ in (36.0, 36.5, 37.0, 37.5): c(T_ - 0.85, 'whistle', 0.3); c(T_, 'blast', 0.6)
    for k in range(16): c(38.0 + 0.25 * k, 'boom', 0.5)
    for k in range(8): c(42.0 + 0.5 * k if k < 4 else 40.0 + 0.5 * (k - 4), 'boom', 0.35)
    for k in range(16): c(40.0 + 0.25 * k + 0.125, 'click', 0.12)
    c(42.0, 'boom', 0.6, f0=34.0); c(43.0, 'boom', 0.6, f0=34.0); c(43.5, 'thump', 0.7, f0=36.0)
    c(45.0, 'tick', 0.4)
    for k in range(8): c(46.0 + k, 'pluck', 0.22, freq=146.8 if k % 2 == 0 else 220.0)
    for k, f0 in enumerate((45, 50, 56, 63)): c(48.0 + 0.5 * k, 'thump', 0.5, f0=float(f0))
    c(50.0, 'chime', 0.4, freq=330); c(52.0, 'whoosh', 0.4, dur=1.0, peak=0.9); c(53.0, 'tick', 0.3)
    # 54-80 Cold War
    for k in range(4): c(54.0 + 0.5 * k, 'thump', 0.8, f0=40.0)
    for k in range(10): c(56.5 + 0.5 * k, 'whoosh', 0.07, dur=0.8, peak=0.35)
    for k, fr in enumerate((1500, 1800, 2100, 2500)): c(58.3 + 0.2 * k, 'tick', 0.3, freq=fr)
    c(60.0, 'chime', 0.5, freq=523); c(60.5, 'thump', 0.5); c(61.0, 'thump', 0.5, f0=60.0)
    for k in range(6): c(62.5 + 0.25 * k, 'tick', 0.22, freq=700)
    for k in range(4): c(64.0 + 0.5 * k, 'hammer', 0.5)
    c(64.0, 'thump', 1.0, f0=38.0); c(66.0, 'thump', 0.9, f0=40.0); c(67.0, 'creak', 0.15)
    for k in range(7): c(68.0 + 0.25 * k, 'tick', 0.28, freq=1200 + 170 * k)
    c(69.5, 'pop', 0.7); c(70.0, 'pluck', 0.35, freq=220); c(71.0, 'pluck', 0.35, freq=330)
    c(73.0, 'riser', 0.7, dur=3.0); c(74.0, 'whoosh', 0.6, dur=4.0, peak=3.9)
    c(76.0, 'chime', 0.6, freq=523); c(76.0, 'thump', 0.9)
    for k in range(6): c(76.5 + 0.25 * k, 'hammer', 0.22)
    c(78.0, 'chime', 0.7, freq=659); c(78.0, 'thump', 1.0)
    # 80-90 Today
    c(80.0, 'pop', 0.6); c(82.0, 'pop', 0.6)
    for k in range(4): c(82.5 + 0.25 * k, 'tick', 0.28, freq=1500 + 200 * k)
    for k in range(6): c(84.0 + 0.25 * k, 'tick', 0.22)
    c(84.0, 'chime', 0.4, freq=220); c(86.0, 'whoosh', 0.6, dur=1.6, peak=1.0); c(89.5, 'tick', 0.4)
    for L_ in layers:
        for k_ in ('from', 'to'):
            if k_ in L_:
                L_[k_] = L_[k_] * TS
    spec = dict(duration=dur, bpm=120 / TS, seed=4242, mix=dict(music=0.55, sfx=0.75, drive=1.1, peak=0.9),
                music=dict(roots=roots, chords=chords, layers=layers), cues=cues,
                ambient=[dict(type='hum', **{'from': 56.0 * TS, 'to': 62.0 * TS}, gain=0.07, f0=90.0, lfo_period=3.0)])
    music, sfx, mx, sr = sy.build(spec)
    S = SynthX(sr, 7); n = len(sfx)

    def bed(t0, t1, gain, lo, shape=1.0):
        t0, t1 = t0 * TS, t1 * TS
        m = int((t1 - t0) * sr); z = S.rng.standard_normal(m); y = np.zeros(m); s = 0.0
        for i in range(m):
            s += lo * (z[i] - s); y[i] = s
        x = np.arange(m) / sr; env = np.minimum(1, x / 0.8) * np.minimum(1, ((t1 - t0) - x) / 0.9)
        i0 = int(t0 * sr); sfx[i0:i0 + m] += (y * 6.0 * env * gain * shape)[:n - i0]

    def drone(t0, t1, gain):
        t0, t1 = t0 * TS, t1 * TS
        i0 = int(t0 * sr); sig = S.drone(dur=t1 - t0, freq=55.0); sfx[i0:i0 + len(sig)] += sig[:n - i0] * gain
    bed(0.0, 8.0, 0.035, 0.02); bed(46.0, 54.0, 0.035, 0.02)
    bed(62.0, 64.6, 0.05, 0.25); bed(72.0, 76.0, 0.06, 0.25); bed(75.0, 78.0, 0.06, 0.3)
    drone(20.0, 28.0, 0.07); drone(32.0, 44.0, 0.07)
    mix = sy.finish(music, sfx, mx)
    sy.write_wav(path, mix, sr)
    tail = mix[-int(0.3 * sr):]
    print('audio: wrote %s  (rms of last 0.3 s: %.4f)' % (path, float(np.sqrt(np.mean(tail ** 2)))))


# =====================================================================================================
# CLI:   python berlin_film.py                    full film (all cores), audio included
#        python berlin_film.py --still 75         one PNG        --sheet / --check-loop as in anim3d
#        python berlin_film.py --segment 0 20     frames 0-20 s -> segment file (for single-core machines); concat later
# =====================================================================================================
def render_segment(a, b, res, fps, ss, workers, out):
    import subprocess
    from multiprocessing import Pool, cpu_count
    import anim3d
    W, H = anim3d.dims(res); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'seg_%05.1f_%05.1f.mp4' % (a, b))
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H), '-r', str(fps), '-i', '-',
           '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p', path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    i0, i1 = int(round(a * fps)), int(round(b * fps)); t0 = time.time()
    with Pool(workers or max(1, cpu_count() - 1), initializer=anim3d._init, initargs=(make_scene, W, H, ss, None, 1, 0.5, fps)) as pool:
        for k, buf in enumerate(pool.imap(anim3d._frame, range(i0, i1), chunksize=2)):
            proc.stdin.write(buf)
            if k % 15 == 0:
                sys.stdout.write('\r  %d/%d  %ds ' % (k + 1, i1 - i0, time.time() - t0)); sys.stdout.flush()
    proc.stdin.close(); proc.wait(); print('\n  ->', path)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--segment', nargs=2, type=float); ap.add_argument('--audio-only', metavar='WAV')
    ap.add_argument('--res', type=int, default=720); ap.add_argument('--ss', type=int, default=2); ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--out', default='.')
    known, rest = ap.parse_known_args()
    if known.audio_only:
        audio(known.audio_only)
    elif known.segment:
        render_segment(known.segment[0], known.segment[1], known.res, FPS, known.ss, known.workers, known.out)
    else:
        run(make_scene, duration=DUR, name='berlin', fps=FPS, audio_fn=audio, default_res=720, argv=None)
