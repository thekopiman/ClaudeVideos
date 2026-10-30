#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SALZBURG  -  a 24 s drone film: top-view map of the city, then a tilt into 3D and a flight from the
Hauptbahnhof through the Old Town to Festung Hohensalzburg.  Same art style as small_hill.py.
Pure Python: every frame is drawn with Pillow, the score and SFX are synthesized with numpy,
ffmpeg only encodes and muxes.

    pip install numpy pillow        (and install ffmpeg: brew install ffmpeg / apt install ffmpeg)

    python salzburg.py                         # 1280x720, 30 fps  -> salzburg_16x9.mp4
    python salzburg.py --res 1080 --fps 60 --mblur 3      # final quality (slow)
    python salzburg.py --still 20.5            # one PNG of t = 20.5 s (fast way to check a look)
    python salzburg.py --sheet                 # contact sheet, one frame per beat
    python salzburg.py --res 360 --fps 15      # quick preview

Rules of the house (same as small_hill.py):
  * the film is a pure function of time: render(t) paints frame t, nothing is carried between frames
  * springs only: closed-form damped springs; a value with several targets = sum of one spring per change
  * seeded randomness only; no timers, no easing curves
  * 120 BPM = 0.5 s per beat, 2 s per bar, 12 bars; picture and sound share the same times
  * the last frame equals the first, so it loops
  * the map is stylized and not to scale; it is drawn heading-up (north arrow points down)
"""
import argparse
import bisect
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

DUR = 24.0
BPM = 120
BEAT = 60.0 / BPM

PAPER = (243, 239, 230); INK = (26, 26, 26); INK2 = (120, 116, 108); ACC = (255, 92, 31); LINE = (205, 199, 186)
GROUND = (236, 228, 212); ROAD = (219, 211, 195); LOT = (246, 239, 226); RIVER = (178, 194, 196); BANK = (228, 226, 212)
SHADOW = (128, 116, 98); COPPER = (146, 176, 150); STONE = (216, 205, 188); ROOFC = (226, 204, 184); GARDEN = (226, 232, 212)
HAZE = (242, 236, 224)
WALLS = [(244, 232, 212), (240, 222, 184), (238, 212, 198), (222, 226, 204), (232, 228, 220), (246, 236, 200)]
CAR_COLORS = [(58, 58, 58), (246, 244, 238), (168, 163, 154), (206, 186, 150), (90, 90, 92)]
PCOL = [(58, 58, 58), (120, 116, 108), (88, 84, 76), (150, 112, 84), (70, 76, 86)]

# ----------------------------------------------------------------------------------------------
# springs: closed form, pure functions of time
# ----------------------------------------------------------------------------------------------
PRESETS = {
    "snappy": (24.0, 0.80),   # labels, pins, presence
    "default": (13.0, 0.90),  # containers, growth
    "heavy": (7.0, 1.00),     # the 3D tilt
    "pop": (22.0, 0.62),      # visible overshoot: pins, badges, callout card
    "glide": (6.0, 1.00),     # camera moves and the drone's flight (overlapping legs = smooth path)
    "return": (11.0, 1.00),    # everything settling back to frame 0
}


def spring(t, t0, preset="default"):
    x = t - t0
    if x <= 0.0:
        return 0.0
    w, z = PRESETS[preset]
    if z < 1.0:
        wd = w * math.sqrt(1.0 - z * z)
        return 1.0 - math.exp(-z * w * x) * (math.cos(wd * x) + (z * w / wd) * math.sin(wd * x))
    return 1.0 - math.exp(-w * x) * (1.0 + w * x)


def track(t, keys, preset="default"):
    return sum(d * spring(t, t0, preset) for t0, d in keys)


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def lerpc(c1, c2, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))


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



# ----------------------------------------------------------------------------------------------
# world data (stylized Salzburg, not to scale; real orientation: x east, y south)
# ----------------------------------------------------------------------------------------------
# ------------------------------------------------------------------ world data (real orientation: x east, y south)
def catmull(pts,n=10):
    out=[]
    P=[pts[0]]+pts+[pts[-1]]
    for i in range(1,len(P)-2):
        p0,p1,p2,p3=P[i-1],P[i],P[i+1],P[i+2]
        for j in range(n):
            t=j/n
            out.append(tuple(0.5*((2*p1[k])+(-p0[k]+p2[k])*t+(2*p0[k]-5*p1[k]+4*p2[k]-p3[k])*t*t+(-p0[k]+3*p1[k]-3*p2[k]+p3[k])*t**3) for k in (0,1)))
    out.append(pts[-1]); return out
RIVER_C=catmull([(70,235),(52,160),(30,100),(22,42),(26,-8),(12,-56),(-8,-104),(-40,-160),(-78,-230)])
def river_edges(w):
    L,R=[],[]
    for i,p in enumerate(RIVER_C):
        a=RIVER_C[max(0,i-1)]; b=RIVER_C[min(len(RIVER_C)-1,i+1)]
        dx,dy=b[0]-a[0],b[1]-a[1]; l=math.hypot(dx,dy) or 1; nx,ny=-dy/l,dx/l
        L.append((p[0]+nx*w/2,p[1]+ny*w/2)); R.append((p[0]-nx*w/2,p[1]-ny*w/2))
    return L,R
def river_dist(x,y): return min(math.hypot(x-p[0],y-p[1]) for p in RIVER_C)

import numpy as np
def smooth(a,b,x):
    t=max(0.0,min(1.0,(x-a)/(b-a))); return t*t*(3-2*t)
HILLS=[dict(name="Festungsberg",cx=-45,cy=92,rx=66,ry=60,rot=0,A=88,pl=0.46,asym=1.30,seed=1.3,trees=40),
       dict(name="Mönchsberg",cx=-168,cy=62,rx=96,ry=34,rot=-8,A=64,pl=0.50,asym=1.35,seed=2.1,trees=42),
       dict(name="Kapuzinerberg",cx=158,cy=10,rx=58,ry=52,rot=0,A=60,pl=0.30,asym=1.0,seed=0.7,trees=34)]
def hill_h(h,x,y):
    ca,sa=math.cos(math.radians(-h['rot'])),math.sin(math.radians(-h['rot']))
    dx,dy=x-h['cx'],y-h['cy']; u=(dx*ca-dy*sa)/h['rx']; v=(dx*sa+dy*ca)/h['ry']
    if v<0: v*=h['asym']                      # steeper, cliff-like side facing the Old Town
    r=math.hypot(u,v)
    f=1-smooth(h['pl'],1.0,r)
    n=0.07*math.sin(4.3*u+h['seed'])*math.sin(3.7*v+1.7*h['seed'])+0.04*math.sin(8.1*u+2.0*v)
    f=max(0.0,min(1.0,f**0.9+n*f*(1-f)*4))
    return h['A']*f
def hill_base(h,n=60):
    ca,sa=math.cos(math.radians(h['rot'])),math.sin(math.radians(h['rot']))
    return [(h['cx']+math.cos(i/n*6.2832)*h['rx']*ca-math.sin(i/n*6.2832)*h['ry']*sa, h['cy']+math.cos(i/n*6.2832)*h['rx']*sa+math.sin(i/n*6.2832)*h['ry']*ca) for i in range(n)]
CELL=3
for _h in HILLS:
    half=max(_h['rx'],_h['ry'])*1.1
    xs=np.arange(_h['cx']-half,_h['cx']+half+CELL,CELL); ys=np.arange(_h['cy']-half,_h['cy']+half+CELL,CELL)
    Z=np.array([[hill_h(_h,x,y) for x in xs] for y in ys])
    gy,gx=np.gradient(Z,CELL,CELL)
    levels=[_h['A']*k/9 for k in range(1,9)]
    cl=[]
    for j in range(len(ys)-1):
        for i in range(len(xs)-1):
            z4=(float(Z[j,i]),float(Z[j,i+1]),float(Z[j+1,i+1]),float(Z[j+1,i]))
            if max(z4)<0.15: continue
            x0,x1,y0,y1=float(xs[i]),float(xs[i+1]),float(ys[j]),float(ys[j+1])
            P=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
            segs=[]
            for li,L_ in enumerate(levels):
                pts=[]
                for e in range(4):
                    za,zb=z4[e],z4[(e+1)%4]
                    if (za-L_)*(zb-L_)<0:
                        t=(L_-za)/(zb-za); pa,pb=P[e],P[(e+1)%4]
                        pts.append((pa[0]+(pb[0]-pa[0])*t,pa[1]+(pb[1]-pa[1])*t))
                if len(pts)==2: segs.append((li,pts[0],pts[1],L_))
                elif len(pts)==4: segs+= [(li,pts[0],pts[1],L_),(li,pts[2],pts[3],L_)]
            cl.append((x0,y0,x1,y1,z4,float((gx[j,i]+gx[j+1,i+1])/2),float((gy[j,i]+gy[j+1,i+1])/2),(x0+x1)/2,(y0+y1)/2,sum(z4)/4/_h['A'],segs))
    _h['cells']=cl
    _tr=random.Random(int(_h['seed']*10)); _h['tree_list']=[]
    tries=0
    while len(_h['tree_list'])<_h['trees'] and tries<4000:
        tries+=1
        a_=_tr.uniform(0,6.2832); rr_=math.sqrt(_tr.uniform(0,1))
        ca,sa=math.cos(math.radians(_h['rot'])),math.sin(math.radians(_h['rot']))
        ux,vy=math.cos(a_)*rr_*_h['rx'],math.sin(a_)*rr_*_h['ry']
        x=_h['cx']+ux*ca-vy*sa; y=_h['cy']+ux*sa+vy*ca
        z=hill_h(_h,x,y); f=z/_h['A']
        if not (0.10<f<0.93): continue
        _h['tree_list'].append((x,y,_tr.uniform(3.0,4.4),z))
def in_hill(x,y,pad=1.12):
    for h in HILLS:
        ca,sa=math.cos(math.radians(-h['rot'])),math.sin(math.radians(-h['rot']))
        dx,dy=x-h['cx'],y-h['cy']; u=dx*ca-dy*sa; v=dx*sa+dy*ca
        if (u/(h['rx']*pad))**2+(v/(h['ry']*pad))**2<1: return True
    return False
def hill_top(h): return h['A']

WP=[(182,-146,10),(135,-122,52),(68,-92,58),(10,-62,54),(-38,-28,56),(-10,14,62),(-44,50,66),(-45,88,165)]
WPNAMES=["Hauptbahnhof","Mirabell Palace","Makartsteg","Getreidegasse","Residenzplatz / Dom","St Peter's","Hohensalzburg"]
WPIDX=[0,2,3,4,5,6,7]

# landmark footprints (x0,y0,x1,y1) used to keep houses out
FOOT=[(-24,16,0,66),(-62,20,-34,36),(-66,48,-30,66),(-52,-10,-28,14),(-36,-20,40,-2),(12,0,38,30),
      (20,-134,116,-52),(126,-160,262,-110),(-100,-64,-30,-38)]
PLAZAS=[(-30,0,10,16)]

rr=random.Random(21)
altstadt=[]
for x0 in range(-96,4,24):
    for y0 in range(-80,58,28):
        cx,cy=x0+12+rr.uniform(-2,2),y0+14+rr.uniform(-2,2)
        if in_hill(cx,cy,1.0) or river_dist(cx,cy)<24: continue
        if any(a-4<=cx<=c+4 and b-4<=cy<=d+4 for a,b,c,d in FOOT+PLAZAS): continue
        if rr.random()<0.12: continue
        n=1 if rr.random()<0.55 else 2
        for i in range(n):
            w=rr.uniform(11,18); dp=rr.uniform(16,24)
            bx=cx+(0 if n==1 else (-6 if i==0 else 6)); by=cy+rr.uniform(-2,2)
            h=rr.uniform(22,36); ridge=rr.choice("xy")
            altstadt.append(dict(b=(bx-w/2,by-dp/2,bx+w/2,by+dp/2),h=h,rh=rr.uniform(7,11),ridge=ridge,col=rr.randrange(len(WALLS)),kind="gable",era="M"))
# Getreidegasse: two rows of tall narrow houses
for side,y in ((-1,-57),(1,-47)):
    for i in range(6):
        x=-92+i*10
        h=rr.uniform(28,36)
        altstadt.append(dict(b=(x-5,y-9,x+5,y+9) if True else None,h=h,rh=rr.uniform(7,10),ridge="y",col=(i*2+(side>0))%len(WALLS),kind="gable",era="M"))
neustadt=[]
for x0 in range(38,260,34):
    for y0 in range(-124,72,30):
        cx,cy=x0+17,y0+15
        if in_hill(cx,cy,1.05) or river_dist(cx,cy)<30: continue
        if any(a-6<=cx<=c+6 and b-6<=cy<=d+6 for a,b,c,d in FOOT): continue
        if cx<60 and -140<cy<-45: continue
        if rr.random()<0.2: continue
        w=rr.uniform(20,28); dp=rr.uniform(18,24)
        neustadt.append(dict(b=(cx-w/2,cy-dp/2,cx+w/2,cy+dp/2),h=rr.uniform(18,30),rh=rr.uniform(5,9),ridge="x" if w>dp else "y",col=rr.randrange(len(WALLS)),kind=rr.choice(["hip","hip","flat"]),era="B"))
outer=[]
for x0 in range(-420,420,40):
    for y0 in range(-300,260,36):
        cx,cy=x0+20,y0+18
        if in_hill(cx,cy,1.1) or river_dist(cx,cy)<34: continue
        if (-110<cx<270 and -175<cy<80): continue
        if 195<cy<280 and abs(cx-174)>0 and river_dist(cx,cy)<40: continue
        if rr.random()<0.25: continue
        w=rr.uniform(22,32); dp=rr.uniform(20,28)
        outer.append(dict(b=(cx-w/2,cy-dp/2,cx+w/2,cy+dp/2),h=rr.uniform(14,26),rh=rr.uniform(5,8),ridge="x" if w>dp else "y",col=rr.randrange(len(WALLS)),kind=rr.choice(["hip","gable","flat"]),era="B"))
TREES=[]
while len(TREES)<70:
    x,y=rr.uniform(-110,260),rr.uniform(-170,100)
    if in_hill(x,y,1.0) or river_dist(x,y)<16: continue
    if any(a-2<=x<=c+2 and b-2<=y<=d+2 for a,b,c,d in FOOT+PLAZAS): continue
    if (20<x<116 and -134<y<-52): continue
    if rr.random()<0.5 and river_dist(x,y)>60 and not (20<x<120): continue
    TREES.append((x,y,rr.uniform(4,6)))
CARS=[]
for (x,y,dx) in [(70,-40,1),(110,-26,-1),(150,-72,1),(190,-26,-1),(230,-58,1),(86,10,1),(176,24,-1)]:
    CARS.append((x,y,dx,rr.choice([(58,58,58),(246,244,238),(168,163,154),(206,186,150)])))
PEOPLE=[(rr.uniform(-70,-10),rr.choice((-35,-26,-17)),rr.randrange(5)) for _ in range(16)]+[(rr.uniform(-26,6),rr.uniform(3,14),rr.randrange(5)) for _ in range(8)]
PCOL=[(58,58,58),(120,116,108),(88,84,76),(150,112,84),(70,76,86)]


# ----------------------------------------------------------------------------------------------
# the scene
# ----------------------------------------------------------------------------------------------
# camera stops (heading-up): (time, cam target, zoom, rotation)  -- every move is one spring per change
CAM0 = ((30.0, -20.0), 1.2, 180.0)
CAM_STOPS = [(12.0, (10.0, -30.0), 1.2, 180.0), (16.0, (120.0, -100.0), 1.5, 190.0), (18.0, (-30.0, -40.0), 1.6, 172.0),
             (20.0, (-18.0, 12.0), 1.75, 180.0), (22.0, (-45.0, 175.0), 1.4, 200.0)]
T_RETURN = 23.0

# callout tags: (text, world anchor, panel-unit offset, t_in, t_out)
TAGS = [("Gothic spire", (-60, 29, 62), (-70, 70), 14.9, 15.9), ("onion dome", (-37, 57, 52), (60, 80), 15.1, 15.95),
        ("twin towers", (-12, 19, 60), (-110, 50), 15.3, 16.0),
        ("Hauptbahnhof", (182, -148, 20), (-20, 70), 16.3, 17.2),
        ("Mirabell · 1606 · Baroque 1721–27", (68, -122, 40), (-150, -75), 17.0, 18.0),
        ("Makartsteg", (10, -62, 3), (60, 60), 18.5, 19.4), ("Getreidegasse · medieval houses", (-64, -52, 36), (-215, -70), 19.0, 20.0),
        ("Residenz · Renaissance → Baroque", (-6, -10, 26), (-10, 90), 20.4, 21.5), ("Cathedral · Baroque · 1628", (-12, 40, 70), (-215, -5), 20.6, 21.7),
        ("Franciscan · Romanesque + Gothic", (-60, 29, 60), (-30, -85), 21.0, 21.9), ("St Peter's · 696", (-37, 57, 56), (20, -40), 21.2, 22.0)]
BADGES = [("M", -45, 86), ("M", -64, -52), ("M", -37, 57), ("R", -8, -10), ("R", 26, 14), ("B", -12, 40), ("B", 68, -122), ("B", -40, 2)]
LABELS2D = [("SALZACH", 24, -30, 3.0, 11.6, 9), ("ALTSTADT", -52, -6, 3.3, 11.6, 9), ("NEUSTADT", 120, -20, 3.6, 11.6, 9),
            ("542 m", -45, 92, 5.0, 11.6, 11), ("Festungsberg", -45, 128, 5.2, 11.6, 8), ("Mönchsberg", -168, 64, 5.5, 11.6, 8),
            ("Kapuzinerberg", 158, 40, 5.8, 11.6, 8)]
HBF = (182, -148)

# landmark primitives: (group, kind, params).  Heights are multiplied by the group's growth factor.
LM = [
    (0, "box", (-23, 22, -1, 66, 30, STONE, "gable", 10, "y", (206, 200, 188))), (0, "box", (-23, 17, -15, 25, 48, STONE, "flat", 0, "x", ROOFC)),
    (0, "box", (-9, 17, -1, 25, 48, STONE, "flat", 0, "x", ROOFC)), (0, "onion", (-19, 21, 48, 5.5, 13)), (0, "onion", (-5, 21, 48, 5.5, 13)),
    (0, "dome", (-12, 46, 30, 10)),
    (1, "box", (-60, 24, -36, 34, 22, STONE, "gable", 9, "x", ROOFC)), (1, "box", (-64, 25, -56, 33, 40, STONE, "flat", 0, "x", ROOFC)),
    (1, "spire", (-60, 29, 5, 40, 38)),
    (2, "box", (-64, 50, -38, 62, 22, STONE, "gable", 8, "x", ROOFC)), (2, "box", (-42, 52, -32, 62, 36, STONE, "flat", 0, "x", ROOFC)),
    (2, "onion", (-37, 57, 36, 6.5, 16)),
    (3, "box", (-50, -8, -30, 12, 30, STONE, "flat", 0, "x", ROOFC)), (3, "dome", (-40, 2, 30, 9)),
    (3, "box", (-50, -12, -44, -6, 40, STONE, "flat", 0, "x", ROOFC)), (3, "box", (-36, -12, -30, -6, 40, STONE, "flat", 0, "x", ROOFC)),
    (4, "box", (-34, -18, 22, -2, 26, WALLS[1], "hip", 8, "x", ROOFC)), (4, "box", (14, 2, 36, 28, 30, WALLS[0], "hip", 8, "y", ROOFC)),
    (4, "box", (30, 22, 38, 30, 50, STONE, "flat", 0, "x", ROOFC)), (4, "onion", (34, 26, 50, 4.5, 10)),
    (5, "box", (26, -132, 110, -112, 24, WALLS[0], "hip", 8, "x", ROOFC)), (5, "box", (20, -134, 36, -110, 28, WALLS[0], "hip", 8, "y", ROOFC)),
    (5, "box", (100, -134, 116, -110, 28, WALLS[0], "hip", 8, "y", ROOFC)),
    (8, "box", (150, -158, 214, -140, 18, (236, 232, 224), "flat", 0, "x", ROOFC)), (8, "box", (130, -136, 262, -112, 5, (226, 222, 214), "flat", 0, "x", ROOFC)),
    (6, "box", (-12, -64, 32, -60, 2, STONE, "flat", 0, "x", ROOFC)), (6, "box", (-2, -27, 46, -13, 2.5, STONE, "flat", 0, "x", ROOFC)),
]
FW = (232, 226, 214)
FORT = [("box", (-80, 76, -10, 80, 11, FW, "flat", 0, "x", ROOFC)), ("box", (-80, 98, -10, 102, 11, FW, "flat", 0, "x", ROOFC)),
        ("box", (-80, 76, -76, 102, 11, FW, "flat", 0, "x", ROOFC)), ("box", (-14, 76, -10, 102, 11, FW, "flat", 0, "x", ROOFC)),
        ("box", (-66, 82, -38, 96, 38, (238, 232, 220), "hip", 13, "x", (200, 190, 176))),
        ("box", (-34, 86, -16, 96, 26, (236, 230, 216), "gable", 9, "x", (200, 190, 176))),
        ("cyl", (-78, 78, 5, 22, FW, 11)), ("cyl", (-12, 78, 5, 22, FW, 11)), ("cyl", (-78, 100, 5, 22, FW, 11)),
        ("cyl", (-12, 100, 5, 22, FW, 11)), ("cyl", (-68, 84, 4.8, 32, (238, 232, 220), 13))]
LOWC = (238, 232, 214); HIGHC = (224, 216, 186); ROCKC = (234, 226, 212)


def _hull(pts):
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


class Scene:
    def __init__(self, W, H, SS):
        self.W, self.H, self.SS = W, H, SS
        self.K = W / 720.0
        for h_ in altstadt + neustadt + outer:
            x0, y0, x1, y1 = h_["b"]
            h_["d"] = clamp(river_dist((x0 + x1) / 2, (y0 + y1) / 2) / 260.0)
        self.rl = [0.0]
        for i in range(1, len(RIVER_C)):
            self.rl.append(self.rl[-1] + math.hypot(RIVER_C[i][0] - RIVER_C[i - 1][0], RIVER_C[i][1] - RIVER_C[i - 1][1]))
        self.RL = self.rl[-1]
        lr = random.Random(77)
        self.cars = []
        for i in range(14):
            dr = 1 if i % 2 == 0 else -1
            v = lr.uniform(28, 44)
            s0 = lr.uniform(20, self.RL - v * 8 - 20) if dr == 1 else lr.uniform(v * 8 + 20, self.RL - 20)
            self.cars.append((s0, v, dr, lr.choice(CAR_COLORS), 15.4 + lr.uniform(0, 0.9), 22.4 + lr.uniform(0, 0.4)))
        self.people = []
        for i in range(22):
            dr = lr.choice((-1, 1)); v = lr.uniform(5, 9)
            s0 = lr.uniform(20, self.RL - 80) if dr == 1 else lr.uniform(80, self.RL - 20)
            self.people.append(("w", s0, v, dr, lr.choice((-1, 1)), lr.choice(PCOL), lr.uniform(0, 6.28), 15.6 + lr.uniform(0, 1.2), 22.0 + lr.uniform(0, 0.4)))
        for i in range(9):
            self.people.append(("p", -10 + lr.uniform(-18, 18), 8 + lr.uniform(-7, 7), lr.choice(PCOL), lr.uniform(0, 6.28), 16.0 + lr.uniform(0, 1.2), 22.0 + lr.uniform(0, 0.4)))
        for i in range(7):
            self.people.append(("p", 68 + lr.uniform(-34, 34), -83 + lr.uniform(-22, 22), lr.choice(PCOL), lr.uniform(0, 6.28), 16.0 + lr.uniform(0, 1.2), 22.0 + lr.uniform(0, 0.4)))
        self.hikers = [(8, PCOL[0]), (38, PCOL[3]), (70, PCOL[1])]
        w, h = W * SS, int(H * SS * 0.36)
        col = np.linspace(1.0, 0.0, h)[:, None] * np.ones((1, w))
        self.g_blur = Image.fromarray((col * 255).astype(np.uint8), "L")
        self.g_haze = Image.fromarray((col * 70).astype(np.uint8), "L")

    def river_at(self, s):
        s = max(0.0, min(self.RL, s))
        i = min(bisect.bisect_right(self.rl, s) - 1, len(RIVER_C) - 2)
        t_ = (s - self.rl[i]) / (self.rl[i + 1] - self.rl[i] + 1e-9)
        a, b = RIVER_C[i], RIVER_C[i + 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        l = math.hypot(dx, dy) or 1.0
        return a[0] + dx * t_, a[1] + dy * t_, dx / l, dy / l

    @staticmethod
    def drone_pos(t):
        x, y, z = WP[0]
        for i in range(7):
            sp = spring(t, 16.0 + i * 1.0, "glide")
            x += (WP[i + 1][0] - WP[i][0]) * sp
            y += (WP[i + 1][1] - WP[i][1]) * sp
            z += (WP[i + 1][2] - WP[i][2]) * sp
        w = spring(t, 22.0, "default") * (1 - spring(t, 23.0, "default"))
        a = (t - 22.0) * 2.2
        x += w * 50 * math.cos(a)
        y += w * 50 * math.sin(a)
        z += 60 * spring(t, 23.0, "default")
        return x, y, z

    @staticmethod
    def wipe(img, x, y, w, h, f, fn):
        if f <= 0.01 or w < 2 or h < 2:
            return
        layer = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
        fn(ImageDraw.Draw(layer))
        cw = int(w) if f > 0.995 else max(1, int(w * min(f, 1.0)))
        c = layer.crop((0, 0, cw, int(h)))
        img.paste(c, (int(x), int(y)), c)

    # ------------------------------------------------------------------------------------------
    def render(self, t):
        W, H, SS, K = self.W, self.H, self.SS, self.K
        U = K * SS

        # ---------------- camera: one spring per change ----------------
        pitch = track(t, [(12.0, 55)], "heavy") + track(t, [(20.0, -5), (22.0, 2)], "glide") + track(t, [(T_RETURN, -52)], "return")
        if abs(pitch) < 0.02:
            pitch = 0.0
        u = clamp(pitch / 55.0)
        kx, ky, kz, kr = [], [], [], []
        px_, zm_, rt_ = CAM0[0], CAM0[1], CAM0[2]
        prev = (px_[0], px_[1], zm_, rt_)
        for (tt, c, z, r) in CAM_STOPS:
            kx.append((tt, c[0] - prev[0])); ky.append((tt, c[1] - prev[1])); kz.append((tt, z - prev[2])); kr.append((tt, r - prev[3]))
            prev = (c[0], c[1], z, r)
        lastc = prev
        camx = CAM0[0][0] + track(t, kx, "glide") + track(t, [(T_RETURN, CAM0[0][0] - lastc[0])], "return")
        camy = CAM0[0][1] + track(t, ky, "glide") + track(t, [(T_RETURN, CAM0[0][1] - lastc[1])], "return")
        zoomp = CAM0[1] + track(t, kz, "glide") + track(t, [(T_RETURN, CAM0[1] - lastc[2])], "return")
        rotd = CAM0[2] + track(t, kr, "glide") + track(t, [(T_RETURN, CAM0[2] - lastc[3])], "return")
        if abs(rotd - 180.0) < 0.02:
            rotd = 180.0
        if abs(camx - CAM0[0][0]) < 0.05 and abs(camy - CAM0[0][1]) < 0.05 and abs(zoomp - CAM0[1]) < 5e-4:
            camx, camy, zoomp = CAM0[0][0], CAM0[0][1], CAM0[1]     # settle exactly: last frame == frame 0
        th = math.radians(pitch)
        ct, st = math.cos(th), math.sin(th)
        r_ = math.radians(rotd)
        cr_, sr_ = math.cos(r_), math.sin(r_)
        zoom = zoomp * K
        cy0 = H / 2 + K * (4 + 20 * u)

        def PJ(x, y, z=0.0):
            x -= camx
            y -= camy
            xr = x * cr_ - y * sr_
            yr = x * sr_ + y * cr_
            pf = zoom * max(0.05, 1 + yr * st * 0.0011)
            return ((W / 2 + xr * pf) * SS, (cy0 + (yr * ct - z * st) * pf) * SS)

        def depth(x, y):
            x -= camx
            y -= camy
            return x * sr_ + y * cr_

        def rotn(nx, ny):
            return (nx * cr_ - ny * sr_, nx * sr_ + ny * cr_)

        def vis(n):
            nxr, nyr = rotn(n[0], n[1])
            return nyr * st + n[2] * ct > 0.002

        def tone(col, n):
            nxr, nyr = rotn(n[0], n[1])
            lit = nxr * -0.6 + nyr * -0.8
            return shade(col, min(1.06, 0.99 + 0.13 * lit + 0.05 * n[2]))

        def onscreen(x, y, m=0.25):
            sx, sy = PJ(x, y, 0)
            return -m * W * SS < sx < (1 + m) * W * SS and -m * H * SS < sy < (1 + 0.4) * H * SS + 200 * U

        # ---------------- animated state ----------------
        g = clamp(track(t, [(12.4, 1), (23.0, -1)]))
        g = 0.0 if g < 1e-3 else g
        hh = max(0.0, track(t, [(12.0, 1), (22.9, -1)], "default"))
        hh = 0.0 if hh < 1e-3 else hh
        hp = [max(0.0, track(t, [(4.0 + 0.7 * i, 1), (23.0 + 0.15 * i, -1)], "default")) for i in range(3)]
        hp = [0.0 if v < 1e-3 else v for v in hp]
        rdp = clamp(track(t, [(7.6, 1), (23.0, -1)], "snappy"))
        rp = clamp(track(t, [(2.0, 0.34), (2.4, 0.33), (2.8, 0.33), (22.8, -0.34), (22.95, -0.33), (23.1, -0.33)]))
        rp = 0.0 if rp < 1e-3 else rp
        gp = clamp(track(t, [(7.0, 1), (22.9, -1)], "default"))
        gp = 0.0 if gp < 1e-3 else gp
        bgc = lerpc(PAPER, GROUND, g)
        img = Image.new("RGB", (W * SS, H * SS), bgc)
        d = ImageDraw.Draw(img)

        gw = max(1, int(SS * max(1.0, K / 2.3)))
        gcol = lerpc(LINE, (226, 218, 200), g)
        for i in range(-24, 25):
            d.line([PJ(i * 34, -700), PJ(i * 34, 700)], fill=gcol, width=gw)
            d.line([PJ(-800, i * 34), PJ(800, i * 34)], fill=gcol, width=gw)

        # ---------------- ground: roads, lots, plazas ----------------
        def pres_of(dd):
            return clamp(track(t, [(6.0 + dd * 1.4, 1), (23.1 + (1 - dd) * 0.3, -1)], "snappy"), 0.0, 1.05)

        def grow_of(dd):
            return max(0.0, track(t, [(14.0 + dd * 1.6, 1), (22.9 + (1 - dd) * 0.3, -1)], "snappy"))

        if rdp > 0.01:
            for (a0, b0, a1, b1) in ((-100, -90, 8, 66), (34, -128, 266, 74)):
                cx_, cy_ = (a0 + a1) / 2, (b0 + b1) / 2
                hx, hy = (a1 - a0) / 2 * rdp, (b1 - b0) / 2 * rdp
                d.polygon([PJ(cx_ - hx, cy_ - hy), PJ(cx_ + hx, cy_ - hy), PJ(cx_ + hx, cy_ + hy), PJ(cx_ - hx, cy_ + hy)], fill=ROAD)
            for h_ in altstadt + neustadt + outer:
                p_ = pres_of(h_["d"])
                if p_ < 0.02:
                    continue
                x0, y0, x1, y1 = h_["b"]
                m = 3 if h_ in altstadt else 5
                cx_, cy_ = (x0 + x1) / 2, (y0 + y1) / 2
                hx, hy = ((x1 - x0) / 2 + m) * p_, ((y1 - y0) / 2 + m) * p_
                if onscreen(cx_, cy_):
                    d.polygon([PJ(cx_ - hx, cy_ - hy), PJ(cx_ + hx, cy_ - hy), PJ(cx_ + hx, cy_ + hy), PJ(cx_ - hx, cy_ + hy)], fill=LOT)
            for (a, b, c, e) in PLAZAS:
                cx_, cy_ = (a + c) / 2, (b + e) / 2
                hx, hy = (c - a) / 2 * rdp, (e - b) / 2 * rdp
                d.polygon([PJ(cx_ - hx, cy_ - hy), PJ(cx_ + hx, cy_ - hy), PJ(cx_ + hx, cy_ + hy), PJ(cx_ - hx, cy_ + hy)], fill=(250, 246, 238))

        # river (draws on from the south, banks first)
        if rp > 0:
            n = max(2, int(rp * len(RIVER_C)))
            L1, R1 = river_edges(36)
            L2, R2 = river_edges(28)
            d.polygon([PJ(*p) for p in L1[:n] + R1[:n][::-1]], fill=BANK)
            d.polygon([PJ(*p) for p in L2[:n] + R2[:n][::-1]], fill=RIVER)

        # station tracks (always: frame 0)
        for k in range(5):
            d.line([PJ(120, -132 + k * 5), PJ(300, -132 + k * 5)], fill=INK2, width=max(1, SS))

        # Mirabell parterre
        if gp > 0.01:
            cx_, cy_ = 68, -83
            sc_ = gp

            def gq(a, b, c, e):
                return [PJ(cx_ + (a - cx_) * sc_, cy_ + (b - cy_) * sc_), PJ(cx_ + (c - cx_) * sc_, cy_ + (b - cy_) * sc_),
                        PJ(cx_ + (c - cx_) * sc_, cy_ + (e - cy_) * sc_), PJ(cx_ + (a - cx_) * sc_, cy_ + (e - cy_) * sc_)]
            d.polygon(gq(28, -110, 108, -56), fill=GARDEN)
            for (a, b, c, e) in [(32, -106, 66, -84), (70, -106, 104, -84), (32, -80, 66, -60), (70, -80, 104, -60)]:
                q = gq(a, b, c, e)
                d.polygon(q, outline=INK2, width=SS)
                d.line([PJ(cx_ + (a - cx_) * sc_, cy_ + ((b + e) / 2 - cy_) * sc_), PJ(cx_ + (c - cx_) * sc_, cy_ + ((b + e) / 2 - cy_) * sc_)], fill=(190, 200, 176), width=SS)
                d.line([PJ(cx_ + ((a + c) / 2 - cx_) * sc_, cy_ + (b - cy_) * sc_), PJ(cx_ + ((a + c) / 2 - cx_) * sc_, cy_ + (e - cy_) * sc_)], fill=(190, 200, 176), width=SS)
            d.polygon([PJ(cx_ + 7 * sc_ * math.cos(i / 20 * 6.28), cy_ + 7 * sc_ * math.sin(i / 20 * 6.28)) for i in range(20)], fill=(214, 226, 226), outline=INK2)

        # ---------------- landmark / house growth tables ----------------
        GT = []
        for gi in range(9):
            pr = 1.0 if gi == 8 else clamp(track(t, [(6.2 + 0.08 * gi, 1), (23.1 + 0.04 * gi, -1)], "snappy"), 0.0, 1.05)
            lg = max(0.0, track(t, [(14.4 + 0.1 * gi, 1), (22.9 + 0.05 * gi, -1)], "snappy"))
            GT.append((pr, 1.7 * (0.06 * pr + 0.94 * lg)))
        ztop = hill_top(HILLS[0]) * hh
        # hill heights seen by items (fortress sits on the plateau)
        items = []
        FACES = [((0, 1, 0), 0), ((0, -1, 0), 1), ((1, 0, 0), 2), ((-1, 0, 0), 3)]

        def add(dep, fn, shp=None):
            items.append((dep, fn, shp))

        def box_item(x0, y0, x1, y1, h, col, roof, rh, ridge, rc, z0, hf, pr, outline=False):
            if pr < 0.02:
                return
            cx_, cy_ = (x0 + x1) / 2, (y0 + y1) / 2
            if not onscreen(cx_, cy_, 0.35):
                return
            hx, hy = (x1 - x0) / 2 * pr, (y1 - y0) / 2 * pr
            x0, x1, y0, y1 = cx_ - hx, cx_ + hx, cy_ - hy, cy_ + hy
            Hh, RH = h * hf, rh * hf

            def draw():
                faces = [((0, 1, 0), [(x0, y1), (x1, y1)]), ((0, -1, 0), [(x1, y0), (x0, y0)]),
                         ((1, 0, 0), [(x1, y1), (x1, y0)]), ((-1, 0, 0), [(x0, y0), (x0, y1)])]
                for n, (a, b) in faces:
                    if vis(n) and Hh > 0.3:
                        d.polygon([PJ(*a, z0), PJ(*b, z0), PJ(*b, z0 + Hh), PJ(*a, z0 + Hh)], fill=tone(col, n))
                xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
                zt = z0 + Hh
                if roof == "flat" or RH < 0.3:
                    d.polygon([PJ(x0, y0, zt), PJ(x1, y0, zt), PJ(x1, y1, zt), PJ(x0, y1, zt)], fill=tone(STONE if roof == "flat" else rc, (0, 0, 1)))
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
                        if vis(n):
                            d.polygon([PJ(*q) for q in p], fill=tone(col, n))
                    for n, p in sl:
                        l = math.sqrt(sum(c * c for c in n))
                        n = tuple(c / l for c in n)
                        if vis(n):
                            d.polygon([PJ(*q) for q in p], fill=tone(rc, n))
                else:
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
                        l = math.sqrt(sum(c * c for c in n)) or 1
                        n = tuple(c / l for c in n)
                        if vis(n):
                            d.polygon([PJ(*q) for q in p], fill=tone(rc, n))
                if outline and u < 0.08:
                    d.line([PJ(x0, y0, zt), PJ(x1, y0, zt), PJ(x1, y1, zt), PJ(x0, y1, zt), PJ(x0, y0, zt)], fill=INK, width=max(1, int(1.3 * U)))
            add(depth(cx_, cy_), draw, (x0, y0, x1, y1, Hh + RH * 0.6, z0))

        def slices(cx, cy, z0, Hh, rfun, col, n=12):
            if Hh < 0.2:
                return

            def draw():
                for i in range(n):
                    zf = (i + 0.5) / n
                    R = rfun(zf)
                    if R <= 0.2:
                        continue
                    z = z0 + Hh * zf
                    d.polygon([PJ(cx + R * math.cos(a / 16 * 6.2832), cy + R * math.sin(a / 16 * 6.2832), z) for a in range(16)], fill=shade(col, 0.94 + 0.14 * zf))
            add(depth(cx, cy) + 0.01, draw, (cx - 6, cy - 6, cx + 6, cy + 6, Hh, z0))

        def lm_prim(kind, p, z_on_hill, hf, pr):
            if kind == "box":
                x0, y0, x1, y1, h, col, roof, rh, ridge, rc = p
                box_item(x0, y0, x1, y1, h, col, roof, rh, ridge, rc, z_on_hill, hf, pr, outline=True)
            elif pr > 0.02 and onscreen(p[0], p[1], 0.35):
                if kind == "onion":
                    cx, cy, zb, R, Hh = p
                    slices(cx, cy, zb * hf + z_on_hill, Hh * hf, lambda zf: R * pr * (0.55 + 0.75 * math.sin(math.pi * zf * 0.82)) * (1 - zf ** 2.4) * (1 if zf < 0.97 else 0.1), COPPER, 12)
                elif kind == "dome":
                    cx, cy, zb, R = p
                    slices(cx, cy, zb * hf + z_on_hill, R * 0.9 * hf, lambda zf: R * pr * math.sqrt(max(0, 1 - zf ** 2)), COPPER, 10)
                elif kind == "spire":
                    cx, cy, hw, zb, Hh = p
                    slices(cx, cy, zb * hf + z_on_hill, Hh * hf, lambda zf: hw * 1.15 * pr * (1 - zf), STONE, 10)
                elif kind == "cyl":
                    cx, cy, R, Hh, col, cone = p
                    slices(cx, cy, z_on_hill, Hh * hf, lambda zf: R * pr, col, 12)
                    slices(cx, cy, z_on_hill + Hh * hf, cone * hf, lambda zf: R * pr * 1.12 * (1 - zf), ROOFC, 12)

        for (gi, kind, p) in LM:
            lm_prim(kind, p, 0.0, GT[gi][1], GT[gi][0])
        for (kind, p) in FORT:
            lm_prim(kind, p, ztop, GT[7][1], GT[7][0])

        # ---------------- houses ----------------
        for h_ in altstadt + neustadt + outer:
            dd = h_["d"]
            pr = pres_of(dd)
            if pr < 0.02:
                continue
            hf = 0.06 * pr + 0.94 * grow_of(dd)
            x0, y0, x1, y1 = h_["b"]
            box_item(x0, y0, x1, y1, h_["h"], WALLS[h_["col"]], h_["kind"], h_["rh"], h_["ridge"], ROOFC, 0.0, hf, pr)

        # ---------------- trees ----------------
        def tree_item(x, y, r, z, ts):
            if ts < 0.03 or not onscreen(x, y, 0.3):
                return

            def draw():
                d.line([PJ(x, y, z), PJ(x, y, z + r * 1.1 * ts)], fill=(120, 110, 90), width=SS)
                px, py = PJ(x, y, z + r * 2.0 * ts)
                pf = r * zoom * SS * ts
                d.ellipse([px - pf, py - pf * 0.95, px + pf, py + pf * 0.95], fill=(150, 165, 122), outline=(128, 144, 102))
            add(depth(x, y), draw, (x - 3, y - 3, x + 3, y + 3, r * 1.6 * ts + 3 * ts, z))
        for k, (x, y, r) in enumerate(TREES):
            tree_item(x, y, r, 0.0, max(0.0, track(t, [(14.8 + (k % 11) * 0.1, 1), (22.9 + (k % 5) * 0.05, -1)], "snappy")))

        # ---------------- cars (on the river embankments) and people ----------------
        def obox(cx, cy, Lc, Wc, ang, hgt, col):
            ca, sa = math.cos(ang), math.sin(ang)
            P = [(cx + ca * a - sa * b, cy + sa * a + ca * b) for a, b in ((Lc / 2, Wc / 2), (Lc / 2, -Wc / 2), (-Lc / 2, -Wc / 2), (-Lc / 2, Wc / 2))]

            def draw():
                for i in range(4):
                    a, b = P[i], P[(i + 1) % 4]
                    ex, ey = b[0] - a[0], b[1] - a[1]
                    l = math.hypot(ex, ey) or 1
                    n = (ey / l, -ex / l, 0)
                    if vis(n):
                        d.polygon([PJ(*a, 0), PJ(*b, 0), PJ(*b, hgt), PJ(*a, hgt)], fill=tone(col, n))
                d.polygon([PJ(p[0], p[1], hgt) for p in P], fill=shade(col, 1.12))
            add(depth(cx, cy), draw, None)
        for (s0, v, dr, col, tin, tout) in self.cars:
            cs_ = max(0.0, track(t, [(tin, 1), (tout, -1)], "snappy"))
            if cs_ < 0.03:
                continue
            x, y, tx, ty = self.river_at(s0 + dr * v * (t - 17.0))
            nx_, ny_ = -ty, tx
            off = 21.0 * dr
            cx, cy = x + nx_ * off, y + ny_ * off
            if onscreen(cx, cy):
                ang = math.atan2(ty * dr, tx * dr)
                sx_, sy_ = PJ(cx - tx * dr * 20, cy - ty * dr * 20, 0.2)
                ex_, ey_ = PJ(cx - tx * dr * 8, cy - ty * dr * 8, 0.2)
                add(depth(cx, cy) - 0.01, (lambda a=(sx_, sy_), b=(ex_, ey_): d.line([a, b], fill=(204, 195, 178), width=max(1, int(1.6 * U)))), None)
                obox(cx, cy, 17.0, 7.0, ang, 6.0 * cs_, col)

        def person(x, y, z0, col, ps, ph):
            if ps < 0.03 or not onscreen(x, y):
                return

            def draw():
                zz = z0 + 0.5 * abs(math.sin(2 * math.pi * 1.9 * t + ph))
                d.line([PJ(x, y, zz), PJ(x, y, zz + 9.5 * ps)], fill=col, width=max(2, int(3.6 * zoom * SS)))
                hx_, hy_ = PJ(x, y, zz + 11.8 * ps)
                rr_ = 2.7 * zoom * SS * ps
                d.ellipse([hx_ - rr_, hy_ - rr_, hx_ + rr_, hy_ + rr_], fill=(222, 186, 160))
            add(depth(x, y) + 0.2, draw, None)
        for p in self.people:
            if p[0] == "w":
                _, s0, v, dr, side, col, ph, tin, tout = p
                ps = max(0.0, track(t, [(tin, 1), (tout, -1)], "snappy"))
                x, y, tx, ty = self.river_at(s0 + dr * v * (t - 17.0))
                person(x - ty * 16.5 * side, y + tx * 16.5 * side, 0.0, col, ps, ph)
            else:
                _, bx, by, col, ph, tin, tout = p
                ps = max(0.0, track(t, [(tin, 1), (tout, -1)], "snappy"))
                person(bx + 3.5 * math.sin(1.7 * t + ph), by + 3.0 * math.cos(1.3 * t + ph * 1.7), 0.0, col, ps, ph)

        # ---------------- hills: real terrain ----------------
        def hill_item(hi, hd):
            hpv = hp[hi]
            if hpv < 0.02:
                return
            hs = hpv

            def draw():
                ordered = sorted(hd["cells"], key=lambda c: depth(c[7], c[8]))
                for (x0, y0, x1, y1, z4, gxx, gyy, cx_, cy_, f, segs) in ordered:
                    fe = f * min(hpv, 1.0)
                    if fe < 0.004 or not onscreen(cx_, cy_, 0.4):
                        continue
                    gx_, gy_ = gxx * hs, gyy * hs
                    nrm = math.sqrt(gx_ * gx_ + gy_ * gy_ + 1)
                    nxr, nyr = rotn(-gx_ / nrm, -gy_ / nrm)
                    lit = nxr * -0.55 + nyr * -0.7 + 0.45 / nrm
                    shd = max(0.74, min(1.08, 0.86 + 0.30 * lit))
                    base = tuple(lerp(a, b, fe) for a, b in zip(LOWC, HIGHC))
                    rock = smooth(0.85, 1.5, math.hypot(gx_, gy_)) * 0.85
                    base = tuple(lerp(a, b, rock) for a, b in zip(base, ROCKC))
                    col = shade(tuple(int(c) for c in base), shd)
                    bl_ = smooth(0.0, 0.35, fe)
                    col = tuple(int(lerp(a, b, bl_)) for a, b in zip(bgc, col))
                    poly = [PJ(x0, y0, z4[0] * hh), PJ(x1, y0, z4[1] * hh), PJ(x1, y1, z4[2] * hh), PJ(x0, y1, z4[3] * hh)]
                    d.polygon(poly, fill=col, outline=col)
                    for (li, pa, pb, L_) in segs:
                        if (li + 1) / 9.0 * 0.95 > hpv:
                            continue
                        idx = (li + 1) % 3 == 0
                        if u > 0.3 and not idx:
                            continue
                        c_ = (INK2 if idx else (150, 142, 126)) if u <= 0.3 else (150, 140, 122)
                        d.line([PJ(pa[0], pa[1], L_ * hh), PJ(pb[0], pb[1], L_ * hh)], fill=c_, width=max(1, int((1.3 if idx else 0.8) * U / 1.3)))
            add(depth(hd["cx"], hd["cy"]) - 0.5, draw, None)
            for k, (x, y, r, z) in enumerate(hd["tree_list"]):
                ts = max(0.0, track(t, [(5.2 + 0.7 * hi + (k % 9) * 0.05, 1), (22.9, -1)], "snappy"))
                tree_item(x, y, r, z * hh, ts)
        for hi, hd in enumerate(HILLS):
            hill_item(hi, hd)

        # ---------------- shadows (one mask, blended once) ----------------
        if hh > 0.3 or any(GT[i][1] > 0.4 for i in range(9)):
            sm = Image.new("L", img.size, 0)
            sd = ImageDraw.Draw(sm)
            for dep, fn, shp in items:
                if shp:
                    x0, y0, x1, y1, hgt, z0 = shp
                    if hgt < 2:
                        continue
                    ps = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                    ps2 = ps + [(x - hgt * 0.55, y - hgt * 0.45) for x, y in ps]
                    hp_ = _hull([(round(a, 2), round(b, 2)) for a, b in ps2])
                    if len(hp_) > 2:
                        sd.polygon([PJ(x, y) for x, y in hp_], fill=255)
            if hh > 0.3:
                for hi, hd in enumerate(HILLS):
                    if hp[hi] < 0.5:
                        continue
                    pts = hill_base(hd)
                    tz = hill_top(hd) * hh
                    ps2 = pts + [(x - tz * 0.55, y - tz * 0.45) for x, y in pts]
                    hp_ = _hull([(round(a, 2), round(b, 2)) for a, b in ps2])
                    if len(hp_) > 2:
                        sd.polygon([PJ(x, y) for x, y in hp_], fill=255)
            img = Image.composite(Image.new("RGB", img.size, SHADOW), img, sm.point(lambda v: int(v * 0.30)))
            d = ImageDraw.Draw(img)
        items.sort(key=lambda it: it[0])
        # items were created with the old draw handle; rebind it (closures read `d` from this scope)
        for dep, fn, shp in items:
            fn()

        # ---------------- depth blur + haze toward the far edge ----------------
        ub = u * (1.0 - clamp(spring(t, 21.8, "glide")))           # no depth blur on the fortress hero shot
        if ub > 0.02:
            h36 = self.g_blur.size[1]
            region = img.crop((0, 0, img.size[0], h36))
            blurred = region.filter(ImageFilter.GaussianBlur(2.2 * U * ub))
            region = Image.composite(blurred, region, self.g_blur.point(lambda v: int(v * ub)))
            region = Image.composite(Image.new("RGB", region.size, HAZE), region, self.g_haze.point(lambda v: int(v * ub)))
            img.paste(region, (0, 0))
            d = ImageDraw.Draw(img)

        # ================= overlays =================
        uz = clamp(u * 3)
        # route: springs along 7 legs, dotted; ink in 2D, accent in the air
        rtp = clamp(track(t, [(8.0 + 0.25 * i, 1 / 7.0) for i in range(7)] + [(22.8 + 0.07 * i, -1 / 7.0) for i in range(7)], "default"))
        rtp = 0.0 if rtp < 1e-3 else rtp
        route_pts = []
        if rtp > 0:
            pts = []
            for i in range(len(WP) - 1):
                for j in range(10):
                    tt = j / 10
                    pts.append(tuple(lerp(WP[i][k], WP[i + 1][k], tt) for k in range(3)))
            pts.append(WP[-1])
            n = max(2, int(rtp * len(pts)))
            rcol = lerpc(INK, ACC, uz)
            for i in range(0, n - 1, 2):
                a, b = pts[i], pts[i + 1]
                d.line([PJ(a[0], a[1], a[2] * uz), PJ(b[0], b[1], b[2] * uz)], fill=rcol, width=max(1, int(2.2 * U / 1.3)))
            route_pts = pts[:n]
            ds_ = max(0.0, track(t, [(8.0, 1), (11.7, -1)], "snappy"))
            if ds_ > 0.02 and route_pts:
                hx_, hy_ = PJ(route_pts[-1][0], route_pts[-1][1], 0)
                r0 = 6 * U * ds_
                d.ellipse([hx_ - r0, hy_ - r0, hx_ + r0, hy_ + r0], fill=ACC)

        # pins (2D)
        for k, wi in enumerate(WPIDX):
            tin = 8.0 if wi == 0 else 8.0 + 0.25 * (wi - 1) + 0.2
            ps = spring(t, tin, "pop") - spring(t, 11.7, "snappy")
            if ps > 0.03:
                x, y, z = WP[wi]
                px, py = PJ(x, y, 0)
                r0 = 9 * U * ps
                hero = k in (0, 6)
                d.ellipse([px - r0, py - r0, px + r0, py + r0], fill=ACC if hero else INK)
                if ps > 0.4:
                    d.text((px, py), str(k + 1), font=font(8.5 * U * ps, "bold"), fill=INK if hero else PAPER, anchor="mm")
        # eras badges (2D)
        for i, (ch, x, y) in enumerate(BADGES):
            ps = spring(t, 10.0 + 0.15 * i, "pop") - spring(t, 11.7, "snappy")
            if ps > 0.03:
                px, py = PJ(x, y, 0)
                r0 = 11 * U * ps
                d.ellipse([px - r0, py - r0, px + r0, py + r0], fill=INK, outline=ACC, width=max(1, int(2 * U / 2)))
                if ps > 0.4:
                    d.text((px, py), ch, font=font(9 * U * ps, "bold"), fill=PAPER, anchor="mm")

        # 2D labels (mask wipes)
        for (txt, x, y, tin, tout, sz) in LABELS2D:
            f = track(t, [(tin, 1), (tout, -1)], "snappy")
            if f > 0.01:
                px, py = PJ(x, y, 0)
                fnt = font(sz * U, "mono")
                tw = d.textlength(txt, font=fnt)
                w_, h_ = tw + 10 * U, sz * 1.9 * U

                def fnl(dd, w_=w_, h_=h_, txt=txt, fnt=fnt):
                    dd.rectangle([0, 0, w_, h_], fill=PAPER)
                    dd.text((w_ / 2, h_ / 2), txt, font=fnt, fill=INK2 if txt != "542 m" else INK, anchor="mm")
                self.wipe(img, px - w_ / 2, py - h_ / 2, w_, h_, f, fnl)

        # legends (2D)
        fl = track(t, [(8.0, 1), (10.0, -1)], "snappy")
        if fl > 0.01:
            lw, lh = 232 * U, (26 + len(WPIDX) * 19) * U

            def fnleg(dd):
                dd.rounded_rectangle([0, 0, lw, lh], radius=10 * U, fill=INK)
                for k in range(len(WPIDX)):
                    yy = (20 + k * 19) * U
                    dd.text((12 * U, yy), str(k + 1), font=font(7.5 * U, "bold"), fill=ACC, anchor="lm")
                    dd.text((30 * U, yy), WPNAMES[k], font=font(7.5 * U, "reg"), fill=(240, 236, 226), anchor="lm")
            self.wipe(img, W * SS - 246 * U, H * SS - lh - 14 * U, lw, lh, fl, fnleg)
        fe = track(t, [(10.0, 1), (11.7, -1)], "snappy")
        if fe > 0.01:
            lw, lh = 304 * U, 108 * U

            def fneras(dd):
                dd.rounded_rectangle([0, 0, lw, lh], radius=12 * U, fill=INK)
                dd.text((14 * U, 14 * U), "THREE AGES IN ONE CITY", font=font(7 * U, "mono"), fill=(170, 166, 158), anchor="la")
                for i, (c, tx) in enumerate([("M", "Medieval  ·  fortress begun 1077"), ("R", "Renaissance  ·  Wolf Dietrich 1587–1612"), ("B", "Baroque  ·  cathedral consecrated 1628")]):
                    yy = (40 + i * 22) * U
                    dd.ellipse([14 * U, yy - 8 * U, 30 * U, yy + 8 * U], fill=ACC)
                    dd.text((22 * U, yy), c, font=font(7.5 * U, "bold"), fill=INK, anchor="mm")
                    dd.text((40 * U, yy), tx, font=font(7.5 * U, "reg"), fill=(240, 236, 226), anchor="lm")
            self.wipe(img, W * SS - 318 * U, H * SS - lh - 14 * U, lw, lh, fe, fneras)

        # north arrow (heading-up map: it points down), only in the top view
        fa = track(t, [(2.2, 1), (11.9, -1)], "snappy")
        if fa > 0.02 and u < 0.5:
            nx_, ny_ = (W - 40 * K) * SS, (H - 80 * K) * SS
            hg = 40 * U * min(fa, 1.0)
            d.polygon([(nx_, ny_ + hg), (nx_ - 9 * U, ny_ + hg * 0.3), (nx_, ny_ + hg * 0.475), (nx_ + 9 * U, ny_ + hg * 0.3)], fill=INK)
            d.text((nx_, ny_ + hg + 12 * U), "N", font=font(8 * U, "bold"), fill=INK, anchor="mm")

        # hook: tiny ring + crosshair + START label on the Hauptbahnhof (this is frame 0)
        hx_, hy_ = PJ(HBF[0], HBF[1], 0)
        rr_ = 9 * U
        if u < 0.08:
            d.ellipse([hx_ - rr_, hy_ - rr_, hx_ + rr_, hy_ + rr_], outline=INK, width=max(1, int(1.8 * U / 1.5)))
        arm = max(0.0, 22 + track(t, [(1.0, 26), (2.0, -38), (12.0, -10), (23.1, 22)], "snappy"))
        if arm > 0.5:
            d.line([(hx_ - arm * U, hy_), (hx_ + arm * U, hy_)], fill=ACC, width=max(1, int(1.6 * U / 1.5)))
            d.line([(hx_, hy_ - arm * U), (hx_, hy_ + arm * U)], fill=ACC, width=max(1, int(1.6 * U / 1.5)))
        f0 = 1 + track(t, [(2.0, -1), (23.1, 1)], "snappy")
        if f0 > 0.01:
            w_, h_ = 110 * U, 34 * U

            def fn0(dd):
                dd.text((0, 4 * U), "START", font=font(8.5 * U, "bold"), fill=INK, anchor="la")
                dd.text((0, 20 * U), "Hauptbahnhof", font=font(7.5 * U, "mono"), fill=INK2, anchor="la")
            self.wipe(img, hx_ + 18 * U, hy_ - 30 * U, w_, h_, f0, fn0)

        # drone
        dsc = max(0.0, track(t, [(16.0, 1), (23.3, -1)], "snappy"))
        if dsc > 0.03:
            dx_, dy_, dz_ = self.drone_pos(t)
            ex, ey, ez = self.drone_pos(t + 0.04)
            bx, by, bz = self.drone_pos(t - 0.04)
            hd = math.atan2(ey - by, ex - bx) if math.hypot(ex - bx, ey - by) > 1e-4 else 0.0
            shp = [PJ(dx_ - dz_ * 0.25 + 7 * dsc * math.cos(i / 16 * 6.2832), dy_ - dz_ * 0.20 + 7 * dsc * math.sin(i / 16 * 6.2832), 0) for i in range(16)]
            d.polygon(shp, fill=(196, 188, 172))
            trail = []
            for k in range(0, 10):
                tk = t - 0.11 * k
                if tk >= 16.0:
                    px3, py3, pz3 = self.drone_pos(tk)
                    trail.append(PJ(px3, py3, pz3))
            if len(trail) > 1:
                d.line(trail, fill=ACC, width=max(1, int(2 * U / 1.3)))
            arm_ = 15.0 * dsc
            for sx_, sy_ in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                ox = (sx_ * math.cos(hd) - sy_ * math.sin(hd)) * arm_ * 0.75
                oy = (sx_ * math.sin(hd) + sy_ * math.cos(hd)) * arm_ * 0.75
                d.line([PJ(dx_, dy_, dz_), PJ(dx_ + ox, dy_ + oy, dz_)], fill=INK, width=max(1, int(2.4 * U / 1.3)))
                rp = [PJ(dx_ + ox + 8 * dsc * math.cos(i / 14 * 6.2832), dy_ + oy + 8 * dsc * math.sin(i / 14 * 6.2832), dz_) for i in range(14)]
                d.polygon(rp, fill=(240, 236, 226), outline=INK, width=max(1, int(1.4 * U / 1.3)))
            d.polygon([PJ(dx_ + 5.5 * dsc * math.cos(i / 14 * 6.2832), dy_ + 5.5 * dsc * math.sin(i / 14 * 6.2832), dz_) for i in range(14)], fill=INK)
            fx, fy = PJ(dx_ + math.cos(hd) * 5.5 * dsc, dy_ + math.sin(hd) * 5.5 * dsc, dz_)
            r0 = 2.4 * U * dsc
            d.ellipse([fx - r0, fy - r0, fx + r0, fy + r0], fill=ACC)

        # info card (3D half), like the "6:15 am" card in the reference
        fc = track(t, [(12.5, 1), (23.1, -1)], "snappy")
        if fc > 0.01:
            cw_, ch_ = 196 * U, 46 * U

            def fnc(dd):
                dd.rectangle([0, 0, cw_, ch_], fill=INK)
                dd.text((10 * U, 20 * U), "Salzburg", font=font(13.5 * U, "cond"), fill=PAPER, anchor="lm")
                dd.text((10 * U, 37 * U), "Altstadt · UNESCO 1996", font=font(6.5 * U, "condreg"), fill=(190, 186, 178), anchor="lm")
            self.wipe(img, 14 * U, 14 * U, cw_, ch_, fc, fnc)

        # callout tags: a leader line grows from the landmark, the pill wipes open
        for (txt, (wx, wy, wz), (ddx, ddy), tin, tout) in TAGS:
            f = track(t, [(tin, 1), (tout, -1)], "snappy")
            if f > 0.02 and u > 0.3:
                ax, ay = PJ(wx, wy, wz * uz)
                px, py = ax + ddx * U, ay + ddy * U
                fnt = font(10 * U, "mono")
                tw = d.textlength(txt, font=fnt)
                fp = min(f, 1.0)
                d.line([(ax, ay), (lerp(ax, px, fp), lerp(ay, py, fp))], fill=ACC, width=max(1, int(2 * U / 1.3)))
                w_, h_ = tw + 20 * U, 24 * U

                def fnt_(dd, w_=w_, h_=h_, txt=txt, fnt=fnt):
                    dd.rounded_rectangle([0, 0, w_ - 1, h_ - 1], radius=11 * U, fill=(20, 20, 20), outline=ACC, width=max(1, int(2 * U / 2)))
                    dd.text((10 * U, h_ / 2), txt, font=fnt, fill=(250, 250, 250), anchor="lm")
                self.wipe(img, px - 9 * U, py - h_ / 2, w_, h_, fp, fnt_)

        # fortress callout card (white, dark border, condensed bold) springs open
        cs = track(t, [(22.3, 1), (22.95, -1)], "pop")
        if cs > 0.05:
            bx, by = W * SS * 0.80, 118 * U
            bw, bh_ = 190 * U * cs, 62 * U * cs
            fx, fy = PJ(-45, 88, ztop + 36)
            d.line([(bx, by), (fx, fy)], fill=INK, width=max(1, int(2 * U / 1.3)))
            d.rectangle([bx - bw / 2, by - bh_, bx + bw / 2, by], fill=(255, 255, 255), outline=INK, width=max(1, int(2 * U * cs / 1.3)))
            if cs > 0.3:
                d.text((bx, by - bh_ * 0.68), "Hohensalzburg", font=font(13.5 * U * cs, "cond"), fill=INK, anchor="mm")
                d.text((bx, by - bh_ * 0.28), "begun 1077 · 542 m", font=font(9.5 * U * cs, "condreg"), fill=INK, anchor="mm")

        return img


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
        return np.sin(2 * np.pi * (45 * x + (95 / 30.0) * (1 - np.exp(-30 * x)))) * np.exp(-x * 9)

    def bass(f):
        x = tt(0.95)
        return (np.sin(2 * np.pi * f * x) + 0.28 * np.sin(4 * np.pi * f * x) * np.exp(-x * 5)) * np.exp(-x * 3.2)

    def pluck(f):
        x = tt(0.7)
        return sum((1.0 / h) * np.sin(2 * np.pi * f * h * x) * np.exp(-x * (5 + 3 * h)) for h in range(1, 5)) * np.minimum(1, x / 0.003)

    def hat():
        x = tt(0.09)
        return np.diff(rng.standard_normal(len(x)), prepend=0) * np.exp(-x * 55)

    def tick(f=1500.0):
        x = tt(0.07)
        return np.sin(2 * np.pi * f * x) * np.exp(-x * 60)

    def click():
        x = tt(0.03)
        return (np.sin(2 * np.pi * 3200 * x) + 0.5 * rng.standard_normal(len(x))) * np.exp(-x * 160)

    def pop():
        x = tt(0.14)
        return np.sin(2 * np.pi * (420 * x + 300 * x * x)) * np.exp(-x * 24)

    def thump():
        x = tt(0.6)
        return np.sin(2 * np.pi * (52 * x + (70 / 14.0) * (1 - np.exp(-14 * x)))) * np.exp(-x * 6.5)

    def whoosh(dur, peak):
        m = int(dur * SR)
        z = rng.standard_normal(m)
        x = np.arange(m) / SR
        env = np.sin(np.pi * np.clip(x / dur, 0, 1) ** (math.log(0.5) / math.log(peak / dur))) ** 2
        a = 1 - np.exp(-2 * np.pi * (300 + 5200 * env) / SR)
        y = np.empty(m)
        s = 0.0
        for i in range(m):
            s += a[i] * (z[i] - s)
            y[i] = s
        return y * env * 3.0

    # ---- music: Am F C G, three times through 12 bars ----------------------------------------
    roots = [33, 29, 36, 31]
    chords = [(69, 72, 76), (65, 69, 72), (72, 76, 79), (67, 71, 74)]
    pattern = [0, 1, 2, 1, 0, 2, 1, 2]
    for bar in range(12):
        b0 = bar * 2.0
        r = midi(roots[bar % 4])
        if bar < 11:
            add(music, b0, bass(r), 0.55)
            add(music, b0 + 1.0, bass(r), 0.45)
            if bar >= 6:
                add(music, b0 + 1.5, bass(r * 1.5), 0.25)
        else:
            add(music, b0, bass(r), 0.5)
        if 1 <= bar <= 11:
            for j in range(8):
                if bar == 11 and j >= 4:
                    break
                note = chords[bar % 4][pattern[j]] + (12 if (j % 4 == 3 and bar >= 6) else 0)
                add(music, b0 + 0.25 * j, pluck(midi(note)), 0.20 if bar < 6 else 0.25)
        if 2 <= bar <= 10:
            for k in range(4):
                add(music, b0 + 0.25 + 0.5 * k, hat(), 0.16)
    for k in range(int(23.0 / BEAT)):
        add(music, k * BEAT, kick(), 0.85)

    # ---- SFX: the same times the picture uses ----------------------------------------------
    cues = [(0.0, "thump", 0.9)]
    cues += [(2.0 + 0.4 * k, "tick", 0.4) for k in range(3)] + [(2.0, "whoosh1", 0.35)]
    cues += [(4.0 + 0.7 * k, "tick", 0.5) for k in range(3)]
    cues += [(6.0 + 0.25 * k, "click", 0.25) for k in range(6)] + [(7.0, "pop", 0.5)]
    cues += [(8.0 + 0.25 * k, ["tick", "tick2", "tick3"][k % 3], 0.4) for k in range(7)]
    cues += [(10.0 + 0.15 * k, "click", 0.45) for k in range(8)]
    cues += [(12.0, "thump", 1.0), (12.0, "whoosh2", 0.75)]
    cues += [(14.0 + 0.5 * k, "tick", 0.35) for k in range(4)]
    cues += [(16.0, "whoosh3", 0.6)]
    cues += [(tin, "pop", 0.55) for (_t, _a, _o, tin, _x) in TAGS]
    cues += [(22.3, "pop", 0.8), (23.0, "whoosh4", 0.7), (23.6, "tick", 0.5)]
    cache = {}
    for tm, kind, gain in cues:
        if kind not in cache:
            cache[kind] = {"thump": thump, "click": click, "pop": pop, "tick": lambda: tick(1500), "tick2": lambda: tick(1900),
                           "tick3": lambda: tick(2400), "whoosh1": lambda: whoosh(1.6, 0.8), "whoosh2": lambda: whoosh(1.6, 0.55),
                           "whoosh3": lambda: whoosh(1.2, 0.5), "whoosh4": lambda: whoosh(1.4, 0.5)}[kind]()
        add(sfx, tm, cache[kind], gain)

    # drone hum 16 -> 23.2 s (swells as it passes the camera side), car passes, footsteps
    m0, m1 = int(16.0 * SR), int(23.2 * SR)
    xh = np.arange(m1 - m0) / SR
    tabs = 16.0 + xh
    envh = np.clip(xh / 0.8, 0, 1) * np.clip((23.2 - tabs) / 0.9, 0, 1)
    near = 0.65 + 0.35 * np.sin(2 * np.pi * xh / 3.5)
    f0 = 150.0 * (1 + 0.01 * np.sin(2 * np.pi * 0.7 * xh))
    phh = 2 * np.pi * np.cumsum(f0) / SR
    hum = (np.sin(phh) + 0.5 * np.sin(2 * phh) + 0.3 * np.sin(3 * phh)) * (0.65 + 0.35 * np.sin(2 * np.pi * 48 * xh))
    add(sfx, 16.0, hum * envh * near, 0.05)
    car_rng = np.random.default_rng(77)
    for k in range(8):
        add(sfx, 15.6 + k * 0.8 + float(car_rng.uniform(0, 0.3)), whoosh(0.7, 0.35), 0.08)
    for k in range(10):
        tm = 16.0 + k * 0.6
        x = tt(0.03)
        step = np.diff(rng.standard_normal(len(x)), prepend=0) * np.exp(-x * 140)
        add(sfx, tm, step, 0.09 + 0.04 * (k % 2))

    mix = 0.55 * music + 0.75 * sfx
    mix = np.tanh(mix * 1.1)
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


def encode(args, audio_path):
    W, H = dims(args.res)
    N = int(round(DUR * args.fps))
    out = os.path.join(args.out, "salzburg_16x9.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(args.fps), "-i", "-"]
    if audio_path:
        cmd += ["-i", audio_path]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]
    if audio_path:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ac", "2", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,apad", "-t", str(DUR)]
    cmd += ["-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    workers = args.workers or max(1, cpu_count() - 1)
    t0 = time.time()
    print("%dx%d, %d frames, %d workers, supersample %dx, motion blur %d" % (W, H, N, workers, args.ss, args.mblur))
    with Pool(workers, initializer=_init, initargs=(W, H, args.ss, args.font, args.mblur, args.shutter, args.fps)) as pool:
        for i, buf in enumerate(pool.imap(_frame, range(N), chunksize=2)):
            proc.stdin.write(buf)
            if i % 10 == 0 or i == N - 1:
                el = time.time() - t0
                sys.stdout.write("\r   frame %d/%d   elapsed %ds   eta %ds   " % (i + 1, N, el, el / (i + 1) * (N - i - 1)))
                sys.stdout.flush()
    proc.stdin.close()
    proc.wait()
    print("\n   -> %s" % out)


def main():
    ap = argparse.ArgumentParser(description="Render the Salzburg drone film (pure Python + ffmpeg).")
    ap.add_argument("--res", type=int, default=720, help="short side in px (720 = 1280x720, 1080 = 1920x1080)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ss", type=int, default=2, help="supersampling for anti-aliasing (1 = fast, 2 = good)")
    ap.add_argument("--mblur", type=int, default=1, help="motion-blur subframes per frame (1 = off, 3-4 = final)")
    ap.add_argument("--shutter", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (default: CPU count - 1)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--font", default=None, help="path to a .ttf for display type")
    ap.add_argument("--still", type=float, default=None, help="render one PNG at this time (s) and exit")
    ap.add_argument("--sheet", action="store_true", help="write a contact sheet (one frame per beat) and exit")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0, help="contact sheet start time")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W, H = dims(args.res)
    if args.still is not None or args.sheet:
        _init(W, H, args.ss, args.font, 1, 0.5, args.fps)
        if args.still is not None:
            p = os.path.join(args.out, "still_%05.2fs.png" % args.still)
            _SC.render(args.still).resize((W, H), Image.LANCZOS).save(p)
            print(p)
        else:
            ts = [args.t_from + i * BEAT for i in range(48)]
            cols, tw = 8, 320
            th_ = int(tw * H / W)
            sheet = Image.new("RGB", (cols * tw, (len(ts) // cols) * th_), (14, 14, 14))
            for i, tm in enumerate(ts):
                sheet.paste(_SC.render(tm).resize((tw, th_), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th_))
            p = os.path.join(args.out, "contact_sheet.png")
            sheet.save(p)
            print(p)
        return
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg was not found. Install it first (macOS: brew install ffmpeg, Ubuntu: sudo apt install ffmpeg).")
    audio = tmp = None
    if not args.no_audio:
        tmp = tempfile.mkdtemp()
        audio = os.path.join(tmp, "score.wav")
        print("synthesizing score + SFX ...")
        synth_audio(audio)
    encode(args, audio)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print("done.")


if __name__ == "__main__":
    main()
