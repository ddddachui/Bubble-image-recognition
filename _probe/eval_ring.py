# -*- coding: utf-8 -*-
"""用人工标注评估环形滤波器的准确率/召回率。"""
import json
import os
import time

import numpy as np
import cv2

import ring_filter as RF

ROOT = r"E:\Bubble image recognition"
ANN = os.path.join(ROOT, "标注工具", "bubble_annotation.json")
FR = os.path.join(ROOT, "标注工具", "frames")
SCALE = 0.5


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def aperture(g):
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats((g > 45).astype(np.uint8), 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m2 = np.zeros_like(m)
    cv2.drawContours(m2, cnts, -1, 1, -1)
    return cv2.erode(m2, np.ones((15, 15), np.uint8))


def build_radii(lo=10, hi=195, ratio=1.16):
    rs, r = [], lo
    while r <= hi:
        rs.append(r)
        r *= ratio
    return rs


def evaluate(thr, radii, data, cache):
    tp = fp = fn = 0
    ghost_hits = 0
    ghost_tot = 0
    used_det = 0
    for a in data:
        key = a["id"]
        if key not in cache:
            continue
        g, m = cache[key]
        det = RF.detect(g, m, radii, score_thr=thr)
        det = [(s, x / SCALE, y / SCALE, r / SCALE) for s, x, y, r in det]
        used_det += len(det)
        hit = [False] * len(det)
        for b in a["bubbles"]:
            best, bi = None, -1
            for i, (s, x, y, r) in enumerate(det):
                d = np.hypot(x - b["x"], y - b["y"])
                if d < 0.55 * b["r"] and 0.45 < r / b["r"] < 2.1:
                    if best is None or d < best:
                        best, bi = d, i
            if bi >= 0:
                tp += 1
                hit[bi] = True
            else:
                fn += 1
        fp += sum(1 for h in hit if not h)
        for gh in a["ghosts"]:
            ghost_tot += 1
            if any(np.hypot(x - gh["x"], y - gh["y"]) < 90 for _, x, y, r in det):
                ghost_hits += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return dict(thr=thr, tp=tp, fp=fp, fn=fn, prec=prec, rec=rec,
                f1=2 * prec * rec / max(1e-9, prec + rec),
                ghost_hit=ghost_hits, ghost_tot=ghost_tot, ndet=used_det)


def main():
    d = json.load(open(ANN, encoding="utf-8"))
    data = d["annotated"]
    cache = {}
    for a in data:
        img = imread_u(os.path.join(FR, a["id"] + "_orig.png"))
        if img is None:
            continue
        small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        cache[a["id"]] = (small, aperture(small))
    print("载入 %d 帧" % len(cache))

    radii = build_radii()
    print("尺度 %d 个: %.0f ... %.0f (半分辨率)" % (len(radii), radii[0], radii[-1]))
    t0 = time.time()
    print("\n阈值扫描：")
    print(" thr    检\tTP\tFP\tFN\t准确率\t召回率\tF1\t虚影误检")
    best = None
    for thr in (0, 1, 2, 3, 4, 5, 6, 8):
        r = evaluate(thr, radii, data, cache)
        print(" %4.1f  %4d\t%3d\t%3d\t%3d\t%.2f\t%.2f\t%.2f\t%d/%d"
              % (r["thr"], r["ndet"], r["tp"], r["fp"], r["fn"],
                 r["prec"], r["rec"], r["f1"], r["ghost_hit"], r["ghost_tot"]))
        if best is None or r["f1"] > best["f1"]:
            best = r
    print("\n用时 %.1fs；最佳 thr=%.1f  F1=%.2f (准确率 %.2f, 召回率 %.2f)"
          % (time.time() - t0, best["thr"], best["f1"], best["prec"], best["rec"]))


if __name__ == "__main__":
    main()
