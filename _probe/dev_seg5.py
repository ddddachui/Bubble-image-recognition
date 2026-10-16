# -*- coding: utf-8 -*-
"""开发用 v5：滞回阈值（强核心 -> 弱范围）+ 距离变换分水岭。"""
import sys
import numpy as np
import cv2

VIDEO = r"E:\Bubble image recognition\video1.mp4"
OUT = r"E:\Bubble image recognition\_probe"
SCALE = 0.5
MED_K = 151


def aperture_mask(g):
    lit = (g > 45).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lit, 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cnts, -1, 1, -1)
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))


def rel_map(g, mask):
    bg = cv2.medianBlur(g, MED_K).astype(np.float32)
    rel = (bg - g.astype(np.float32)) / (bg + 1e-6)
    rel[~mask.astype(bool)] = 0.0
    return rel


def hysteresis(rel, mask, lo, hi, open_r=6):
    strong = (rel > hi).astype(np.uint8)
    weak = (rel > lo).astype(np.uint8)
    if open_r:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1))
        weak = cv2.morphologyEx(weak, cv2.MORPH_OPEN, k)
    n, lab = cv2.connectedComponents(weak, 8)
    keep = np.zeros_like(weak)
    for i in range(1, n):
        sel = (lab == i)
        if (strong[sel] > 0).sum() >= 30:
            keep[sel] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fill = np.zeros_like(keep)
    cv2.drawContours(fill, cnts, -1, 1, -1)
    return keep, fill


def split(fill, min_area, min_r):
    dist = cv2.distanceTransform(fill, cv2.DIST_L2, 5)
    dsm = cv2.GaussianBlur(dist, (0, 0), 3)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * min_r + 1, 2 * min_r + 1))
    peaks = ((dsm == cv2.dilate(dsm, ker)) & (dsm > min_r * 0.7)).astype(np.uint8)
    nlab, markers = cv2.connectedComponents(peaks, 8)
    if nlab <= 1:
        return [], dist
    m2 = markers.astype(np.int32) + 1
    m2[fill == 0] = 1
    tmp = np.zeros(fill.shape + (3,), np.uint8)
    cv2.watershed(tmp, m2)
    objs = []
    for i in range(2, nlab + 1):
        comp = ((m2 == i) & (fill > 0)).astype(np.uint8)
        a = int(comp.sum())
        if a < min_area:
            continue
        mm = cv2.moments(comp, True)
        objs.append(dict(area=a, cx=mm["m10"] / a, cy=mm["m01"] / a, comp=comp))
    return objs, dist


def main():
    times = [float(x) for x in sys.argv[1:]] or [52.5]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        if not ok:
            continue
        small = cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        m = aperture_mask(g)
        rel = rel_map(g, m)
        print("t=%.1fs" % t)
        panels = []
        for (lo, hi) in ((0.03, 0.10), (0.04, 0.12), (0.02, 0.08)):
            keep, fill = hysteresis(rel, m, lo, hi)
            objs, dist = split(fill, min_area=700, min_r=14)
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            vis[fill > 0] = (0.55 * vis[fill > 0] + 0.45 * np.array([0, 190, 0])).astype(np.uint8)
            for j, o in enumerate(sorted(objs, key=lambda z: -z["area"])):
                cnts, _ = cv2.findContours(o["comp"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
            cv2.putText(vis, "lo=%.2f hi=%.2f n=%d" % (lo, hi, len(objs)), (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            print("   lo=%.2f hi=%.2f -> fill=%d px, objects=%d, areas=%s"
                  % (lo, hi, int(fill.sum()), len(objs),
                     sorted([o["area"] for o in objs], reverse=True)[:12]))
            panels.append(vis)
        rows.append(np.hstack(panels))
    if rows:
        cv2.imwrite(OUT + r"\dev_seg5.png", np.vstack(rows))
        print("saved dev_seg5.png")
    cap.release()


if __name__ == "__main__":
    main()
