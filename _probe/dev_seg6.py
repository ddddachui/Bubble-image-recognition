# -*- coding: utf-8 -*-
"""开发用 v6：滞回阈值 + 自适应腐蚀取种子 + 最近种子划分（替代分水岭）。"""
import sys
import time as _t
import numpy as np
import cv2

VIDEO = r"E:\Bubble image recognition\video1.mp4"
OUT = r"E:\Bubble image recognition\_probe"
SCALE = 0.5
MED_K = 151
LO, HI = 0.030, 0.100
MIN_AREA = 700          # 半分辨率像素
MAX_AREA = 40000


def aperture_mask(g, erode=21):
    lit = (g > 45).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lit, 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cnts, -1, 1, -1)
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)))


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r) + 1, 2 * int(r) + 1))


def rel_map(g, mask):
    bg = cv2.medianBlur(g, MED_K).astype(np.float32)
    rel = (bg - g.astype(np.float32)) / (bg + 1e-6)
    rel[~mask.astype(bool)] = 0.0
    return rel


def blob_mask(rel, mask):
    strong = (rel > HI).astype(np.uint8)
    weak = (rel > LO).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_OPEN, disk(6))
    n, lab = cv2.connectedComponents(weak, 8)
    keep = np.zeros_like(weak)
    for i in range(1, n):
        sel = lab == i
        if int(strong[sel].sum()) >= 25:
            keep[sel] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, disk(17))
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fill = np.zeros_like(keep)
    cv2.drawContours(fill, cnts, -1, 1, -1)
    return fill


def split_blobs(fill, min_area=MIN_AREA, max_area=MAX_AREA):
    dist = cv2.distanceTransform(fill, cv2.DIST_L2, 5)
    n, lab = cv2.connectedComponents(fill, 8)
    seed = np.zeros_like(fill)
    for i in range(1, n):
        m = lab == i
        a = int(m.sum())
        if a < min_area:
            continue
        rr = float(dist[m].max())
        k = int(max(4, round(rr * 0.55)))
        er = cv2.erode(m.astype(np.uint8), disk(k))
        if er.sum() == 0:
            er = m.astype(np.uint8)
        seed[er > 0] = 1
    if seed.sum() == 0:
        return []
    inv = (seed == 0).astype(np.uint8)
    _, slabel = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    region = np.where(fill > 0, slabel, 0)
    objs = []
    for rid in np.unique(region):
        if rid == 0:
            continue
        comp = (region == rid).astype(np.uint8)
        a = int(comp.sum())
        if a < min_area or a > max_area:
            continue
        mm = cv2.moments(comp, True)
        r_eq = np.sqrt(a / np.pi)
        circ = 4 * np.pi * a / max(1.0, cv2.arcLength(
            cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0], True) ** 2)
        objs.append(dict(area=a, cx=mm["m10"] / a, cy=mm["m01"] / a,
                         r=r_eq, circ=circ, comp=comp))
    return objs


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
        t0 = _t.time()
        small = cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        m = aperture_mask(g)
        rel = rel_map(g, m)
        fill = blob_mask(rel, m)
        objs = split_blobs(fill)
        dt = (_t.time() - t0) * 1000
        print("t=%.1fs  %.0fms  fill=%d  对象=%d  面积=%s" %
              (t, dt, int(fill.sum()), len(objs),
               sorted([o["area"] for o in objs], reverse=True)[:14]))
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        vis[fill > 0] = (0.55 * vis[fill > 0] + 0.45 * np.array([0, 170, 0])).astype(np.uint8)
        for j, o in enumerate(sorted(objs, key=lambda z: -z["area"])):
            cnts, _ = cv2.findContours(o["comp"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
            cv2.circle(vis, (int(o["cx"]), int(o["cy"])), 2, (255, 0, 255), -1)
            cv2.putText(vis, "%d" % j, (int(o["cx"]) + 4, int(o["cy"]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis, "t=%.1f n=%d" % (t, len(objs)), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        rows.append(vis)
    if rows:
        cv2.imwrite(OUT + r"\dev_seg6.png", np.vstack(rows))
        print("saved dev_seg6.png")
    cap.release()


if __name__ == "__main__":
    main()
