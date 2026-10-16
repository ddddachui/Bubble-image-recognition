# -*- coding: utf-8 -*-
"""开发用 v4：median 背景 + 自适应阈值 + 闭运算补形 + 分水岭拆分。"""
import sys
import numpy as np
import cv2

VIDEO = r"E:\Bubble image recognition\video1.mp4"
OUT = r"E:\Bubble image recognition\_probe"
SCALE = 0.5
MED_K = 151          # 背景中值核
OPEN_R = 8           # 去除氮化镓孔洞(半分辨率约12-15px)
CLOSE_K = 41         # 把月牙形补成整圆
MIN_AREA = 900


def aperture_mask(g):
    lit = (g > 45).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lit, 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cnts, -1, 1, -1)
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))


def analyze(g, mask, thr=None, verbose=True):
    bg = cv2.medianBlur(g, MED_K).astype(np.float32)
    rel = (bg - g.astype(np.float32)) / (bg + 1e-6)
    rel[~mask.astype(bool)] = 0.0
    if thr is None:
        vals = rel[mask.astype(bool)]
        thr = float(np.percentile(vals, 92))
        thr = min(max(thr, 0.04), 0.12)
    binary = (rel > thr).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (2 * OPEN_R + 1, 2 * OPEN_R + 1)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K)))
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    # ---- 分水岭拆分粘连 ----
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    dsm = cv2.GaussianBlur(dist, (0, 0), 4)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    peaks = ((dsm == cv2.dilate(dsm, ker)) & (dsm > 10)).astype(np.uint8)
    nlab, markers = cv2.connectedComponents(peaks, 8)
    markers = markers + 1
    markers[filled == 0] = 1
    img3 = cv2.cvtColor(np.clip(rel * 800, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img3, markers)
    objs = []
    for i in range(2, nlab + 1):
        comp = ((markers == i) & (filled > 0)).astype(np.uint8)
        a = int(comp.sum())
        if a < MIN_AREA:
            continue
        mm = cv2.moments(comp, True)
        objs.append(dict(area=a, cx=mm["m10"] / a, cy=mm["m01"] / a, comp=comp))
    if verbose:
        print("   thr=%.3f  原始连通域=%d  拆分后对象=%d  面积=%s"
              % (thr, int((cv2.connectedComponentsWithStats(filled, 8)[0]) - 1), len(objs),
                 sorted([o["area"] for o in objs], reverse=True)[:12]))
    return rel, filled, markers, objs


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
        print("t=%.1f s" % t)
        rel, filled, markers, objs = analyze(g, m)
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        vis[filled > 0] = (0.5 * vis[filled > 0] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
        for j, o in enumerate(sorted(objs, key=lambda z: -z["area"])):
            cnts, _ = cv2.findContours(o["comp"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
            cv2.putText(vis, "%d" % j, (int(o["cx"]) - 8, int(o["cy"]) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        relv = cv2.cvtColor(np.clip(rel * 700, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, "t=%.1f  n=%d" % (t, len(objs)), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        rows.append(np.hstack([vis, relv]))
    if rows:
        cv2.imwrite(OUT + r"\dev_seg4.png", np.vstack(rows))
        print("saved dev_seg4.png")
    cap.release()


if __name__ == "__main__":
    main()
