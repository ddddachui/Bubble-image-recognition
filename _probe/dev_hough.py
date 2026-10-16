# -*- coding: utf-8 -*-
"""开发用：在 rel 响应图上做霍夫圆检测。"""
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


def detect(g, mask, p2=28, minR=18, maxR=75, minDist=38, lo=0.03, hi=0.25):
    rel = rel_map(g, mask)
    img = np.clip((rel - lo) / (hi - lo), 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    c = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.5, minDist=minDist,
                         param1=120, param2=p2, minRadius=minR, maxRadius=maxR)
    out = [] if c is None else [tuple(map(float, x)) for x in c[0]]
    return rel, img, out


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
        print("t=%.1fs" % t)
        panels = []
        for p2 in (22, 28, 34):
            rel, img, cs = detect(g, m, p2=p2)
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            for (cx, cy, r) in cs:
                cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), 2)
                cv2.circle(vis, (int(cx), int(cy)), 2, (255, 0, 255), -1)
            cv2.putText(vis, "p2=%d n=%d" % (p2, len(cs)), (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            print("   p2=%d -> %d circles, radii=%s" % (p2, len(cs), sorted([int(c[2]) for c in cs], reverse=True)[:12]))
            panels.append(vis)
        rows.append(np.hstack(panels))
    if rows:
        cv2.imwrite(OUT + r"\dev_hough.png", np.vstack(rows))
        print("saved dev_hough.png")
    cap.release()


if __name__ == "__main__":
    main()
