# -*- coding: utf-8 -*-
"""开发用 v2：诊断对比度 + 自适应阈值分割。"""
import sys
import numpy as np
import cv2

VIDEO = r"E:\Bubble image recognition\video1.mp4"
OUT = r"E:\Bubble image recognition\_probe"
SCALE = 0.5


def aperture_mask(g):
    lit = (g > 45).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lit, 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cnts, -1, 1, -1)
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))


def rel_map(g, mask, K=201):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K, K))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, ker)
    bg = cv2.GaussianBlur(bg, (0, 0), K / 5.0).astype(np.float32)
    rel = (bg - g.astype(np.float32)) / (bg + 1e-6)
    rel[~mask.astype(bool)] = 0
    return rel


def main():
    times = [float(x) for x in sys.argv[1:]] or [52.0]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        if not ok:
            print("read fail", t); continue
        small = cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        m = aperture_mask(g)
        rel = rel_map(g, m)
        r = rel[m.astype(bool)]
        print("t=%6.1fs  rel: p50=%.4f p90=%.4f p95=%.4f p99=%.4f max=%.4f"
              % (t, np.percentile(r, 50), np.percentile(r, 90),
                 np.percentile(r, 95), np.percentile(r, 99), r.max()))
        pos = rel[rel > 0.01]
        hist = (np.clip(pos, 0, 0.5) * 100).astype(np.uint8)
        otsu, _ = cv2.threshold(hist.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print("           otsu on positive rel = %.4f" % (otsu / 100.0))
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        for thr in (0.04, 0.06, 0.08, 0.10):
            b = (rel > thr).astype(np.uint8)
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
            nn, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
            areas = sorted([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, nn) if stats[i, cv2.CC_STAT_AREA] >= 500],
                           reverse=True)
            print("           thr=%.2f -> %d comps >=500px: %s" % (thr, len(areas), areas[:10]))
            if abs(thr - 0.06) < 1e-9:
                for i in range(1, nn):
                    a = stats[i, cv2.CC_STAT_AREA]
                    if a >= 500:
                        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        relv = cv2.cvtColor(np.clip(rel * 800, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        board = np.hstack([vis, relv])
        cv2.putText(board, "t=%.1f" % t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        rows.append(board)
    if rows:
        cv2.imwrite(OUT + r"\dev_seg2.png", np.vstack(rows))
        print("saved dev_seg2.png")
    cap.release()


if __name__ == "__main__":
    main()
