# -*- coding: utf-8 -*-
"""开发用：比较不同背景估计方法（强镜面反光是主要干扰）。"""
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


def bg_closing(g, K=151):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K, K))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, ker)
    return cv2.GaussianBlur(bg, (0, 0), K / 5.0)


def bg_closing_clipped(g, K=151, pct=97):
    hi = np.percentile(g[g > 45], pct)
    gc = np.minimum(g, hi)
    return bg_closing(gc.astype(np.uint8), K)


def bg_median(g, K=151):
    return cv2.medianBlur(g, K if K % 2 else K + 1)


def bg_pyramid(g, f=8, k=7):
    h, w = g.shape
    s = cv2.resize(g, (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA)
    s = cv2.medianBlur(s, k if k % 2 else k + 1)
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)


def bg_hybrid(g, f=8, k=11, K=91):
    """金字塔粗估 + 小核闭运算修正（对强反光稳健，且能贴合气泡边界）"""
    h, w = g.shape
    s = cv2.resize(g, (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA)
    s = cv2.medianBlur(s, k if k % 2 else k + 1)
    coarse = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.int16)
    # 用粗背景把反光压回合理范围，再做小核闭运算
    cap = np.minimum(g.astype(np.int16), coarse + 25).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K, K))
    fine = cv2.morphologyEx(cap, cv2.MORPH_CLOSE, ker)
    return cv2.GaussianBlur(fine, (0, 0), K / 4.0)


METHODS = [("closing151", bg_closing), ("clipped", bg_closing_clipped),
           ("median151", bg_median), ("pyramid8", bg_pyramid), ("hybrid", bg_hybrid)]


def main():
    times = [float(x) for x in sys.argv[1:]] or [52.5]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        if not ok:
            continue
        small = cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        m = aperture_mask(g).astype(bool)
        print("=== t=%.1f s ===" % t)
        rows = []
        for name, fn in METHODS:
            import time as _t
            t0 = _t.time()
            bg = fn(g).astype(np.float32)
            dt = _t.time() - t0
            rel = (bg - g.astype(np.float32)) / (bg + 1e-6)
            rel[~m] = 0
            frac = float((rel > 0.05).sum()) / max(1, m.sum())
            b = (rel > 0.06).astype(np.uint8)
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
            nn, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
            areas = sorted([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, nn)
                            if stats[i, cv2.CC_STAT_AREA] >= 800], reverse=True)
            print("  %-12s %.0fms  rel p50=%.3f p90=%.3f | frac>0.05=%.3f | comps>=800: %s"
                  % (name, dt * 1000, np.percentile(rel[m], 50), np.percentile(rel[m], 90),
                     frac, areas[:8]))
            relv = cv2.cvtColor(np.clip(rel * 800, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.putText(relv, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            rows.append(relv)
        cv2.imwrite(OUT + r"\dev_bg_%d.png" % int(t), np.vstack(rows))
    cap.release()
    print("saved dev_bg_*.png")


if __name__ == "__main__":
    main()
