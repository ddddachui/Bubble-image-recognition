# -*- coding: utf-8 -*-
"""用 trackpy.locate 试检测气泡（气泡比背景暗 -> invert=True）。"""
import os
import sys
import numpy as np
import cv2
import trackpy as tp

ROOT = r"E:\Bubble image recognition"
OUT = os.path.join(ROOT, "_probe")
VIDEO = os.path.join(ROOT, "video1.mp4")


def frame_at(t, scale=0.5):
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * cap.get(cv2.CAP_PROP_FPS))))
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None, None
    s = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return f, cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)


def main():
    times = [float(x) for x in sys.argv[1:]] or [53.0]
    rows = []
    for t in times:
        full, g = frame_at(t)
        if g is None:
            continue
        # 圆形视场外是纯黑 -> 先把外部填成视场内中位数，避免边界被当成暗特征
        m = (g > 45).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        inside = cv2.erode((lab == k).astype(np.uint8), np.ones((41, 41), np.uint8))
        med = int(np.median(g[inside > 0]))
        g2 = g.copy()
        g2[inside == 0] = med
        print("=== t=%.1fs ===" % t)
        for diam in (45, 71, 101):
            f = tp.locate(g2, diameter=diam, invert=True, minmass=0,
                          separation=int(diam * 0.85), characterize=True,
                          percentile=64)
            ms = f["mass"].values if len(f) else np.array([])
            print("  diameter=%3d -> %2d 个特征; mass 分位 %s"
                  % (diam, len(f), np.round(np.percentile(ms, [10, 50, 90]), 0) if len(ms) else "-"))
            if diam == 71:
                vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                for _, r in f.iterrows():
                    cv2.circle(vis, (int(r.x), int(r.y)), int(diam // 2), (0, 0, 255), 2)
                    cv2.circle(vis, (int(r.x), int(r.y)), 2, (255, 0, 255), -1)
                cv2.putText(vis, "trackpy diam=71 n=%d" % len(f), (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                rows.append(vis)
    if rows:
        cv2.imwrite(os.path.join(OUT, "test_trackpy.png"), np.hstack(rows))
        print("saved test_trackpy.png")


if __name__ == "__main__":
    main()
