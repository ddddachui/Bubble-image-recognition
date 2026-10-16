# -*- coding: utf-8 -*-
"""测试"时间累积证据"：把前后若干帧的响应图平均，噪声平均掉、真气泡留下来。"""
import os
import sys
import numpy as np
import cv2

ROOT = r"E:\Bubble image recognition"
sys.path.insert(0, ROOT)
from src import segment as S                                    # noqa: E402

VIDEO = os.path.join(ROOT, "video1.mp4")
OUT = os.path.join(ROOT, "_probe")
HALF = 12          # 前后各取多少帧（25fps -> 共 1s 窗口）


def main():
    times = [float(x) for x in sys.argv[1:]] or [53.0]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    for t in times:
        c = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, c - HALF)
        acc = None
        ap = None
        for k in range(2 * HALF + 1):
            ok, f = cap.read()
            if not ok:
                break
            s = cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
            if ap is None:
                ap = S.aperture_mask(g)
            rel = S.rel_response(g, ap)
            acc = rel if acc is None else acc + rel
        acc /= (2 * HALF + 1)
        print("=== t=%.1fs ===" % t)
        panels = []
        for tag, rr in (("single", rel), ("temporal", acc)):
            for lo, hi in ((0.030, 0.14), (0.020, 0.09)):
                fill = S.blob_mask(rr, ap, lo=lo, hi=hi, open_r=9, close_r=17)
                dsm = S.distance_map(fill)
                o = S.find_bubbles(fill, dsm, min_r=17.0, min_sep=15, prominence=0.70,
                                   max_r=62.0, merge_ratio=0.88)
                print("   %-8s lo=%.3f hi=%.2f  mask=%6d  n=%2d  r=%s"
                      % (tag, lo, hi, int(fill.sum()), len(o),
                         " ".join("%.0f" % x["r"] for x in sorted(o, key=lambda z: -z["r"])[:10])))
                vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                vis[fill > 0] = (0.6 * vis[fill > 0] + 0.4 * np.array([0, 160, 0])).astype(np.uint8)
                for x in o:
                    cv2.circle(vis, (int(x["cx"]), int(x["cy"])), int(round(x["r"])),
                               (0, 0, 255), 2)
                cv2.putText(vis, "%s lo=%.2f n=%d" % (tag, lo, len(o)), (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                panels.append(vis)
        cv2.imwrite(os.path.join(OUT, "temporal_%d.png" % int(t)), np.hstack(panels))
    cap.release()
    print("saved temporal_*.png")


if __name__ == "__main__":
    main()
