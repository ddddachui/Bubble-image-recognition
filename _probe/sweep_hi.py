# -*- coding: utf-8 -*-
"""扫描"强核心"阈值 hi，选出让误检最少的取值。"""
import os
import sys
import numpy as np
import cv2

ROOT = r"E:\Bubble image recognition"
sys.path.insert(0, ROOT)
from src import segment as S                                  # noqa: E402

VIDEO = os.path.join(ROOT, "video1.mp4")
OUT = os.path.join(ROOT, "_probe")
HIS = [0.10, 0.13, 0.16, 0.19, 0.22]
MIN_R = float(os.environ.get("MIN_R", "17"))


def main():
    times = [float(x) for x in sys.argv[1:]] or [52.5]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        if not ok:
            continue
        small = cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        m = S.aperture_mask(g)
        rel = S.rel_response(g, m)
        print("=== t=%.1fs ===" % t)
        panels = []
        for hi in HIS:
            fill = S.blob_mask(rel, m, lo=0.035, hi=hi, open_r=6, close_r=17)
            dsm = S.distance_map(fill)
            objs = S.find_bubbles(fill, dsm, min_r=MIN_R, min_sep=13,
                                  prominence=0.70, max_r=62.0, merge_ratio=0.88)
            areas = int(fill.sum())
            print("  hi=%.2f  mask=%6d px (%.3f of aperture)  n=%2d  r=%s"
                  % (hi, areas, areas / max(1, int(m.sum())), len(objs),
                     " ".join("%.0f" % o["r"] for o in
                              sorted(objs, key=lambda z: -z["r"])[:12])))
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            vis[fill > 0] = (0.6 * vis[fill > 0] + 0.4 * np.array([0, 160, 0])).astype(np.uint8)
            for o in objs:
                cv2.circle(vis, (int(o["cx"]), int(o["cy"])), int(round(o["r"])),
                           (0, 0, 255), 2)
            cv2.putText(vis, "hi=%.2f n=%d" % (hi, len(objs)), (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            panels.append(vis)
        cv2.imwrite(os.path.join(OUT, "sweep_hi_%d.png" % int(t)), np.hstack(panels))
    cap.release()
    print("saved sweep_hi_*.png")


if __name__ == "__main__":
    main()
