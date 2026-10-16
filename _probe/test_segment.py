# -*- coding: utf-8 -*-
"""测试 src/segment.py 在若干时刻的输出。"""
import os
import sys
import time
import numpy as np
import cv2

ROOT = r"E:\Bubble image recognition"
sys.path.insert(0, ROOT)
from src import segment as S                                     # noqa: E402

VIDEO = os.path.join(ROOT, "video1.mp4")
OUT = os.path.join(ROOT, "_probe")
SCALE = 0.5


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
        t0 = time.time()
        small = cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        m = S.aperture_mask(g)
        rel, fill, objs = S.detect(g, m)
        dt = (time.time() - t0) * 1000
        print("t=%6.1fs %6.0fms  fill=%6d  n=%2d  r=%s" %
              (t, dt, int(fill.sum()), len(objs),
               " ".join("%.0f" % o["r"] for o in sorted(objs, key=lambda z: -z["r"]))))
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        vis[fill > 0] = (0.6 * vis[fill > 0] + 0.4 * np.array([0, 160, 0])).astype(np.uint8)
        for j, o in enumerate(sorted(objs, key=lambda z: -z["r"])):
            cv2.circle(vis, (int(o["cx"]), int(o["cy"])), int(round(o["r"])), (0, 0, 255), 2)
            cv2.circle(vis, (int(o["cx"]), int(o["cy"])), 2, (255, 0, 255), -1)
            cv2.putText(vis, "%d" % j, (int(o["cx"]) + 4, int(o["cy"]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis, "t=%.1f n=%d" % (t, len(objs)), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        rows.append(np.hstack([vis, cv2.cvtColor(np.clip(rel * 700, 0, 255).astype(np.uint8),
                                                 cv2.COLOR_GRAY2BGR)]))
    if rows:
        cv2.imwrite(os.path.join(OUT, "test_segment.png"), np.vstack(rows))
        print("saved test_segment.png")
    cap.release()


if __name__ == "__main__":
    main()
