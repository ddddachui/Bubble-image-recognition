# -*- coding: utf-8 -*-
"""扫描弱阈值 lo + 开运算半径，观察误检数量。"""
import os
import sys
import cv2

ROOT = r"E:\Bubble image recognition"
sys.path.insert(0, ROOT)
from src import segment as S                                    # noqa: E402

VIDEO = os.path.join(ROOT, "video1.mp4")


def frame_at(t):
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * cap.get(cv2.CAP_PROP_FPS))))
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def main():
    times = [float(x) for x in sys.argv[1:]] or [53.0]
    for t in times:
        f = frame_at(t)
        if f is None:
            print("t=%.0f 读取失败" % t)
            continue
        s = cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        m = S.aperture_mask(g)
        rel = S.rel_response(g, m)
        cells = []
        for lo, opr in ((0.030, 6), (0.045, 9), (0.060, 11), (0.080, 13)):
            fill = S.blob_mask(rel, m, lo=lo, hi=0.14, open_r=opr, close_r=17)
            dsm = S.distance_map(fill)
            o = S.find_bubbles(fill, dsm, min_r=17.0, min_sep=15,
                               prominence=0.70, max_r=62.0, merge_ratio=0.88)
            cells.append("lo=%.3f open=%2d mask=%6d n=%2d" % (lo, opr, int(fill.sum()), len(o)))
        print("t=%.0fs  %s" % (t, " | ".join(cells)))


if __name__ == "__main__":
    main()
