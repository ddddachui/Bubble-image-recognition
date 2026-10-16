# -*- coding: utf-8 -*-
"""诊断：候选点与标注气泡的对应关系。"""
import os
import sys

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.build_dataset import read_window, candidates, VIDEO, SCALE, RADII   # noqa: E402
from src.filters import aperture_mask                                          # noqa: E402
from src.annotations import load_all                                           # noqa: E402


def main():
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    ann = load_all(os.path.join(ROOT, "标注工具"))
    a = ann[round(t, 1)]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig, hp, en = read_window(cap, fps, t)
    cap.release()
    mask = aperture_mask(orig)
    cand = candidates(en.astype(np.float32), mask, RADII)
    print("t=%.0fs  候选 %d 个，标注气泡 %d 个" % (t, len(cand), len(a["bubbles"])))
    print()
    print("每个标注气泡 -> 最近的候选点")
    print(" 标注x    y    r | 最近候选x    y     r   得分  距离  半径比")
    hit = 0
    for b in a["bubbles"]:
        bx, by, br = b["x"] * SCALE, b["y"] * SCALE, b["r"] * SCALE
        best, bc = 1e18, None
        for sc, x, y, r in cand:
            d = np.hypot(x - bx, y - by)
            if d < best:
                best, bc = d, (sc, x, y, r)
        if bc is None:
            print("  %4.0f %5.0f %4.0f | 无候选" % (b["x"], b["y"], b["r"]))
            continue
        ok = best < 0.45 * br and 0.4 < bc[3] / br < 2.2
        hit += ok
        print("  %4.0f %5.0f %4.0f | %8.0f %5.0f %5.1f %6.1f %5.1f  %4.2f  %s"
              % (b["x"], b["y"], b["r"], bc[1] / SCALE, bc[2] / SCALE, bc[3] / SCALE,
                 bc[0], best / SCALE, bc[3] / br, "OK" if ok else ""))
    print()
    print("匹配上 %d / %d" % (hit, len(a["bubbles"])))
    print()
    print("候选点半径分布 (半分辨率): %s" %
          np.round(np.percentile([c[3] for c in cand], [5, 25, 50, 75, 95]), 1))


if __name__ == "__main__":
    main()
