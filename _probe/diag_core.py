# -*- coding: utf-8 -*-
"""诊断：在人工标注的气泡中心上，中心核响应到底是多少？"""
import os
import sys

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.build_dataset import read_window, VIDEO, SCALE      # noqa: E402
from src.filters import core_score, integral, boxmean          # noqa: E402
from src.annotations import load_all                           # noqa: E402


def main():
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    ann = load_all(os.path.join(ROOT, "标注工具"))
    a = ann[round(t, 1)]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig, hp, en = read_window(cap, fps, t)
    cap.release()
    enf = en.astype(np.float32)
    orf = orig.astype(np.float32)
    It_e, It_o = integral(enf), integral(orf)
    H, W = enf.shape

    print("t=%.0fs  标注气泡 %d 个，虚影 %d 个" % (t, len(a["bubbles"]), len(a["ghosts"])))
    print()
    print(" 标注x    y    r(全)   r(半) | core(增强) 最佳尺度 最-佳值 | core(原图) | 中心灰度 外圈灰度")
    for b in a["bubbles"][:20]:
        x, y, r = b["x"] * SCALE, b["y"] * SCALE, b["r"] * SCALE
        xi, yi = int(round(x)), int(round(y))
        xi = min(max(xi, 0), W - 1); yi = min(max(yi, 0), H - 1)
        s_at = float(core_score(enf, r, It_e)[yi, xi])
        best, br = -1e9, 0.0
        for k in np.arange(0.3, 3.01, 0.1):
            v = float(core_score(enf, r * k, It_e)[yi, xi])
            if v > best:
                best, br = v, r * k
        s_o = float(core_score(orf, r, It_o)[yi, xi])
        c_g = float(boxmean(It_e, max(1, 0.13 * r))[yi, xi])
        r_g = float(boxmean(It_e, max(2, 0.60 * r))[yi, xi])
        print("  %4.0f %5.0f %7.0f %6.1f | %8.2f %8.1f %8.2f | %8.2f | %8.1f %8.1f"
              % (b["x"], b["y"], b["r"], r, s_at, br, best, s_o, c_g, r_g))

    print()
    print(" 虚影点:")
    for g in a["ghosts"][:10]:
        x, y = g["x"] * SCALE, g["y"] * SCALE
        xi = min(max(int(round(x)), 0), W - 1)
        yi = min(max(int(round(y)), 0), H - 1)
        best, br = -1e9, 0.0
        for k in np.arange(0.3, 3.01, 0.1):
            v = float(core_score(enf, 60 * k, It_e)[yi, xi])
            if v > best:
                best, br = v, 60 * k
        print("  (%4.0f,%5.0f)  最大 core = %6.2f @ r=%.0f" % (g["x"], g["y"], best, br))


if __name__ == "__main__":
    main()
