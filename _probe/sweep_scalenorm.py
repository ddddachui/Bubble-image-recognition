# -*- coding: utf-8 -*-
"""扫描尺度归一化指数，看候选点匹配率。"""
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
    ann = load_all(os.path.join(ROOT, "标注工具"))
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    times = [20.0, 30.0, 40.0]
    cache = {}
    for t in times:
        o, h, e = read_window(cap, fps, t)
        cache[t] = (aperture_mask(o), e.astype(np.float32))
    cap.release()
    for sn in (0.0, 0.5, 1.0, 1.5, 2.0):
        line = []
        tot_hit = tot = 0
        for t in times:
            mask, enf = cache[t]
            cand = candidates(enf, mask, RADII, scale_norm=sn)
            bubs = [(b["x"] * SCALE, b["y"] * SCALE, b["r"] * SCALE) for b in ann[t]["bubbles"]]
            hit = 0
            for bx, by, br in bubs:
                for sc, x, y, r in cand:
                    if np.hypot(x - bx, y - by) < 0.45 * br and 0.4 < r / br < 2.2:
                        hit += 1
                        break
            tot_hit += hit
            tot += len(bubs)
            line.append("t%.0f: %2d/%-2d (%3d候选)" % (t, hit, len(bubs), len(cand)))
        print("归一化指数 %.1f | %s | 合计 %d/%d = %.0f%%"
              % (sn, "  ".join(line), tot_hit, tot, 100.0 * tot_hit / tot))


if __name__ == "__main__":
    main()
