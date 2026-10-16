# -*- coding: utf-8 -*-
"""决定性诊断：把慢漂移减掉后的"残差"放大显示，看真正在变的东西长什么样。"""
import os
import sys
import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d

ROOT = r"E:\Bubble image recognition"
OUT = os.path.join(ROOT, "_probe")
W, H = 675, 540


def main():
    path = os.path.join(OUT, "std.raw")     # 40-100s @5fps
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // (W * H)
    A = raw[: n * W * H].reshape(n, H, W).astype(np.float32)
    slow = uniform_filter1d(A, size=25, axis=0, mode="nearest")
    hp = A - slow

    nlab, lab, stats, _ = cv2.connectedComponentsWithStats((A[0] > 45).astype(np.uint8), 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    inside = (lab == k)

    picks = [0, 40, 80, 120, 160, 220]      # 对应 40, 48, 56, 64, 72, 84 秒
    rows = []
    for i in picks:
        g = A[i].copy()
        r = hp[i].copy()
        r[~inside] = 0
        lo, hi = np.percentile(r[inside], 2), np.percentile(r[inside], 98)
        rv = np.clip((r - lo) / max(1e-6, hi - lo), 0, 1)
        rv[~inside] = 0
        rv = (rv * 255).astype(np.uint8)
        a = cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        b = cv2.cvtColor(rv, cv2.COLOR_GRAY2BGR)
        t = 40 + i / 5.0
        for img, tag in ((a, "frame %.1fs" % t), (b, "residual %.1fs" % t)):
            cv2.putText(img, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows.append(np.hstack([a, b]))
    cv2.imwrite(os.path.join(OUT, "residual.png"), np.vstack(rows))
    print("saved residual.png  (上:原图 下:去掉慢漂移后的残差)")


if __name__ == "__main__":
    main()
