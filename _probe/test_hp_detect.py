# -*- coding: utf-8 -*-
"""在"时间高通残差"上做检测，看是否比在原图上检测干净得多。"""
import os
import numpy as np
import cv2
import trackpy as tp
from scipy.ndimage import uniform_filter1d

ROOT = r"E:\Bubble image recognition"
OUT = os.path.join(ROOT, "_probe")
W, H = 675, 540


def prep(A, win=25):
    slow = uniform_filter1d(A, size=win, axis=0, mode="nearest")
    hp = A - slow                                   # 时间高通
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats((A[0] > 45).astype(np.uint8), 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    inside = (lab == k)
    hp[:, ~inside] = 0
    return hp, inside


def show(gray, name, diam=61, minmass=None, tag=""):
    f = tp.locate(gray.astype(np.uint8), diameter=diam, invert=True,
                  minmass=minmass, separation=int(diam * 0.85),
                  characterize=True, percentile=64)
    m = f["mass"].values if len(f) else np.array([])
    print("  %-22s %2d 个   mass %s" % (tag, len(f),
          np.round(np.percentile(m, [10, 50, 90]), 0) if len(m) else "-"))
    return f


def main():
    raw = np.fromfile(os.path.join(OUT, "std.raw"), dtype=np.uint8)
    n = raw.size // (W * H)
    A = raw[: n * W * H].reshape(n, H, W).astype(np.float32)
    hp, inside = prep(A)
    print("时间高通残差已生成")
    picks = [60, 80, 120, 160]                     # 52, 56, 64, 72 秒
    rows = []
    for i in picks:
        t = 40 + i / 5.0
        print("t=%.1f s" % t)
        g0 = A[i].copy()
        g0[~inside] = int(np.median(A[i][inside]))
        g1 = hp[i].copy()
        g1 = 128 + g1 * 2.0                        # 放大对比度便于 locate
        g1 = np.clip(g1, 0, 255)
        g1[~inside] = 128
        show(g0, None, tag="原图")
        show(g1, None, tag="时间高通残差")

        # 叠加图（用残差结果）
        f = tp.locate(g1.astype(np.uint8), diameter=61, invert=True, minmass=None,
                      separation=52, characterize=True, percentile=64)
        vis = cv2.cvtColor(np.clip(128 + (A[i] - A[i].mean()) * 2.2, 0, 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)
        vis[~inside] = (vis[~inside] * 0.15).astype(np.uint8)
        for _, r in f.iterrows():
            cv2.circle(vis, (int(r.x), int(r.y)), int(r["size"] * 2.2), (0, 0, 255), 2)
            cv2.circle(vis, (int(r.x), int(r.y)), 2, (255, 0, 255), -1)
        cv2.putText(vis, "t=%.1fs  n=%d" % (t, len(f)), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2)
        rows.append(vis)
    cv2.imwrite(os.path.join(OUT, "hp_detect.png"), np.vstack([np.hstack(rows[:2]),
                                                              np.hstack(rows[2:])]))
    print("saved hp_detect.png")


if __name__ == "__main__":
    main()
