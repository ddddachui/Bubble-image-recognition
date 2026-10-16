# -*- coding: utf-8 -*-
"""
用人工标注量出"气泡"和"虚影"的径向剖面差异。

用户给的判据：气泡有"中心核"（中心亮）且边界明显；虚影有黑度但没有中心核。
这里把两者的径向剖面统计出来，看判据在像素上到底长什么样、能不能量化。
"""
import json
import os
import sys

import numpy as np
import cv2

ROOT = r"E:\Bubble image recognition"
ANN = os.path.join(ROOT, "标注工具", "bubble_annotation.json")
FR = os.path.join(ROOT, "标注工具", "frames")
NB = 16                                  # 径向分箱数（u = 距离/r）


def imread_u(path):
    """cv2.imread 不支持中文路径。"""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def profile(img, x, y, r, us):
    """在以 (x,y) 为心、按 r 归一化的半径上采样平均灰度。"""
    out = np.zeros(len(us), np.float32)
    for i, u in enumerate(us):
        rad = u * r
        n = max(12, int(2 * np.pi * rad / 2.0))
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = np.clip(np.round(x + rad * np.cos(th)).astype(int), 0, img.shape[1] - 1)
        ys = np.clip(np.round(y + rad * np.sin(th)).astype(int), 0, img.shape[0] - 1)
        out[i] = img[ys, xs].mean()
    return out


def main():
    d = json.load(open(ANN, encoding="utf-8"))
    us = np.linspace(0.08, 1.6, NB)
    pb, pg = [], []
    cb, cg = [], []
    for a in d["annotated"]:
        img = imread_u(os.path.join(FR, a["id"] + "_orig.png"))
        if img is None:
            print("读不到", a["id"]); continue
        bg = float(np.median(img[img > 5]))
        for b in a["bubbles"]:
            p = profile(img, b["x"], b["y"], b["r"], us) - bg
            pb.append(p)
            core = img[int(b["y"]), int(b["x"])]
            ring = img[int(np.clip(b["y"], 0, 1079)),
                       int(np.clip(b["x"] + b["r"] * 0.9, 0, 1349))]
            cb.append(core - ring)
        for g in a["ghosts"]:
            # 虚影只标了中心点，没有半径；用 100 px 当参考尺度（气泡中位半径 78）
            p = profile(img, g["x"], g["y"], 100.0, us) - bg
            pg.append(p)
            core = img[int(g["y"]), int(g["x"])]
            ring = img[int(np.clip(g["y"], 0, 1079)),
                       int(np.clip(g["x"] + 90, 0, 1349))]
            cg.append(core - ring)

    pb = np.array(pb); pg = np.array(pg)
    cb = np.array(cb); cg = np.array(cg)
    print("气泡样本 %d, 虚影样本 %d" % (len(pb), len(pg)))
    print("\n径向剖面（相对背景的灰度差，已按各自半径归一化）")
    print("  u      " + " ".join("%6.2f" % u for u in us))
    print("  气泡   " + " ".join("%6.1f" % v for v in pb.mean(0)))
    print("  虚影   " + " ".join("%6.1f" % v for v in pg.mean(0)))
    print("\n  气泡中位 " + " ".join("%6.1f" % v for v in np.median(pb, 0)))
    print("  虚影中位 " + " ".join("%6.1f" % v for v in np.median(pg, 0)))

    print("\n中心核反差 = I(中心) - I(0.9r处)")
    print("  气泡: 中位=%.1f  四分位=[%.1f, %.1f]  正比例=%.0f%%"
          % (np.median(cb), np.percentile(cb, 25), np.percentile(cb, 75),
             100.0 * (cb > 0).mean()))
    print("  虚影: 中位=%.1f  四分位=[%.1f, %.1f]  正比例=%.0f%%"
          % (np.median(cg), np.percentile(cg, 25), np.percentile(cg, 75),
             100.0 * (cg > 0).mean()))

    # 用两个简单特征做可分性检查
    def core_minus_ring(p):
        core = p[:4].mean()
        ann = p[-6:].mean()
        return core - ann
    fb = np.array([core_minus_ring(p) for p in pb])
    fg = np.array([core_minus_ring(p) for p in pg])
    thr = (np.median(fb) + np.median(fg)) / 2.0
    acc = ((fb > thr).mean() + (fg <= thr).mean()) / 2.0
    print("\n特征 core-annulus:  气泡 中位=%.1f   虚影 中位=%.1f" % (np.median(fb), np.median(fg)))
    print("  取阈值 %.1f 时，二分类准确率 ≈ %.0f%%" % (thr, 100 * acc))


if __name__ == "__main__":
    main()
