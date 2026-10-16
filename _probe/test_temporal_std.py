# -*- coding: utf-8 -*-
"""
用时间波动区分"真气泡"和"固定虚影"。

真气泡会随时间出现/长大/消失 -> 该像素灰度波动大。
固定的东西（氮化镓孔洞、光照不均、镜面反光）-> 波动小。
"""
import os
import sys
import numpy as np
import cv2

ROOT = r"E:\Bubble image recognition"
OUT = os.path.join(ROOT, "_probe")
W, H = 675, 540


def load(path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // (W * H)
    return raw[: n * W * H].reshape(n, H, W).astype(np.float32)


def main():
    raw_path = os.path.join(OUT, sys.argv[1]) if len(sys.argv) > 1 else os.path.join(OUT, "std.raw")
    if not os.path.exists(raw_path):
        print("缺少 %s" % raw_path)
        return
    A = load(raw_path)
    n = A.shape[0]
    print("帧数 = %d" % n)

    mv = A[0] > 45
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(mv.astype(np.uint8), 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    inside = cv2.erode((lab == k).astype(np.uint8), np.ones((21, 21), np.uint8))

    mean = A.mean(0)
    std = A.std(0)
    std[inside == 0] = 0
    print("std: 中位=%.2f  p90=%.2f  p99=%.2f  max=%.2f"
          % (np.median(std[inside > 0]), np.percentile(std[inside > 0], 90),
             np.percentile(std[inside > 0], 99), std.max()))

    # 归一化显示
    mu = inside > 0
    lo, hi = np.percentile(std[mu], 50), np.percentile(std[mu], 99.5)
    stdv = np.clip((std - lo) / max(1e-6, hi - lo), 0, 1)
    stdv[~mu] = 0
    heat = cv2.applyColorMap((stdv * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    meanv = cv2.cvtColor(mean.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    blend = cv2.addWeighted(meanv, 0.45, heat, 0.55, 0)

    # 右侧：只在"高波动"区域显示的图 -> 理论上就是真气泡的活动区
    thr = np.percentile(std[mu], 97)
    act = (std > thr).astype(np.uint8)
    act = cv2.morphologyEx(act, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    active = cv2.cvtColor(mean.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    active[act == 0] = (active[act == 0] * 0.25).astype(np.uint8)
    cnts, _ = cv2.findContours(act, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in cnts if cv2.contourArea(c) > 300]
    cv2.drawContours(active, big, -1, (0, 255, 0), 2)
    cv2.putText(active, "temporal-active regions: %d" % len(big), (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    panel = np.hstack([meanv, blend, active])
    cv2.putText(panel, "mean", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, "mean + temporal std", (W + 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(OUT, "temporal_std.png"), panel)

    # 统计：高波动区域占比
    print("高波动(>p97)像素占比 = %.1f%% (占视场)" % (100.0 * act.sum() / max(1, mu.sum())))
    print("saved temporal_std.png")


if __name__ == "__main__":
    main()
