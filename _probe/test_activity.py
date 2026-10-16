# -*- coding: utf-8 -*-
"""
把逐像素时间序列做高通（减掉慢漂移），剩下的就是"气泡活动"。
慢漂移 = 光照变化、相机自动增益；气泡 = 出现/长大/消失，属于较快的变化。
"""
import os
import sys
import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d

ROOT = r"E:\Bubble image recognition"
OUT = os.path.join(ROOT, "_probe")
W, H = 675, 540


def main():
    path = os.path.join(OUT, sys.argv[1] if len(sys.argv) > 1 else "std.raw")
    win = int(sys.argv[2]) if len(sys.argv) > 2 else 40      # 慢漂移窗长（帧）
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // (W * H)
    A = raw[: n * W * H].reshape(n, H, W).astype(np.float32)
    print("帧数=%d, 高通窗长=%d 帧" % (n, win))

    nlab, lab, stats, _ = cv2.connectedComponentsWithStats((A[0] > 45).astype(np.uint8), 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    inside = cv2.erode((lab == k).astype(np.uint8), np.ones((21, 21), np.uint8))
    mu = inside > 0

    slow = uniform_filter1d(A, size=win, axis=0, mode="nearest")
    hp = A - slow
    act = hp.std(0)
    act[~mu] = 0
    print("高通后活动度: 中位=%.2f p90=%.2f p99=%.2f max=%.2f"
          % (np.median(act[mu]), np.percentile(act[mu], 90),
             np.percentile(act[mu], 99), act.max()))

    lo, hi = np.percentile(act[mu], 60), np.percentile(act[mu], 99.7)
    v = np.clip((act - lo) / max(1e-6, hi - lo), 0, 1)
    v[~mu] = 0
    heat = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    meanv = cv2.cvtColor(A.mean(0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    blend = cv2.addWeighted(meanv, 0.5, heat, 0.5, 0)

    thr = np.percentile(act[mu], 96)
    m = (act > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask3 = cv2.cvtColor(A.mean(0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask3[m == 0] = (mask3[m == 0] * 0.2).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in cnts if cv2.contourArea(c) > 400]
    cv2.drawContours(mask3, big, -1, (0, 255, 0), 2)
    areas = sorted([int(cv2.contourArea(c)) for c in big], reverse=True)
    print("高活动区: %d 块, 面积(半分辨率px)=%s" % (len(big), areas[:15]))
    for c in big:
        (x, y), r = cv2.minEnclosingCircle(c)
        cv2.circle(mask3, (int(x), int(y)), int(r), (255, 0, 255), 1)
    cv2.putText(mask3, "high-activity blobs: %d" % len(big), (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    panel = np.hstack([meanv, blend, mask3])
    for i, s in enumerate(("temporal mean", "mean + high-pass activity", "high-activity blobs")):
        cv2.putText(panel, s, (i * W + 10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(OUT, "activity.png"), panel)
    print("saved activity.png")


if __name__ == "__main__":
    main()
