# -*- coding: utf-8 -*-
# 对比探针：判读图里的 90 个"噪声候选" vs 验证帧上命中的真气泡，
# 看时间维度（动态性）和局部对比度上有没有可分信号。
#
# 指标（都在检测分辨率/半分辨率上取）：
#   hp      时间高通值（中心）         —— 是否偏离慢漂移
#   tstd    时间标准差（±5s，中心）     —— 动态 = 气泡；静态 = 麻点/固定结构
#   tdiff   帧间差均值（±5s，中心）     —— 抖动/闪烁
#   en_mean 增强图局部均值（disk）
#   en_std  增强图局部对比度（disk）
#
# 输出：各组指标中位数/四分位 + 区分度（rank AUC），打印即可。
import csv
import os
import sys

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.build_dataset import SCALE, enhance             # noqa: E402
from src.annotations import load_all                        # noqa: E402

VIDEO = os.path.join(ROOT, "video1.mp4")
T_MAX = 48.0
HALF = 5.0


def read_maps(cap, fps, t):
    """返回半分辨率下的 orig/hp/en/tstd/tdiff 五张图。"""
    c = int(round(t * fps))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a = max(0, c - int(HALF * fps))
    b = min(n_total - 1, c + int(HALF * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    s = s2 = dsum = None
    prev = center = None
    cnt = 0
    for i in range(a, b + 1):
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, None, fx=SCALE, fy=SCALE,
                                    interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        if i == c:
            center = g.copy()
        if s is None:
            s, s2, dsum = g.copy(), g * g, np.zeros_like(g)
        else:
            s += g
            s2 += g * g
            dsum += np.abs(g - prev)
        prev = g
        cnt += 1
    if center is None or cnt < 5:
        return None
    mean = s / cnt
    tstd = np.sqrt(np.maximum(s2 / cnt - mean * mean, 0))
    tdiff = dsum / (cnt - 1)
    orig = center
    hp = orig - mean
    en = enhance(orig, hp)
    return orig, hp, en, tstd, tdiff


def disk_mean(mat, x, y, rd):
    H, W = mat.shape
    x0, x1 = max(0, x - rd), min(W, x + rd + 1)
    y0, y1 = max(0, y - rd), min(H, y + rd + 1)
    return float(mat[y0:y1, x0:x1].mean())


def main():
    ann = load_all(os.path.join(ROOT, "标注工具"))
    times = sorted(t for t in ann if t <= T_MAX)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cache = {}

    # 噪声组：judge_list.csv 里的 90 个
    noise = []
    with open(os.path.join(ROOT, "results_train", "judge", "judge_list.csv"),
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            noise.append((float(r["时间s"]), float(r["x"]), float(r["y"]),
                          float(r["半径(全分辨率px)"])))

    # 真气泡组：验证帧上标注的气泡中心
    val_t = (16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0)
    true = []
    for t in times:
        if t in val_t:
            for b in ann[t]["bubbles"]:
                true.append((t, b["x"], b["y"], b["r"]))

    def metrics_for(points, tag):
        rows = []
        by_t = {}
        for t, x, y, r in points:
            by_t.setdefault(t, []).append((x, y, r))
        for t, pts in sorted(by_t.items()):
            if t not in cache:
                cache[t] = read_maps(cap, fps, t)
            maps = cache[t]
            if maps is None:
                print("  !! t=%.0f 读取失败" % t)
                continue
            orig, hp, en, tstd, tdiff = maps
            for x, y, r in pts:
                xi = int(round(x * SCALE))
                yi = int(round(y * SCALE))
                xi = min(max(xi, 0), hp.shape[1] - 1)
                yi = min(max(yi, 0), hp.shape[0] - 1)
                rd = max(3, int(0.5 * r * SCALE))
                rows.append((t, x, y, r,
                             abs(float(hp[yi, xi])),
                             float(tstd[yi, xi]),
                             float(tdiff[yi, xi]),
                             disk_mean(en, xi, yi, rd),
                             float(en[max(0, yi - rd):yi + rd + 1,
                                       max(0, xi - rd):xi + rd + 1].std())))
            print("  t=%.0f %s 处理完（%d 个点）" % (t, tag, len(pts)), flush=True)
        return rows

    print("噪声组 %d 个点：" % len(noise))
    rn = metrics_for(noise, "噪声")
    print("真气泡组 %d 个点：" % len(true))
    rt = metrics_for(true, "真气泡")

    names = ["hp", "tstd", "tdiff", "en_mean", "en_std"]
    A = np.array([r[4:] for r in rn], float)
    B = np.array([r[4:] for r in rt], float)
    print()
    print("指标        噪声中位 [P25,P75]          真气泡中位 [P25,P75]        区分度AUC")
    for k, nm in enumerate(names):
        qa = np.percentile(A[:, k], [25, 50, 75])
        qb = np.percentile(B[:, k], [25, 50, 75])
        allv = np.concatenate([A[:, k], B[:, k]])
        rk = np.argsort(np.argsort(allv)) + 1.0
        n1, n0 = len(A), len(B)
        u = rk[:n1].sum() - n1 * (n1 + 1) / 2.0
        auc = u / (n1 * n0)          # 0.5 = 不可分；越接近 0 或 1 越可分
        print("%-8s  %6.1f [%6.1f,%6.1f]      %6.1f [%6.1f,%6.1f]      %.3f"
              % (nm, qa[1], qa[0], qa[2], qb[1], qb[0], qb[2], auc))

    m16 = np.array([r for r in rn if r[0] == 16.0])
    if len(m16):
        print()
        print("t=16（麻点帧）噪声组 %d 个：tstd 中位 %.2f  hp 中位 %.2f"
              % (len(m16), np.median(m16[:, 5]), np.median(m16[:, 4])))


if __name__ == "__main__":
    main()
