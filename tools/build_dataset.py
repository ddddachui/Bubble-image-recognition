# -*- coding: utf-8 -*-
"""
从人工标注生成"候选点 + 特征"训练集。

对每个标注帧：
  1. 从视频读前后各 5 秒的帧，用"窗口均值"当慢漂移，减掉得到时间高通图
     （光照会慢慢漂，气泡不会；减掉慢漂移对比度大幅提升）
  2. 把高通细节叠回原图 -> 增强图（和标注工具里显示的是同一种图）
  3. 在增强图上做多尺度"中心核"响应，生成候选点（故意放宽，宁多勿漏）
  4. 每个候选点提特征
  5. 落在标注气泡内的 = 正样本，其余 = 负样本

输出: results_train/dataset.npz
"""
import os
import sys
import time

import numpy as np
import cv2
from scipy.ndimage import maximum_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.filters import integral, boxmean, core_score, aperture_mask, build_radii  # noqa: E402
from src.annotations import load_all                                                # noqa: E402

VIDEO = os.path.join(ROOT, "video1.mp4")
OUTDIR = os.path.join(ROOT, "results_train")
SCALE = 0.5
HP_HALF = 5.0            # 时间高通读前后各多少秒（要远大于气泡变化的时标）
RADII = build_radii(9, 200, 1.18)

NAMES = ["r", "log_r", "core@0.5", "core@0.7", "core@1.0", "core@1.4", "core@2.0",
         "orig_core", "orig_ring", "orig_core-ring", "enh_core", "enh_ring",
         "enh_core-ring", "enh_std", "orig_std", "edge", "gray_at_c", "enh_at_c"]


def enhance(orig, hp):
    """把时间高通细节叠回原图，再做局部对比增强（与标注工具里一致）。"""
    lo, hi = np.percentile(hp, 3), np.percentile(hp, 99)
    en = orig + np.clip(hp - (lo + hi) / 2.0, -60, 60) * 1.6
    en = np.clip(en, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(en)


def read_window(cap, fps, t):
    """
    读 t 前后各 HP_HALF 秒的帧，返回 (原始中心帧, 时间高通图, 增强图)，半分辨率。
    只需要窗口均值，所以直接用 sum/len，比跑一遍滑动平均快很多。
    """
    c = int(round(t * fps))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a = max(0, c - int(HP_HALF * fps))
    b = min(n_total - 1, c + int(HP_HALF * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    frames, center = [], None
    for i in range(a, b + 1):
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, None, fx=SCALE, fy=SCALE,
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if i == c:
            center = len(frames)
        frames.append(g)
    if not frames or center is None:
        return None, None, None
    stack = np.stack(frames)                       # uint8 (N,H,W)，省内存
    orig = stack[center].astype(np.float32)
    slow = stack.sum(axis=0, dtype=np.float32) / float(len(frames))
    hp = orig - slow
    return orig, hp, enhance(orig, hp)


def read_window2(cap, fps, t):
    """
    同 read_window，但另外返回两张"动态性"图：
      tstd  —— 窗口内每个像素的时间标准差（气泡会动 -> 大；固定结构 -> 小）
      tdiff —— 相邻帧差均值（闪烁/抖动程度）
    用于给候选块 CNN 提供"这个点在时间上像不像气泡"的信息。
    """
    c = int(round(t * fps))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a = max(0, c - int(HP_HALF * fps))
    b = min(n_total - 1, c + int(HP_HALF * fps))
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
        return None, None, None, None, None
    mean = s / cnt
    tstd = np.sqrt(np.maximum(s2 / cnt - mean * mean, 0))
    tdiff = dsum / (cnt - 1)
    orig = center
    hp = orig - mean
    return orig, hp, enhance(orig, hp), tstd, tdiff


def candidates(img, mask, radii, thr=0.5, min_sep=9, cap=3000, scale_norm=1.0):
    """
    多尺度中心核响应 -> 候选点 (score, x, y, r)。
    先在每像素取"跨所有尺度的最大值"及对应半径，再做一次固定大小的非极大抑制。

    scale_norm: 尺度归一化指数。匹配滤波理论下，均值之差的噪声标准差 ∝ 1/r，
                所以乘以 r^scale_norm 才能让不同尺度的分数可比，
                否则小尺度的纹理响应会把大气泡的峰压掉。
    """
    It = integral(img)
    best = np.full(img.shape, -1e9, np.float32)
    bestr = np.zeros(img.shape, np.float32)
    for r in radii:
        s = core_score(img, r, It) * (r ** scale_norm)
        s[mask == 0] = -1e9
        upd = s > best
        best[upd] = s[upd]
        bestr[upd] = r
    best[mask == 0] = -1e9
    mx = maximum_filter(best, size=2 * min_sep + 1)
    pk = (best >= mx - 1e-6) & (best > thr)
    ys, xs = np.nonzero(pk)
    if ys.size == 0:
        return []
    order = np.argsort(-best[ys, xs])[:cap]
    ys, xs = ys[order], xs[order]
    keep = []
    for y, x in zip(ys, xs):
        s = float(best[y, x]); r = float(bestr[y, x])
        ok = True
        for _, kx, ky, kr in keep:
            if (x - kx) ** 2 + (y - ky) ** 2 < (0.5 * (r + kr)) ** 2:
                ok = False
                break
        if ok:
            keep.append((s, float(x), float(y), r))
    return keep


def features(orig, hp, en, x, y, r):
    """候选点的特征向量。"""
    It_o, It_h, It_e = integral(orig), integral(hp), integral(en)
    H, W = orig.shape
    xi, yi = int(round(x)), int(round(y))

    def at(mat, xx, yy):
        return float(mat[int(np.clip(yy, 0, H - 1)), int(np.clip(xx, 0, W - 1))])

    f_core = [at(core_score(en, r * k, It_e), xi, yi) for k in (0.5, 0.7, 1.0, 1.4, 2.0)]
    o_c = at(boxmean(It_o, max(1, 0.13 * r)), xi, yi)
    o_r = at(boxmean(It_o, max(2, 0.60 * r)), xi, yi)
    e_c = at(boxmean(It_e, max(1, 0.13 * r)), xi, yi)
    e_r = at(boxmean(It_e, max(2, 0.60 * r)), xi, yi)
    h_c = at(boxmean(It_h, max(1, 0.13 * r)), xi, yi)

    h2 = max(2, int(1.5 * r))
    y0, y1 = max(0, yi - h2), min(H, yi + h2 + 1)
    x0, x1 = max(0, xi - h2), min(W, xi + h2 + 1)
    patch = en[y0:y1, x0:x1]
    pstd = float(patch.std()) if patch.size else 0.0
    ostd = float(orig[y0:y1, x0:x1].std()) if patch.size else 0.0

    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    px = np.clip(np.round(xi + r * np.cos(th)).astype(int), 0, W - 1)
    py = np.clip(np.round(yi + r * np.sin(th)).astype(int), 0, H - 1)
    gx = cv2.Sobel(en, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(en, cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.mean(np.hypot(gx[py, px], gy[py, px])))

    return np.array([r, np.log1p(r)] + f_core +
                    [o_c, o_r, o_c - o_r, e_c, e_r, e_c - e_r,
                     pstd, ostd, edge, h_c, at(en, xi, yi)], np.float32)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ann = load_all(os.path.join(ROOT, "标注工具"))
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("标注帧 %d，尺度 %d 档 (%.0f..%.0f 半分辨率)"
          % (len(ann), len(RADII), RADII[0], RADII[-1]), flush=True)

    X, y, R, info = [], [], [], []
    t0 = time.time()
    for t, a in sorted(ann.items()):
        orig, hp, en = read_window(cap, fps, t)
        if orig is None:
            print("  跳过 t=%.0f" % t, flush=True)
            continue
        mask = aperture_mask(orig)
        cand = candidates(en.astype(np.float32), mask, RADII)
        bubs = [(b["x"] * SCALE, b["y"] * SCALE, b["r"] * SCALE) for b in a["bubbles"]]
        npos = 0
        for sc, x, yc, r in cand:
            lab = 0
            for bx, by, br in bubs:
                if np.hypot(x - bx, yc - by) < 0.45 * br and 0.4 < r / br < 2.2:
                    lab = 1
                    break
            X.append(features(orig, hp, en, x, yc, r))
            y.append(lab)
            R.append(r)
            info.append((t, x / SCALE, yc / SCALE, r / SCALE))
            npos += lab
        print("  t=%5.1fs  候选 %4d  正 %3d / 标注 %3d"
              % (t, len(cand), npos, len(bubs)), flush=True)

    X = np.stack(X)
    y = np.array(y, np.int8)
    R = np.array(R, np.float32)
    np.savez_compressed(os.path.join(OUTDIR, "dataset.npz"),
                        X=X, y=y, r=R, info=np.array(info, np.float32),
                        names=np.array(NAMES))
    print("\n共 %d 个候选，正样本 %d (%.1f%%)，用时 %.0fs"
          % (len(y), int(y.sum()), 100.0 * y.mean(), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
