# -*- coding: utf-8 -*-
"""
环形匹配滤波器（直接用人工标注量出来的气泡特征）。

气泡的径向剖面：中心亮、外面一圈暗。虚影：中心最暗、往外变亮。
所以响应定义为：  score = mean(中心圆盘 0.35r) - mean(环带 0.35r~0.85r)
气泡 -> 明显正值；虚影 -> 负值；平坦背景 -> 0。

用积分图算盒子均值，任意尺度都是 O(1)/像素。
"""
import numpy as np
import cv2


def integral(img):
    return cv2.integral2(img.astype(np.float32))[0]


def boxmean(It, h):
    """半宽 h 的盒子均值，边界用复制填充。It 是积分图。"""
    H = It.shape[0] - 1
    W = It.shape[1] - 1
    h = max(1, int(round(h)))
    k = 2 * h + 1
    if k >= min(H, W):
        return np.full((H, W), float(It[H, W] / (H * W)), np.float32)
    S = It[k:, k:] - It[:-k, k:] - It[k:, :-k] + It[:-k, :-k]
    out = np.empty((H, W), np.float32)
    out[h:H - h, h:W - h] = S / float(k * k)
    out[:h, :] = out[h:h + 1, :]
    out[H - h:, :] = out[H - h - 1:H - h, :]
    out[:, :h] = out[:, h:h + 1]
    out[:, W - h:] = out[:, W - h - 1:W - h]
    return out


def ring_score(img, r, It=None, core=0.13, outer=0.60):
    """
    "中心核"响应：  score = mean(中心小盘 0.13r) - mean(外盘 0.60r)

    依据人工标注量出来的实测剖面（相对背景的灰度差，已按各自半径归一化）：
      气泡:  中心 +1  ->  外圈 -13   （中心比外圈亮 ~14 级）
      虚影:  中心 -10 ->  外圈 -4    （中心比外圈暗 ~6 级）
    所以"中心减外圈"这个量对气泡为正、对虚影为负，是两者的可分特征。
    """
    It = integral(img) if It is None else It
    m1 = boxmean(It, max(1.0, core * r))
    m2 = boxmean(It, max(2.0, outer * r))
    return m1 - m2


def detect(img, mask, radii, min_sep_frac=0.6, score_thr=2.0):
    """多尺度环形滤波 + 跨尺度非极大抑制。"""
    It = integral(img)
    cands = []
    for r in radii:
        s = ring_score(img, r, It)
        s[mask == 0] = 0.0
        sep = max(3, int(round(min_sep_frac * r)))
        mx = cv2.dilate(s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sep + 1,) * 2))
        pk = (s >= mx - 1e-6) & (s > score_thr)
        ys, xs = np.nonzero(pk)
        for y, x in zip(ys, xs):
            cands.append((float(s[y, x]), float(x), float(y), float(r)))
    cands.sort(reverse=True)
    keep = []
    for sc, x, y, r in cands:
        ok = True
        for _, kx, ky, kr in keep:
            if np.hypot(x - kx, y - ky) < 0.6 * (r + kr):
                ok = False
                break
        if ok:
            keep.append((sc, x, y, r))
    return keep
