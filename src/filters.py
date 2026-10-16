# -*- coding: utf-8 -*-
"""
特征滤波工具：积分图 + 盒子均值 + "中心核"响应。

依据人工标注量出来的实测径向剖面（相对背景的灰度差）：
  气泡:  中心 +1  -> 外圈 -13   （中心比外圈亮约 14 级）
  虚影:  中心 -10 -> 外圈 -4    （中心比外圈暗约 6 级）
所以 core(r) = mean(中心小盘 0.13r) - mean(外盘 0.60r)
对气泡为正、对虚影为负，是两者的可分特征。
"""
import numpy as np
import cv2


def integral(img):
    """积分图，用于 O(1) 算任意大小盒子的均值。"""
    return cv2.integral2(img.astype(np.float32))[0]


def boxmean(It, h):
    """半宽 h 的盒子均值；边界复制填充。"""
    H = It.shape[0] - 1
    W = It.shape[1] - 1
    h = max(1, int(round(h)))
    k = 2 * h + 1
    if k >= min(H, W):
        return np.full((H, W), float(It[H, W]) / (H * W), np.float32)
    S = It[k:, k:] - It[:-k, k:] - It[k:, :-k] + It[:-k, :-k]
    out = np.empty((H, W), np.float32)
    out[h:H - h, h:W - h] = S / float(k * k)
    out[:h, :] = out[h:h + 1, :]
    out[H - h:, :] = out[H - h - 1:H - h, :]
    out[:, :h] = out[:, h:h + 1]
    out[:, W - h:] = out[:, W - h - 1:W - h]
    return out


def core_score(img, r, It=None, core=0.13, outer=0.60):
    """中心核响应：中心小盘均值 - 外盘均值。"""
    It = integral(img) if It is None else It
    m1 = boxmean(It, max(1.0, core * r))
    m2 = boxmean(It, max(2.0, outer * r))
    return m1 - m2


def aperture_mask(gray, thresh=45, erode=15):
    """圆形视场掩膜。"""
    lit = (gray > thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lit, 8)
    if n <= 1:
        return np.ones_like(lit)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cnts, -1, 1, -1)
    if erode > 0:
        m = cv2.erode(m, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * (erode // 2) + 1,) * 2))
    return m


def build_radii(lo, hi, ratio=1.18):
    """几何级数的尺度阶梯。"""
    rs, r = [], float(lo)
    while r <= hi:
        rs.append(r)
        r *= ratio
    return rs
