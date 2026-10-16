# -*- coding: utf-8 -*-
"""
气泡分割 / 检测模块（M0 传统方法）

流程：
  1. 圆形视场掩膜（黑角 + 光圈边界）
  2. 中值滤波估计局部背景（对强镜面反光稳健）
  3. 相对暗度响应 rel = (bg - I) / bg   —— 气泡比局部背景暗
  4. 滞回阈值：强核心 + 弱范围 -> 连通域筛选 -> 闭运算补形 -> 填洞
  5. 距离变换 + 局部极大 -> 每个气泡一个"中心 + 等效半径"

说明：本模块把每个气泡近似成一个圆盘（中心 + 半径）。
对近圆形气泡足够用，且比分水岭快得多、稳定得多。
"""
import numpy as np
import cv2


def disk(r):
    """生成圆结构元，r 为半径（像素）。"""
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def aperture_mask(gray, thresh=45, erode=21):
    """圆形视场掩膜：非黑区域的最大连通域，填洞后向内腐蚀若干像素。"""
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
        m = cv2.erode(m, disk(erode // 2), iterations=1)
    return m


def rel_response(gray, mask, med_k=151, glint_pct=99.7, glint_dilate=11):
    """
    相对暗度响应：正值表示比局部背景暗（气泡）。

    强镜面反光（星形高亮）会把它周围"显得"很暗，形成大片假响应，
    所以先把反光区填掉再估计背景，最后把反光区本身置零。
    """
    g = gray
    glint = np.zeros_like(gray, dtype=np.uint8)
    vals = gray[mask > 0]
    if vals.size:
        gthr = np.percentile(vals, glint_pct)
        glint = (gray >= gthr).astype(np.uint8)
        glint = cv2.dilate(glint, disk(max(1, glint_dilate // 2)))
        if glint.sum() > 0:
            med = cv2.medianBlur(gray, 51)
            g = gray.copy()
            g[glint > 0] = med[glint > 0]
    bg = cv2.medianBlur(g, med_k if med_k % 2 else med_k + 1).astype(np.float32)
    rel = (bg - gray.astype(np.float32)) / (bg + 1e-6)
    rel[mask == 0] = 0.0
    rel[glint > 0] = 0.0
    return rel


def blob_mask(rel, mask, lo=0.030, hi=0.100, open_r=6, close_r=17,
              min_strong=25, min_weak_area=200):
    """滞回阈值得到气泡区域（弱范围里必须含强核心，避免把背景纹理算进来）。"""
    strong = (rel > hi).astype(np.uint8)
    weak = (rel > lo).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_OPEN, disk(open_r))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(weak, 8)
    keep = np.zeros_like(weak)
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_weak_area:
            continue
        sel = (lab == i)
        if int(strong[sel].sum()) >= min_strong:
            keep[sel] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, disk(close_r))
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fill = np.zeros_like(keep)
    cv2.drawContours(fill, cnts, -1, 1, -1)
    fill[mask == 0] = 0
    return fill


def distance_map(fill):
    """气泡区域的距离变换（平滑后），后续既用于找中心，也用于量"颈部"。"""
    dist = cv2.distanceTransform(fill, cv2.DIST_L2, 5)
    return cv2.GaussianBlur(dist, (0, 0), 2.0)


def _neck(dsm, x1, y1, x2, y2, n=41):
    """沿两点连线采样距离图，返回最小值（= 两气泡之间的"颈部宽度"）。"""
    h, w = dsm.shape
    ts = np.linspace(0.0, 1.0, n)
    xs = np.clip(np.round(x1 + (x2 - x1) * ts).astype(int), 0, w - 1)
    ys = np.clip(np.round(y1 + (y2 - y1) * ts).astype(int), 0, h - 1)
    return float(dsm[ys, xs].min())


def find_bubbles(fill, dsm=None, min_r=9.0, min_sep=13, prominence=0.72,
                 max_r=70.0, merge_ratio=0.80):
    """
    距离变换 + 局部极大 -> 气泡中心与半径。

    min_r      最小可检半径（像素）
    min_sep    两个中心的最小间距
    prominence 峰值至少达到"同一粘连簇内最大半径"的比例，用来抑制假峰
    max_r      最大可接受半径（超过则判为大片伪影/未拆分的大团块）
    """
    if int(fill.sum()) == 0:
        return []
    if dsm is None:
        dsm = distance_map(fill)
    mx = cv2.dilate(dsm, disk(min_sep))
    pk = ((dsm >= mx - 1e-6) & (dsm > min_r)).astype(np.uint8)
    if int(pk.sum()) == 0:
        return []
    n, lab, stats, cent = cv2.connectedComponentsWithStats(pk, 8)
    nfill, flab, fstats, _ = cv2.connectedComponentsWithStats(fill, 8)
    cluster_max = {}
    for i in range(1, nfill):
        sel = flab == i
        cluster_max[i] = float(dsm[sel].max()) if sel.any() else 0.0
    cand = []
    for i in range(1, n):
        cx, cy = float(cent[i][0]), float(cent[i][1])
        r = float(dsm[int(round(cy)), int(round(cx))])
        cid = int(flab[int(round(cy)), int(round(cx))])
        if cid == 0:
            continue
        if r > max_r:
            continue
        if r < prominence * cluster_max.get(cid, r):
            continue
        cand.append(dict(cx=cx, cy=cy, r=r, area=float(np.pi * r * r)))

    # 去重：两个候选峰若"颈部"已经被填平（neck/min(r) > merge_ratio），
    # 说明它们其实是同一个气泡上的两个假峰，保留较大者。
    cand.sort(key=lambda d: -d["r"])
    out = []
    for c in cand:
        dup = False
        for k in out:
            dist = float(np.hypot(c["cx"] - k["cx"], c["cy"] - k["cy"]))
            if dist > 1.4 * (c["r"] + k["r"]):
                continue
            nk = _neck(dsm, c["cx"], c["cy"], k["cx"], k["cy"])
            if nk > merge_ratio * min(c["r"], k["r"]):
                dup = True
                break
        if not dup:
            out.append(c)
    return out


def detect(gray, mask, cfg=None):
    """灰度图 + 掩膜 -> (响应图, 气泡区域掩膜, 距离图, 气泡列表)。"""
    cfg = cfg or {}
    rel = rel_response(gray, mask, med_k=cfg.get("med_k", 151))
    fill = blob_mask(rel, mask,
                     lo=cfg.get("lo", 0.030), hi=cfg.get("hi", 0.100),
                     open_r=cfg.get("open_r", 6), close_r=cfg.get("close_r", 17))
    dsm = distance_map(fill)
    objs = find_bubbles(fill, dsm,
                        min_r=cfg.get("min_r", 9.0),
                        min_sep=cfg.get("min_sep", 13),
                        prominence=cfg.get("prominence", 0.72),
                        max_r=cfg.get("max_r", 70.0),
                        merge_ratio=cfg.get("merge_ratio", 0.80))
    return rel, fill, dsm, objs
