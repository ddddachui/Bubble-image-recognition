# -*- coding: utf-8 -*-
"""
逐像素训练集：不再"给每个像素选一个尺度"（那个做法被小尺度纹理压垮了），
而是给每个像素算出**跨所有尺度的中心核响应向量**，让分类器自己学
"什么样的多尺度响应组合 = 一个真气泡的中心"。

标签：落在标注气泡中心附近 = 正；远离所有气泡 = 负（离气泡边缘太近的丢弃，
避免"到底算不算"的模糊样本，尤其是虚影附近那些。）

输出: results_train/pixel_dataset.npz
"""
import os
import sys
import time

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.build_dataset import read_window, VIDEO, SCALE          # noqa: E402
from src.filters import integral, boxmean, core_score, aperture_mask  # noqa: E402
from src.annotations import load_all                                # noqa: E402

OUTDIR = os.path.join(ROOT, "results_train")
SCALES = [9, 12, 15, 19, 24, 30, 38, 48, 60, 76, 95, 120, 150]
NEG_PER_FRAME = 3000

NAMES = (["core@%d" % s for s in SCALES] +
         ["core_max", "core_argmax", "orig_gray", "enh_gray",
          "enh_std30", "enh_std60", "edge30"])


def feats_all(orig, en):
    """返回 (特征立方体 (H,W,F), 每像素最优尺度)。"""
    It_e = integral(en)
    It_o = integral(orig)
    stack = []
    best = np.full(en.shape, -1e9, np.float32)
    bestr = np.zeros(en.shape, np.float32)
    for i, r in enumerate(SCALES):
        s = core_score(en, r, It_e)
        stack.append(s)
        upd = s > best
        best[upd] = s[upd]
        bestr[upd] = i
    arr = np.stack(stack, axis=-1)                       # (H,W,S)
    o_g = orig
    e_g = en.astype(np.float32)
    e_std30 = np.sqrt(np.maximum(boxmean(integral(en ** 2), 30) -
                                 boxmean(It_e, 30) ** 2, 0))
    e_std60 = np.sqrt(np.maximum(boxmean(integral(en ** 2), 60) -
                                 boxmean(It_e, 60) ** 2, 0))
    gx = cv2.Sobel(en, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(en, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 5)
    F = np.concatenate([arr, best[..., None], bestr[..., None],
                        o_g[..., None], e_g[..., None],
                        e_std30[..., None], e_std60[..., None],
                        gmag[..., None]], axis=-1)
    return F, SCALES


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ann = load_all(os.path.join(ROOT, "标注工具"))
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rng = np.random.default_rng(0)
    Xs, ys, fps_id, rads, poss = [], [], [], [], []
    t0 = time.time()
    for fi, (t, a) in enumerate(sorted(ann.items())):
        orig, hp, en = read_window(cap, fps, t)
        if orig is None:
            continue
        mask = aperture_mask(orig)
        F, _ = feats_all(orig, en)
        H, W = orig.shape
        bubs = [(b["x"] * SCALE, b["y"] * SCALE, b["r"] * SCALE) for b in a["bubbles"]]
        ghosts = [(g["x"] * SCALE, g["y"] * SCALE) for g in a["ghosts"]]

        yy, xx = np.mgrid[0:H, 0:W]
        # 到最近气泡中心的距离（按各自半径归一化）
        dnorm = np.full((H, W), 1e9, np.float32)
        rnear = np.full((H, W), 1.0, np.float32)
        inside = np.zeros((H, W), bool)
        for bx, by, br in bubs:
            d = np.hypot(xx - bx, yy - by) / max(1.0, br)
            inside |= (d < 0.30)
            sel = d < dnorm
            dnorm[sel] = d[sel]
            rnear[sel] = br
        valid = (mask > 0)
        pos_idx = np.flatnonzero((inside & valid).ravel())
        if pos_idx.size > 400:
            pos_idx = rng.choice(pos_idx, 400, replace=False)
        neg_pool = np.flatnonzero((valid & (dnorm > 0.75)).ravel())
        neg_idx = (rng.choice(neg_pool, NEG_PER_FRAME, replace=False)
                   if neg_pool.size > NEG_PER_FRAME else neg_pool)
        idx = np.concatenate([pos_idx, neg_idx])
        lab = np.concatenate([np.ones(pos_idx.size, np.float32),
                              np.zeros(neg_idx.size, np.float32)])
        Xs.append(F.reshape(-1, F.shape[-1])[idx])
        ys.append(lab)
        fps_id.append(np.full(idx.size, fi, np.int16))
        rads.append(rnear.ravel()[idx])
        poss.append(np.stack([xx.ravel()[idx], yy.ravel()[idx]], 1))
        print("  t=%5.1fs 正 %4d 负 %4d" % (t, pos_idx.size, neg_idx.size), flush=True)

    X = np.concatenate(Xs); y = np.concatenate(ys)
    fid = np.concatenate(fps_id); rad = np.concatenate(rads)
    pos = np.concatenate(poss)
    np.savez_compressed(os.path.join(OUTDIR, "pixel_dataset.npz"),
                        X=X.astype(np.float32), y=y, frame=fid, r=rad,
                        pos=pos.astype(np.float32), names=np.array(NAMES),
                        times=np.array(sorted(ann.keys()), np.float32))
    print("\n共 %d 个像素样本，正 %d (%.1f%%)，用时 %.0fs"
          % (len(y), int(y.sum()), 100.0 * y.mean(), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
