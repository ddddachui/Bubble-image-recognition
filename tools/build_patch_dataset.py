# -*- coding: utf-8 -*-
"""
候选点 + 图像块 数据集。

候选点来自逐像素分类器的概率峰（召回 0.93，够用）；
每个候选点裁一块**按该候选半径归一化**的图像块（3.2r -> 40x40），
让 CNN 直接看"中心核"长什么样，而不是靠我手工设计的特征。

另外给每个候选存 4 个"绝对量纲"特征（extra 列）：
  tstd   时间标准差（±5s 窗口）    —— 动态性
  tdiff  帧间差均值                 —— 闪烁程度
  en_std 增强图局部对比度（0.6r）    —— 边界/中心核明显程度
  ccon   中心(0.35r)-外圈(1.4r)对比  —— 中心核相对背景的亮度差

输出: results_train/patch_dataset.npz（X/y/r/meta + extra）
"""
import os
import sys
import time

import numpy as np
import cv2
import torch
from scipy.ndimage import maximum_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.build_dataset import read_window2, VIDEO, SCALE             # noqa: E402
from tools.build_pixel_dataset import feats_all, SCALES               # noqa: E402
from tools.eval_detector import load_model, prob_map                  # noqa: E402
from src.filters import aperture_mask, integral, boxmean              # noqa: E402
from src.annotations import load_all                                  # noqa: E402

OUT = os.path.join(ROOT, "results_train")
P = 40            # 图像块边长
COVER = 3.2       # 图像块覆盖气泡半径的倍数


def patch_at(img, x, y, r, cover=COVER, p=P):
    """以 (x,y) 为中心裁一块 2*cover*r 大小的图，缩放到 p×p。"""
    h = max(4, int(round(cover * r)))
    x0, x1 = int(round(x)) - h, int(round(x)) + h
    y0, y1 = int(round(y)) - h, int(round(y)) + h
    H, W = img.shape
    out = np.zeros((2 * h, 2 * h), np.float32)
    sx0, sx1 = max(0, x0), min(W, x1)
    sy0, sy1 = max(0, y0), min(H, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    pb = cv2.resize(out, (p, p), interpolation=cv2.INTER_AREA)
    m, s = float(pb.mean()), float(pb.std()) + 1e-6
    return ((pb - m) / s).astype(np.float32)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, mu, sd = load_model(dev)
    ann = load_all(os.path.join(ROOT, "标注工具"))
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rng = np.random.default_rng(0)

    Xs, ys, meta, rads, exs = [], [], [], [], []
    t0 = time.time()
    for t, a in sorted(ann.items()):
        orig, hp, en, tstd, tdiff = read_window2(cap, fps, t)
        if orig is None:
            continue
        mask = aperture_mask(orig)
        F, _ = feats_all(orig, en)
        p = prob_map(model, mu, sd, F, dev)
        p = np.where(mask > 0, p, -1)
        mx = maximum_filter(p, size=19)
        ys_, xs_ = np.nonzero((p >= mx - 1e-9) & (p > 0.5))
        props = [(float(p[y_, x_]), float(x_), float(y_)) for y_, x_ in zip(ys_, xs_)]

        bubs = [(b["x"] * SCALE, b["y"] * SCALE, b["r"] * SCALE) for b in a["bubbles"]]
        # 把标注中心也作为候选（保证正样本不漏）
        for bx, by, br in bubs:
            props.append((1.0, bx, by))

        en_f = en.astype(np.float32)
        It_e = integral(en_f)
        It_e2 = integral(en_f * en_f)
        bm_cache = {}

        def bm_of(kind, h):
            """按 (kind, 半宽) 缓存盒子均值图（尺度只有 13 档，重复很多）。"""
            key = (kind, round(float(h), 1))
            if key not in bm_cache:
                bm_cache[key] = boxmean(It_e if kind == "m" else It_e2, h)
            return bm_cache[key]

        npos = 0
        for sc, x, y in props:
            xi = int(np.clip(round(x), 0, orig.shape[1] - 1))
            yi = int(np.clip(round(y), 0, orig.shape[0] - 1))
            r = float(SCALES[int(np.argmax(F[yi, xi, :len(SCALES)]))])
            lab = 0
            for bx, by, br in bubs:
                if np.hypot(x - bx, y - by) < 0.5 * br:
                    lab = 1
                    break
            Xs.append(patch_at(en_f, x, y, max(r, 6.0)))
            ys.append(lab)
            rads.append(r)
            meta.append((t, x / SCALE, y / SCALE, r / SCALE))
            # 绝对量纲特征
            inner = float(bm_of("m", max(1.0, 0.35 * r))[yi, xi])
            outer = float(bm_of("m", max(2.0, 1.40 * r))[yi, xi])
            kk = max(2.0, 0.60 * r)
            m1 = float(bm_of("m", kk)[yi, xi])
            m2 = float(bm_of("s", kk)[yi, xi])
            en_std = float(np.sqrt(max(m2 - m1 * m1, 0)))
            exs.append((float(tstd[yi, xi]), float(tdiff[yi, xi]),
                        en_std, inner - outer))
            npos += lab
        print("  t=%5.1fs 候选 %4d  正 %3d / 标注 %3d"
              % (t, len(props), npos, len(bubs)), flush=True)

    X = np.stack(Xs)[:, None, :, :]
    np.savez_compressed(os.path.join(OUT, "patch_dataset.npz"),
                        X=X.astype(np.float32), y=np.array(ys, np.int8),
                        r=np.array(rads, np.float32),
                        meta=np.array(meta, np.float32),
                        extra=np.array(exs, np.float32))
    print("\n共 %d 个候选块，正 %d (%.1f%%)，用时 %.0fs"
          % (len(ys), int(sum(ys)), 100.0 * np.mean(ys), time.time() - t0), flush=True)
    ex = np.array(exs)
    print("extra 中位: tstd=%.1f tdiff=%.1f en_std=%.1f ccon=%.1f"
          % tuple(np.median(ex, axis=0)), flush=True)


if __name__ == "__main__":
    main()
