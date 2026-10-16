# -*- coding: utf-8 -*-
"""按概率取前 K 个峰，看准确率/召回率 —— 评估"排序质量"，绕开阈值细节。"""
import os
import sys

import numpy as np
import cv2
import torch
from scipy.ndimage import maximum_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.build_dataset import read_window, VIDEO, SCALE          # noqa: E402
from tools.build_pixel_dataset import feats_all                    # noqa: E402
from tools.eval_detector import load_model, prob_map               # noqa: E402
from src.filters import aperture_mask                              # noqa: E402
from src.annotations import load_all                               # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, mu, sd = load_model(dev)
    d = np.load(os.path.join(ROOT, "results_train", "pixel_dataset.npz"))
    times = d["times"]
    val = sorted(set(range(0, len(times), 4)))
    ann = load_all(os.path.join(ROOT, "标注工具"))
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)

    PK, NB, MATCH = [], 0, 0
    for fi in val:
        t = float(times[fi])
        a = ann.get(round(t, 1))
        if a is None:
            continue
        orig, hp, en = read_window(cap, fps, t)
        if orig is None:
            continue
        mask = aperture_mask(orig)
        F, _ = feats_all(orig, en)
        p = prob_map(model, mu, sd, F, dev)
        p = np.where(mask > 0, p, -1)
        mx = maximum_filter(p, size=19)
        ys, xs = np.nonzero((p >= mx - 1e-9) & (p > 0.5))
        for y, x in zip(ys, xs):
            PK.append((float(p[y, x]), t, float(x), float(y)))
        NB += len(a["bubbles"])
    cap.release()

    PK.sort(reverse=True)
    print("验证帧共 %d 帧，标注气泡 %d 个，候选峰 %d 个" % (len(val), NB, len(PK)))
    print()
    print("  取前K个   命中   误检   准确率  召回率")
    for K in (100, 130, 160, 200, 300, 500, 800, 1200):
        sub = PK[:K]
        used_b = set()
        hit = 0
        for sc, t, x, y in sub:
            a = ann.get(round(t, 1))
            if a is None:
                continue
            for bi, b in enumerate(a["bubbles"]):
                key = (round(t, 1), bi)
                if key in used_b:
                    continue
                bx, by, br = b["x"] * SCALE, b["y"] * SCALE, b["r"] * SCALE
                if np.hypot(x - bx, y - by) < 0.5 * br:
                    used_b.add(key); hit += 1; break
        prec = hit / K
        rec = hit / max(1, NB)
        print("   %5d   %5d  %5d   %.3f   %.3f" % (K, hit, K - hit, prec, rec))


if __name__ == "__main__":
    main()
