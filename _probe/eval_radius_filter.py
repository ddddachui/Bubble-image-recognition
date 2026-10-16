# -*- coding: utf-8 -*-
"""
验证"孔洞点阵是主要误检源"：按候选半径过滤后再看检测级指标。
孔洞很小（半径约 6-8 像素@半分辨率），按最小半径卡掉应该能大幅提升准确率。
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.train_patch_cnn import CNN                    # noqa: E402
from src.annotations import load_all                     # noqa: E402

OUT = os.path.join(ROOT, "results_train")


def main():
    d = np.load(os.path.join(OUT, "patch_dataset.npz"))
    X, y, meta, r = d["X"], d["y"].astype(np.float32), d["meta"], d["r"]
    times = sorted(set(np.round(meta[:, 0], 1)))
    val_times = set(times[::4])
    is_val = np.array([round(float(t), 1) in val_times for t in meta[:, 0]])
    va = np.flatnonzero(is_val)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(OUT, "patch_cnn.pt"), weights_only=False)
    net = CNN().to(dev); net.load_state_dict(ck["state"]); net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(X, device=dev))).cpu().numpy()
    ann = load_all(os.path.join(ROOT, "标注工具"))
    NB = sum(len(ann[round(float(t), 1)]["bubbles"]) for t in val_times
             if round(float(t), 1) in ann)
    print("验证帧标注气泡 %d 个；验证候选 %d 个" % (NB, len(va)))
    print()
    print(" 最小半径   候选数   top130命中 准确率  召回率")
    for rmin in (0, 8, 10, 12, 14, 16, 18, 20):
        keep = va[r[va] >= rmin]
        if keep.size == 0:
            continue
        order = keep[np.argsort(-p[keep])]
        used, hit = set(), 0
        for i in order[:130]:
            t, x, yy = meta[i, 0], meta[i, 1], meta[i, 2]
            a = ann.get(round(float(t), 1))
            if a is None:
                continue
            for bi, b in enumerate(a["bubbles"]):
                k = (round(float(t), 1), bi)
                if k in used:
                    continue
                if np.hypot(x - b["x"], yy - b["y"]) < 0.5 * b["r"]:
                    used.add(k); hit += 1; break
        print("   %4d     %5d     %5d     %.3f   %.3f"
              % (rmin, keep.size, hit, hit / 130.0, hit / max(1, NB)))


if __name__ == "__main__":
    main()
