# -*- coding: utf-8 -*-
"""
训练逐像素分类器：判断"这个像素是不是气泡中心"。

特征 = 跨 13 个尺度的中心核响应 + 若干辅助量（见 build_pixel_dataset.py）。
模型 = 小 MLP（19 -> 64 -> 64 -> 1），BCE 损失，CUDA 上几秒钟训完。

按**帧**划分训练/验证（不是按像素），否则相邻像素泄漏会让指标虚高。
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results_train", "pixel_dataset.npz")
OUT = os.path.join(ROOT, "results_train")


def metrics(y, p, thr):
    pred = p >= thr
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return prec, rec, f1


def auc(y, p):
    """ROC AUC（用秩和公式，避免依赖 sklearn）。"""
    order = np.argsort(p)
    ranks = np.empty(len(p), np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = float(y.sum()); n0 = float(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


class MLP(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    d = np.load(DATA)
    X, y, fid = d["X"], d["y"].astype(np.float32), d["frame"]
    times = d["times"]
    nframe = int(fid.max()) + 1
    # 按帧划分：每 4 帧留 1 帧做验证（时间上均匀分布）
    val_frames = set(range(0, nframe, 4))
    tr = np.array([i for i in range(len(y)) if fid[i] not in val_frames])
    va = np.array([i for i in range(len(y)) if fid[i] in val_frames])
    print("总样本 %d；训练 %d（%d 帧），验证 %d（%d 帧）"
          % (len(y), len(tr), nframe - len(val_frames), len(va), len(val_frames)))

    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", dev)
    Xt = torch.tensor(Xn[tr], device=dev)
    yt = torch.tensor(y[tr], device=dev)
    Xv = torch.tensor(Xn[va], device=dev)
    yv = y[va]

    torch.manual_seed(0)
    model = MLP(X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_w = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    bs = 8192
    best = (-1, None)
    t0 = time.time()
    for ep in range(400):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        tot = 0.0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if ep % 20 == 0 or ep == 399:
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xv)).cpu().numpy()
            a = auc(yv, pv)
            bestf1, bestthr = max(((metrics(yv, pv, t)[2], t)
                                   for t in np.linspace(0.1, 0.9, 33)))
            print("  ep %3d  loss=%.4f  验证 AUC=%.4f  F1=%.3f @ thr=%.2f  (%.0fs)"
                  % (ep, tot / len(Xt), a, bestf1, bestthr, time.time() - t0), flush=True)
            if a > best[0]:
                best = (a, {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        bestthr)

    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(Xv)).cpu().numpy()
    print("\n最佳验证 AUC = %.4f" % best[0])
    print(" thr   准确率  召回率   F1")
    for t in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        pr, rc, f1 = metrics(yv, pv, t)
        print(" %.2f   %.3f   %.3f   %.3f" % (t, pr, rc, f1))

    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd,
                "names": d["names"], "thr": best[2]},
               os.path.join(OUT, "pixel_clf.pt"))
    print("\n模型已保存 -> %s" % os.path.join(OUT, "pixel_clf.pt"))


if __name__ == "__main__":
    main()
