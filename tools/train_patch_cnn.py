# -*- coding: utf-8 -*-
"""
训练"候选块分类 CNN"，并做检测级评估（按概率取前 K 个峰）。

三版对照：
  v1  只看图像块（按候选半径归一化）
  v2  图像块 + log(半径) 尺寸特征
  v3  图像块 + [log半径, 时间动态(tstd/tdiff), 绝对对比度(en_std/ccon)]
      （v2 针对"小孔洞放大后和大气泡长得一样"；v3 针对"暗而平的大团块"）
本脚本训练 v3 并保存 patch_cnn_v3.pt；v1/v2 读已保存模型做对照，不重训、不覆盖。
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.annotations import load_all            # noqa: E402

OUT = os.path.join(ROOT, "results_train")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64))   # 40->5
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(64 * 5 * 5, 64),
                               nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.h(self.f(x)).squeeze(-1)


class CNNR(nn.Module):
    """v2：图像块 + 一个标量尺寸特征 log(半径)（已标准化）。"""

    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64))   # 40->5
        self.h = nn.Sequential(nn.Linear(64 * 5 * 5 + 1, 64),
                               nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x, rv):
        f = self.f(x).flatten(1)
        return self.h(torch.cat([f, rv], dim=1)).squeeze(-1)


class CNNV3(nn.Module):
    """v3：图像块 + 5 个标量（log 半径 + 时间动态 2 个 + 对比度 2 个，已标准化）。"""

    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64))   # 40->5
        self.h = nn.Sequential(nn.Linear(64 * 5 * 5 + 5, 64),
                               nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x, ev):
        f = self.f(x).flatten(1)
        return self.h(torch.cat([f, ev], dim=1)).squeeze(-1)


def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = float(y.sum()); n0 = float(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def topk_report(meta, prob, val_times, ann, tag=""):
    """按概率取前 K 个候选，算检测级准确率/召回率。"""
    order = np.argsort(-prob)
    NB = 0
    for t in val_times:
        a = ann.get(round(float(t), 1))
        if a:
            NB += len(a["bubbles"])
    print("  [%s] 取前K   命中  误检  准确率 召回率   (验证帧标注气泡共 %d)" % (tag, NB))
    for K in (100, 130, 160, 200, 300, 500):
        used, hit, fps = set(), 0, []
        for i in order[:K]:
            t, x, y = meta[i, 0], meta[i, 1], meta[i, 2]
            a = ann.get(round(float(t), 1))
            if a is None:
                continue
            matched = False
            for bi, b in enumerate(a["bubbles"]):
                k = (round(float(t), 1), bi)
                if k in used:
                    continue
                if np.hypot(x - b["x"], y - b["y"]) < 0.5 * b["r"]:
                    used.add(k); hit += 1; matched = True; break
            if not matched:
                fps.append(i)
        print("   %5d  %5d %5d  %.3f  %.3f" % (K, hit, K - hit, hit / K, hit / max(1, NB)))
        if K == 130 and fps:
            rr = meta[fps, 3]        # 与人工标注同单位的半径（整幅视频像素）
            print("          FP@130 半径: 中位 %.0f, >150 占 %.0f%%, >200 占 %.0f%%"
                  % (np.median(rr), 100.0 * (rr > 150).mean(), 100.0 * (rr > 200).mean()))


def main():
    d = np.load(os.path.join(OUT, "patch_dataset.npz"))
    X, y, meta = d["X"], d["y"].astype(np.float32), d["meta"]
    r = d["r"].astype(np.float32)
    extra = d["extra"].astype(np.float32)      # [tstd, tdiff, en_std, ccon]
    times = sorted(set(np.round(meta[:, 0], 1)))
    val_times = set(times[::4])
    is_val = np.array([round(float(t), 1) in val_times for t in meta[:, 0]])
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print("候选块 %d；训练 %d，验证 %d（验证时间点：%s）"
          % (len(y), len(tr), len(va), ", ".join("%.0f" % t for t in sorted(val_times))))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(X, device=dev); yt = torch.tensor(y, device=dev)
    logr = np.log(np.maximum(r, 1.0)).astype(np.float32)
    mu, sd = float(logr[tr].mean()), float(logr[tr].std() + 1e-6)
    rz = torch.tensor(((logr - mu) / sd)[:, None], device=dev)
    emu = extra[tr].mean(axis=0).astype(np.float32)
    esd = extra[tr].std(axis=0).astype(np.float32) + 1e-6
    ez = torch.tensor((extra - emu) / esd, device=dev)
    print("尺寸特征 log(r)：训练集 mu=%.3f  sd=%.3f" % (mu, sd))
    print("额外特征 [tstd, tdiff, en_std, ccon]：训练集 mu=%s sd=%s"
          % (np.round(emu, 1).tolist(), np.round(esd, 1).tolist()))

    def fwd(net, mode, xb, rb, eb):
        if mode == "v3":
            return net(xb, torch.cat([rb, eb], dim=1))
        if mode == "v2":
            return net(xb, rb)
        return net(xb)

    def fit(net, mode, tag):
        net = net.to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        pw = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())], device=dev)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        best_auc, best_state = -1, None
        t0 = time.time()
        for ep in range(60):
            net.train()
            perm = np.random.permutation(len(tr))
            for i in range(0, len(tr), 256):
                idx = tr[perm[i:i + 256]]
                opt.zero_grad()
                lossf(fwd(net, mode, Xt[idx], rz[idx], ez[idx]), yt[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pv = torch.sigmoid(fwd(net, mode, Xt[va], rz[va], ez[va])).cpu().numpy()
            a = auc(y[va], pv)
            if a > best_auc:
                best_auc, best_state = a, {k: v.cpu().clone() for k, v in net.state_dict().items()}
            if ep % 10 == 0 or ep == 59:
                print("  [%s] ep %2d  验证 AUC=%.4f  (%.0fs)" % (tag, ep, a, time.time() - t0),
                      flush=True)
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            pall = torch.sigmoid(fwd(net, mode, Xt, rz, ez)).cpu().numpy()
        return best_auc, pall

    def ckpt_probs(name, net, mode):
        """读已保存模型算全量概率（v1/v2 参考对照用，不重训）。"""
        ck = torch.load(os.path.join(OUT, name), weights_only=False)
        net = net.to(dev); net.load_state_dict(ck["state"]); net.eval()
        with torch.no_grad():
            return torch.sigmoid(fwd(net, mode, Xt, rz, ez)).cpu().numpy()

    print("\n--- 参考：v1 / v2（读已保存的模型）---")
    p1 = ckpt_probs("patch_cnn.pt", CNN(), "v1")
    p2 = ckpt_probs("patch_cnn_v2.pt", CNNR(), "v2")
    print("--- v3: 图像块 + [log半径, 时间动态, 对比度] ---")
    np.random.seed(0); torch.manual_seed(0)
    net3 = CNNV3()
    auc3, p3 = fit(net3, "v3", "v3")

    print("\n验证 AUC: v1=%.4f  v2=%.4f  v3=%.4f"
          % (auc(y[va], p1[va]), auc(y[va], p2[va]), auc3))
    ann = load_all(os.path.join(ROOT, "标注工具"))
    topk_report(meta[va], p1[va], sorted(val_times), ann, "v1")
    topk_report(meta[va], p2[va], sorted(val_times), ann, "v2")
    topk_report(meta[va], p3[va], sorted(val_times), ann, "v3")
    torch.save({"state": net3.state_dict(), "val_auc": auc3,
                "r_mu": mu, "r_sd": sd, "e_mu": emu, "e_sd": esd},
               os.path.join(OUT, "patch_cnn_v3.pt"))
    print("\nv3 模型 -> %s   （v1/v2 的模型文件未改动）"
          % os.path.join(OUT, "patch_cnn_v3.pt"))


if __name__ == "__main__":
    main()
