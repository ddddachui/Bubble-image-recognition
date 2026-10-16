# -*- coding: utf-8 -*-
"""
检测级评估：概率图取峰 -> 和人工标注对账（准确率/召回率）。
同时导出带标注对比的可视化图，便于人眼复核。
"""
import os
import sys

import numpy as np
import cv2
import torch
from scipy.ndimage import maximum_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.build_dataset import read_window, VIDEO, SCALE               # noqa: E402
from tools.build_pixel_dataset import feats_all, SCALES                 # noqa: E402
from tools.train_pixel_clf import MLP                                   # noqa: E402
from src.filters import aperture_mask                                   # noqa: E402
from src.annotations import load_all                                    # noqa: E402

OUT = os.path.join(ROOT, "results_train")


def load_model(dev):
    ck = torch.load(os.path.join(OUT, "pixel_clf.pt"), weights_only=False)
    m = MLP(len(ck["names"])).to(dev)
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck["mu"], ck["sd"]


def prob_map(model, mu, sd, F, dev):
    H, W, C = F.shape
    X = torch.tensor(((F.reshape(-1, C) - mu) / sd).astype(np.float32), device=dev)
    with torch.no_grad():
        p = torch.sigmoid(model(X)).cpu().numpy()
    return p.reshape(H, W)


def peaks(p, mask, thr, min_sep=11, maxn=200):
    q = p.copy(); q[mask == 0] = -1
    mx = maximum_filter(q, size=2 * min_sep + 1)
    pk = (q >= mx - 1e-9) & (q > thr)
    ys, xs = np.nonzero(pk)
    if ys.size == 0:
        return []
    order = np.argsort(-q[ys, xs])[:maxn]
    return [(float(q[y, x]), float(x), float(y)) for y, x in zip(ys[order], xs[order])]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, mu, sd = load_model(dev)
    d = np.load(os.path.join(OUT, "pixel_dataset.npz"))
    times = d["times"]
    val = set(range(0, len(times), 4))
    ann = load_all(os.path.join(ROOT, "标注工具"))

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []
    print("检测级评估（验证帧 = 每 4 帧取 1）")
    print("  时间  阈值  检出  标注  命中  误检  漏检   准确率 召回率  F1")
    for thr in (0.5, 0.6, 0.7, 0.8):
        TP = FP = FN = ND = NB = 0
        for fi in sorted(val):
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
            pk = peaks(p, mask, thr)
            ND += len(pk)
            hits = 0
            used = set()
            for b in a["bubbles"]:
                bx, by, br = b["x"] * SCALE, b["y"] * SCALE, b["r"] * SCALE
                for i, (sc, x, y) in enumerate(pk):
                    if i in used:
                        continue
                    if np.hypot(x - bx, y - by) < 0.5 * br:
                        hits += 1; used.add(i); break
            NB += len(a["bubbles"])
            TP += hits
            FN += len(a["bubbles"]) - hits
            FP += len(pk) - hits
        prec = TP / max(1, TP + FP)
        rec = TP / max(1, TP + FN)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        print("        %.1f  %4d  %4d  %4d  %4d  %4d   %.3f  %.3f  %.3f"
              % (thr, ND, NB, TP, FP, FN, prec, rec, f1))

    # 可视化：最后一个阈值，画两帧
    thr = 0.7
    for fi in sorted(val)[:3]:
        t = float(times[fi])
        a = ann.get(round(t, 1))
        if a is None:
            continue
        orig, hp, en = read_window(cap, fps, t)
        mask = aperture_mask(orig)
        F, _ = feats_all(orig, en)
        p = prob_map(model, mu, sd, F, dev)
        pk = peaks(p, mask, thr)
        vis = cv2.cvtColor(en, cv2.COLOR_GRAY2BGR)
        for b in a["bubbles"]:
            cv2.circle(vis, (int(b["x"] * SCALE), int(b["y"] * SCALE)),
                       int(b["r"] * SCALE), (0, 255, 0), 2)
        for g in a["ghosts"]:
            x, y = int(g["x"] * SCALE), int(g["y"] * SCALE)
            cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        for sc, x, y in pk:
            cv2.circle(vis, (int(x), int(y)), 4, (255, 0, 255), -1)
        cv2.putText(vis, "t=%.0fs  绿=你标的 红叉=你说的虚影 品红点=程序检出" % t,
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        rows.append(vis)
    if rows:
        cv2.imwrite(os.path.join(OUT, "eval_overlay.png"), np.vstack(rows))
        print("\n对比图 -> %s" % os.path.join(OUT, "eval_overlay.png"))
    cap.release()


if __name__ == "__main__":
    main()
