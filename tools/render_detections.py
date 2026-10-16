# -*- coding: utf-8 -*-
"""按正确方式渲染检出结果：高置信度 + 去重后再画。

MODEL=v1 -> 旧模型 patch_cnn.pt，输出 render_clean.png
MODEL=v2（默认） -> 带尺寸特征的 patch_cnn_v2.pt，输出 render_v2_clean.png
"""
import os
import sys

import numpy as np
import cv2
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.train_patch_cnn import CNN, CNNR       # noqa: E402
from src.annotations import load_all              # noqa: E402

OUT = os.path.join(ROOT, "results_train")
THR = float(os.environ.get("THR", "0.95"))
MODEL = os.environ.get("MODEL", "v2")


def main():
    d = np.load(os.path.join(OUT, "patch_dataset.npz"))
    X, meta, r = d["X"], d["meta"], d["r"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if MODEL == "v1":
        ck = torch.load(os.path.join(OUT, "patch_cnn.pt"), weights_only=False)
        net = CNN().to(dev); net.load_state_dict(ck["state"]); net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(torch.tensor(X, device=dev))).cpu().numpy()
    else:
        ck = torch.load(os.path.join(OUT, "patch_cnn_v2.pt"), weights_only=False)
        net = CNNR().to(dev); net.load_state_dict(ck["state"]); net.eval()
        rz = ((np.log(np.maximum(r, 1.0)) - ck["r_mu"]) / ck["r_sd"]).astype(np.float32)
        rz = torch.tensor(rz[:, None], device=dev)
        with torch.no_grad():
            p = torch.sigmoid(net(torch.tensor(X, device=dev), rz)).cpu().numpy()

    ann = load_all(os.path.join(ROOT, "标注工具"))
    panels = []
    for t in (20.0, 30.0, 40.0, 48.0):
        a = ann.get(round(t, 1))
        if a is None:
            continue
        img = cv2.imdecode(np.fromfile(
            os.path.join(ROOT, "标注工具", "frames", "f%05.1f_enh.png" % t),
            dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        m = (np.round(meta[:, 0], 1) == round(t, 1)) & (p >= THR)
        idx = np.flatnonzero(m)
        idx = idx[np.argsort(-p[idx])]
        keep = []
        for i in idx:
            x, y, r = meta[i, 1], meta[i, 2], max(meta[i, 3], 25)
            if all((x - meta[j, 1]) ** 2 + (y - meta[j, 2]) ** 2 >
                   (0.55 * (r + max(meta[j, 3], 25))) ** 2 for j in keep):
                keep.append(i)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for b in a["bubbles"]:
            cv2.circle(vis, (int(b["x"]), int(b["y"])), int(b["r"]), (0, 255, 0), 2)
        for g in a["ghosts"]:
            cv2.drawMarker(vis, (int(g["x"]), int(g["y"])), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 22, 3)
        for i in keep:
            x, y, r = int(meta[i, 1]), int(meta[i, 2]), int(meta[i, 3])
            cv2.circle(vis, (x, y), r, (255, 0, 255), 2)
            cv2.circle(vis, (x, y), 4, (255, 255, 0), -1)
        small = cv2.resize(vis, None, fx=0.62, fy=0.62, interpolation=cv2.INTER_AREA)
        cv2.putText(small, "t=%.0fs  绿=你标的  红叉=虚影  品红=%d 个检出(p>=%.2f,已去重)"
                    % (t, len(keep), THR), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
        panels.append(small)
    if panels:
        name = "render_clean.png" if MODEL == "v1" else "render_v2_clean.png"
        cv2.imwrite(os.path.join(OUT, name), np.vstack(panels))
        print("saved %s  (阈值 %.2f, 模型 %s)" % (name, THR, MODEL))


if __name__ == "__main__":
    main()
