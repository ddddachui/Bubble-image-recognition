# -*- coding: utf-8 -*-
"""生成复核图：人工标注 vs 程序检出（含局部放大）。

MODEL=v1 -> patch_cnn.pt，输出 review_overlay.png
MODEL=v2（默认） -> patch_cnn_v2.pt，输出 review_v2.png
"""
import os
import sys

import numpy as np
import cv2
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.train_patch_cnn import CNN, CNNR              # noqa: E402
from src.annotations import load_all                     # noqa: E402

OUT = os.path.join(ROOT, "results_train")
SCALE = 0.5
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
    meta = np.concatenate([meta, p[:, None]], 1)

    ann = load_all(os.path.join(ROOT, "标注工具"))
    panels = []
    for t in (20.0, 30.0, 40.0):
        a = ann[round(t, 1)]
        # 用已有增强图（从标注工具目录取）作为底图
        img = cv2.imdecode(np.fromfile(
            os.path.join(ROOT, "标注工具", "frames", "f%05.1f_enh.png" % t),
            dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        sel = meta[np.round(meta[:, 0], 1) == round(t, 1)]
        top = sel[np.argsort(-sel[:, 4])][:130]
        for _, x, y, r, prob in top:
            cv2.circle(vis, (int(x), int(y)), max(3, int(r)), (255, 0, 255), 1)
        for b in a["bubbles"]:
            cv2.circle(vis, (int(b["x"]), int(b["y"])), int(b["r"]), (0, 255, 0), 3)
        for g in a["ghosts"]:
            cv2.drawMarker(vis, (int(g["x"]), int(g["y"])), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 20, 3)
        small = cv2.resize(vis, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        cv2.putText(small, "t=%.0fs  绿=你标的  品红=程序前130检出" % t, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        panels.append(small)
    if panels:
        name = "review_overlay.png" if MODEL == "v1" else "review_v2.png"
        cv2.imwrite(os.path.join(OUT, name), np.vstack(panels))
        print("saved %s  (模型 %s)" % (name, MODEL))


if __name__ == "__main__":
    main()
