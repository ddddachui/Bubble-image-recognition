# -*- coding: utf-8 -*-
"""
导出待标注帧（给人工点标注用）。

输出：
  标注工具/frames/fNN_orig.png   原始帧
  标注工具/frames/fNN_enh.png    增强帧（时间高通去慢漂移，更容易看清气泡）
  标注工具/frames.js             帧列表（供 index.html 直接读取，无需本地服务器）

用法：python tools/export_frames.py --video video1.mp4 --start 20 --end 80 --step 4
"""
import argparse
import os
import sys

import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d


def imwrite_u(path, img):
    """cv2.imwrite 在 Windows 上不支持中文路径，这里用 imencode + Python 写文件。"""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError("编码失败：%s" % path)
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="video1.mp4")
    ap.add_argument("--out", default="标注工具")
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--end", type=float, default=80.0)
    ap.add_argument("--step", type=float, default=4.0)
    ap.add_argument("--hp-window", type=int, default=125,
                    help="时间高通窗长（帧）；125 帧 @25fps = 5 s")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print("打不开视频：%s" % a.video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 为了算时间高通，需要一段连续帧；这里按需读取
    f0 = int(round(a.start * fps))
    f1 = int(round(a.end * fps))
    targets = [int(round(t * fps)) for t in
               np.arange(a.start, a.end + 1e-6, a.step)]
    need = sorted(set(range(f0, f1 + 1)))
    print("需要读取 %d 帧（%.0f-%.0f s），计算时间高通（窗长 %d 帧 = %.1f s）"
          % (len(need), a.start, a.end, a.hp_window, a.hp_window / fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    gray = np.zeros((len(need), H, W), np.uint8)
    for i in range(len(need)):
        ok, f = cap.read()
        if not ok:
            gray = gray[:i]
            break
        gray[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    cap.release()
    print("实读 %d 帧" % len(gray))

    A = gray.astype(np.float32)
    slow = uniform_filter1d(A, size=a.hp_window, axis=0, mode="nearest")
    hp = A - slow

    nlab, lab, stats, _ = cv2.connectedComponentsWithStats((gray[0] > 45).astype(np.uint8), 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    inside = (lab == k)
    hp[:, ~inside] = 0

    outdir = os.path.join(a.out, "frames")
    os.makedirs(outdir, exist_ok=True)
    items = []
    for t in np.arange(a.start, a.end + 1e-6, a.step):
        idx = int(round(t * fps))
        j = idx - f0
        if j < 0 or j >= len(gray):
            continue
        name = "f%05.1f" % t
        orig = gray[j].copy()
        orig[~inside] = 0
        imwrite_u(os.path.join(outdir, name + "_orig.png"), orig)

        r = hp[j].copy()
        lo, hi = np.percentile(r[inside], 3), np.percentile(r[inside], 99)
        # 用原图灰度作为底，把高通残差按比例叠上去（既保留外观又提升对比）
        en = orig.astype(np.float32) + np.clip(r - (lo + hi) / 2.0, -60, 60) * 1.6
        en = np.clip(en, 0, 255).astype(np.uint8)
        en[~inside] = 0
        en = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(en)
        en[~inside] = 0
        imwrite_u(os.path.join(outdir, name + "_enh.png"), en)
        items.append((name, "%.2f" % t))

    with open(os.path.join(a.out, "frames.js"), "w", encoding="utf-8") as fh:
        fh.write("var FRAMES=[\n")
        for name, t in items:
            fh.write('  {id:"%s", t:%s},\n' % (name, t))
        fh.write("];\n")
        fh.write("var FRAME_W=%d, FRAME_H=%d;\n" % (W, H))
    print("已导出 %d 帧到 %s（每帧两张：原图 / 增强）" % (len(items), outdir))
    print("直接双击打开 %s" % os.path.join(a.out, "index.html"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
