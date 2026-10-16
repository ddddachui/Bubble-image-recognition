# -*- coding: utf-8 -*-
# 视频侧抢救探针：
#   V0 现状增强 | V1 均值堆叠±0.2s | V2 均值堆叠±0.5s | V3 稳定化+均值堆叠±0.5s
# 输出：
#   results_train/rescue/rescue_t16.png / t32 / t48  （同区域四行对比图）
#   以及"噪声点 vs 真气泡"的局部对比度可分性统计（打印）
import csv
import os
import sys

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.build_dataset import SCALE, enhance            # noqa: E402
from src.filters import integral, boxmean                 # noqa: E402
from src.annotations import load_all                      # noqa: E402

VIDEO = os.path.join(ROOT, "video1.mp4")
OUT = os.path.join(ROOT, "results_train", "rescue")
FONT = "C:/Windows/Fonts/msyh.ttc"
VNAMES = {"V0": "V0 现状", "V1": "V1 均值±0.2s", "V2": "V2 均值±0.5s",
          "V3": "V3 稳定+均值±0.5s"}


def F(sz):
    return ImageFont.truetype(FONT, sz)


def aligned_mean(stack, ref):
    """把一叠帧对齐到 ref 后取均值；返回 (均值, 平均位移幅度)。"""
    h, w = ref.shape
    out, mags = [], []
    for img in stack:
        (dx, dy), _ = cv2.phaseCorrelate(ref.astype(np.float32),
                                         img.astype(np.float32))
        best = None
        for sgn in (1.0, -1.0):
            M = np.float32([[1, 0, sgn * dx], [0, 1, sgn * dy]])
            wp = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            err = float(np.abs(wp - ref).mean())
            if best is None or err < best[0]:
                best = (err, wp)
        out.append(best[1])
        mags.append(float(np.hypot(dx, dy)))
    return np.mean(out, axis=0), float(np.mean(mags))


def build_variants(cap, fps, t):
    c = int(round(t * fps))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h5 = int(5 * fps)
    h02 = max(1, int(round(0.2 * fps)))
    h05 = max(1, int(round(0.5 * fps)))
    a = max(0, c - h5)
    b = min(n_total - 1, c + h5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    buf = {}
    s = None
    cnt = 0
    for i in range(a, b + 1):
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, None, fx=SCALE, fy=SCALE,
                                    interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        if s is None:
            s = np.zeros_like(g)
        s += g
        cnt += 1
        if abs(i - c) <= h05:
            buf[i] = g
    if c not in buf or cnt < 20:
        return None
    slow = s / cnt
    center = buf[c]
    w02 = [buf[i] for i in range(c - h02, c + h02 + 1) if i in buf]
    w05 = [buf[i] for i in range(c - h05, c + h05 + 1) if i in buf]
    m02 = np.mean(w02, axis=0)
    m05 = np.mean(w05, axis=0)
    am05, shift = aligned_mean(np.stack(w05), center)
    return {
        "V0": enhance(center, center - slow),
        "V1": enhance(m02, m02 - slow),
        "V2": enhance(m05, m05 - slow),
        "V3": enhance(am05, am05 - slow),
        "shift": shift,
    }


def en_std_at(img, pts):
    """给点集 (x,y,r 全分辨率)，返回增强图上的 0.6r 盒子内标准差。"""
    f = img.astype(np.float32)
    It, It2 = integral(f), integral(f * f)
    H, W = f.shape
    vals = []
    for x, y, r in pts:
        xi = min(max(int(round(x * SCALE)), 0), W - 1)
        yi = min(max(int(round(y * SCALE)), 0), H - 1)
        k = max(2.0, 0.6 * r * SCALE)
        m1 = float(boxmean(It, k)[yi, xi])
        m2 = float(boxmean(It2, k)[yi, xi])
        vals.append(float(np.sqrt(max(m2 - m1 * m1, 0))))
    return np.array(vals)


def sep_auc(noise_vals, true_vals):
    """true > noise 的 rank AUC（>0.5 表示真气泡对比度更高、可分）。"""
    a = np.asarray(noise_vals, float)
    b = np.asarray(true_vals, float)
    allv = np.concatenate([a, b])
    rk = np.argsort(np.argsort(allv)).astype(float) + 1.0
    u = rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2.0
    return 1.0 - u / (len(a) * len(b))


def render_time(t, variants, pts, ann, path):
    pts_sorted = sorted(pts, key=lambda d: -d[4])
    regs = []
    for d in pts_sorted:
        if all((d[0] - r2[0]) ** 2 + (d[1] - r2[1]) ** 2 > 350 ** 2 for r2 in regs):
            regs.append(d)
        if len(regs) == 3:
            break
    if not regs:
        return
    CELL, HW = 460, 150                    # 半分辨率下裁 2*150 px
    rows = ["V0", "V1", "V2", "V3"]
    sheet = Image.new("RGB", (CELL * len(regs), 104 + CELL * 4), (12, 12, 12))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 5), "视频侧抢救对比 t=%.0fs：每行一种处理，三列=三个噪声集中区域"
                     "（品红圈=判读候选，绿=已标气泡，红叉=虚影）" % t,
            font=F(17), fill=(255, 255, 255))
    dr.text((10, 30), "V0=现状 | V1=±0.2s均值 | V2=±0.5s均值 | V3=稳定化+±0.5s均值"
                      "    该时刻平均抖动位移 %.1f px(半分辨率)"
            % variants["shift"], font=F(17), fill=(150, 220, 255))
    a = ann.get(round(t, 1)) or {"bubbles": [], "ghosts": []}
    for ci, (x, y, r, no, p) in enumerate(regs):
        cx, cy = x * SCALE, y * SCALE
        x0, y0 = int(cx - HW), int(cy - HW)
        for ri, v in enumerate(rows):
            img = variants[v]
            xc0, xc1 = max(0, x0), min(img.shape[1], x0 + 2 * HW)
            yc0, yc1 = max(0, y0), min(img.shape[0], y0 + 2 * HW)
            canvas = np.zeros((2 * HW, 2 * HW), np.uint8)
            canvas[yc0 - y0:yc1 - y0, xc0 - x0:xc1 - x0] = img[yc0:yc1, xc0:xc1]
            rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
            pil = Image.fromarray(rgb).resize((CELL, CELL), Image.LANCZOS)
            d2 = ImageDraw.Draw(pil)
            sc = CELL / (2.0 * HW)
            for b in a["bubbles"]:
                gx, gy = (b["x"] * SCALE - x0) * sc, (b["y"] * SCALE - y0) * sc
                gr = b["r"] * SCALE * sc
                if -gr < gx < CELL + gr and -gr < gy < CELL + gr:
                    d2.ellipse([gx - gr, gy - gr, gx + gr, gy + gr],
                               outline=(0, 220, 0), width=3)
            for g in a["ghosts"]:
                gx, gy = (g["x"] * SCALE - x0) * sc, (g["y"] * SCALE - y0) * sc
                if 0 <= gx < CELL and 0 <= gy < CELL:
                    dd = 8
                    d2.line([gx - dd, gy - dd, gx + dd, gy + dd], fill=(255, 60, 60), width=3)
                    d2.line([gx - dd, gy + dd, gx + dd, gy - dd], fill=(255, 60, 60), width=3)
            mx, my = (cx - x0) * sc, (cy - y0) * sc
            mr = r * SCALE * sc
            d2.ellipse([mx - mr, my - mr, mx + mr, my + mr], outline=(255, 0, 255), width=3)
            d2.rectangle([0, 0, CELL, 24], fill=(0, 0, 0))
            d2.text((5, 3), "%s   #%d" % (VNAMES[v], no), font=F(15), fill=(255, 255, 0))
            sheet.paste(pil, (ci * CELL, 104 + ri * CELL))
            dr.rectangle([ci * CELL, 104 + ri * CELL, (ci + 1) * CELL - 1,
                          104 + (ri + 1) * CELL - 1], outline=(90, 90, 90))
    sheet.save(path)
    print("saved", path)


def main():
    os.makedirs(OUT, exist_ok=True)
    ann = load_all(os.path.join(ROOT, "标注工具"))
    judge = {}
    with open(os.path.join(ROOT, "results_train", "judge", "judge_list.csv"),
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t = round(float(r["时间s"]), 1)
            judge.setdefault(t, []).append(
                (float(r["x"]), float(r["y"]), float(r["半径(全分辨率px)"]),
                 int(r["编号"]), float(r["概率p"])))
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)

    times = [16, 20, 24, 28, 32, 36, 40, 44, 48]
    stats = {v: {"noise": [], "true": []} for v in ("V0", "V1", "V2", "V3")}
    cache = {}
    for t in times:
        variants = build_variants(cap, fps, float(t))
        if variants is None:
            print("!! t=%d 读取失败" % t)
            continue
        cache[t] = variants
        noise_pts = [(x, y, r) for x, y, r, no, p in judge.get(float(t), [])]
        true_pts = [(b["x"], b["y"], b["r"]) for b in ann.get(float(t), {}).get("bubbles", [])]
        print("t=%2d  平均抖动 %.1f px   噪声点 %d  真气泡 %d"
              % (t, variants["shift"], len(noise_pts), len(true_pts)), flush=True)
        for v in ("V0", "V1", "V2", "V3"):
            if noise_pts:
                stats[v]["noise"].append(en_std_at(variants[v], noise_pts))
            if true_pts:
                stats[v]["true"].append(en_std_at(variants[v], true_pts))

    print()
    print("处理           噪声en_std中位   真气泡en_std中位   可分性AUC(true>noise)")
    for v in ("V0", "V1", "V2", "V3"):
        nz = np.concatenate(stats[v]["noise"]) if stats[v]["noise"] else np.array([0.0])
        tr = np.concatenate(stats[v]["true"]) if stats[v]["true"] else np.array([0.0])
        print("%-14s %8.1f %14.1f %14.3f"
              % (VNAMES[v], np.median(nz), np.median(tr), sep_auc(nz, tr)))

    for t in (16, 32, 48):
        if t in cache:
            render_time(float(t), cache[t], judge.get(float(t), []), ann,
                        os.path.join(OUT, "rescue_t%02d.png" % t))


if __name__ == "__main__":
    main()
