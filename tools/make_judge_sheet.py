# -*- coding: utf-8 -*-
"""
生成"边界样本判读图"（给人工判读用）。

池子：50s 以前的验证帧上"模型自信、但没对上人工标注"的候选
      （p>=0.9 的全部；0.5<=p<0.9 的按 p 从高到低补足到 100 个），
      同一位置附近（8 秒内、半径相近）只留一个。

每个候选裁一块局部增强图，画上：
  品红圈 = 程序检出（就是你要判的）    白点 = 候选中心
  绿圈   = 你已标注的气泡              红叉 = 你标注的虚影

输出：
  results_train/judge/判读图_第N页.png    （每页 20 格，编号连续）
  results_train/judge/judge_list.csv     （编号 -> 时间/坐标/半径/概率）
"""
import csv
import glob
import os
import sys

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.train_patch_cnn import CNNR           # noqa: E402
from src.annotations import load_all             # noqa: E402

OUT = os.path.join(ROOT, "results_train")
JD = os.path.join(OUT, "judge")
FRAMES = os.path.join(ROOT, "标注工具", "frames")

CELL = 260
COLS, ROWS = 5, 4
PER_PAGE = COLS * ROWS
MAX_N = 100
T_MAX = 48.0                     # 与"先做 50s 以前"的要求一致
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


def F(size):
    return ImageFont.truetype(FONT_PATH, size)


def load_probs():
    d = np.load(os.path.join(OUT, "patch_dataset.npz"))
    X, meta, r = d["X"], d["meta"], d["r"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(OUT, "patch_cnn_v2.pt"), weights_only=False)
    net = CNNR().to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    rz = ((np.log(np.maximum(r, 1.0)) - ck["r_mu"]) / ck["r_sd"]).astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(X, device=dev),
                              torch.tensor(rz[:, None], device=dev))).cpu().numpy()
    return meta, p


def matched_flags(meta, ann, times):
    matched = np.zeros(len(meta), bool)
    for t in times:
        a = ann.get(round(float(t), 1))
        if a is None:
            continue
        m = np.round(meta[:, 0], 1) == round(float(t), 1)
        for i in np.flatnonzero(m):
            for b in a["bubbles"]:
                if np.hypot(meta[i, 1] - b["x"], meta[i, 2] - b["y"]) < 0.5 * b["r"]:
                    matched[i] = True
                    break
    return matched


def select(meta, p, matched, is_val, times_files_ok):
    """挑候选：先全部 p>=0.9，再 0.7-0.9，再 0.5-0.7，取到 MAX_N 为止。"""
    ok = is_val & (~matched) & (p >= 0.5) & (meta[:, 0] <= T_MAX)
    ok &= np.array([round(float(t), 1) in times_files_ok for t in meta[:, 0]])
    order = np.flatnonzero(ok)
    order = order[np.argsort(-p[order])]

    def band(lo, hi):
        return [i for i in order if lo <= p[i] < hi]

    picked = []

    def try_add(i):
        x, y, r, t = meta[i, 1], meta[i, 2], meta[i, 3], meta[i, 0]
        for j in picked:
            gap = abs(t - meta[j, 0])
            if gap < 8.0 and np.hypot(x - meta[j, 1], y - meta[j, 2]) < 0.6 * max(r, meta[j, 3]) \
                    and 0.6 < r / max(meta[j, 3], 1e-6) < 1.6:
                return False
        picked.append(i)
        return True

    for lo, hi in ((0.9, 1.01), (0.7, 0.9), (0.5, 0.7)):
        for i in band(lo, hi):
            if len(picked) >= MAX_N:
                break
            try_add(i)
        if len(picked) >= MAX_N:
            break
    picked.sort(key=lambda i: (round(float(meta[i, 0]), 1), -p[i]))
    return picked


def nearest_ann(meta, i, ann):
    t = round(float(meta[i, 0]), 1)
    a = ann.get(t)
    if not a or not a["bubbles"]:
        return -1.0, 0.0
    dst, rbest = 1e9, 0.0
    for b in a["bubbles"]:
        dd = float(np.hypot(meta[i, 1] - b["x"], meta[i, 2] - b["y"]))
        if dd < dst:
            dst, rbest = dd, float(b["r"])
    return dst, rbest


def render_cell(meta, i, p, ann, img_cache, no):
    t = round(float(meta[i, 0]), 1)
    x, y, r = float(meta[i, 1]), float(meta[i, 2]), float(meta[i, 3])
    img = img_cache[t]
    H, W = img.shape
    view = max(90.0, 1.7 * r)
    x0, y0 = x - view, y - view
    xc0, xc1 = max(0, int(round(x0))), min(W, int(round(x - view + 2 * view)))
    yc0, yc1 = max(0, int(round(y0))), min(H, int(round(y - view + 2 * view)))
    s = CELL / (2.0 * view)
    cell = Image.new("RGB", (CELL, CELL), (18, 18, 18))
    crop = img[yc0:yc1, xc0:xc1]
    big = cv2.resize(crop, None, fx=s, fy=s,
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    cell.paste(Image.fromarray(big), (int((xc0 - x0) * s), int((yc0 - y0) * s)))
    dr = ImageDraw.Draw(cell)

    a = ann.get(t)
    if a:
        for b in a["bubbles"]:
            gx, gy, gr = (b["x"] - x0) * s, (b["y"] - y0) * s, b["r"] * s
            if -gr < gx < CELL + gr and -gr < gy < CELL + gr:
                dr.ellipse([int(gx - gr), int(gy - gr), int(gx + gr), int(gy + gr)],
                           outline=(0, 220, 0), width=3)
        for g in a["ghosts"]:
            gx, gy = (g["x"] - x0) * s, (g["y"] - y0) * s
            if 0 <= gx < CELL and 0 <= gy < CELL:
                d = 9
                dr.line([int(gx - d), int(gy - d), int(gx + d), int(gy + d)], fill=(255, 60, 60), width=3)
                dr.line([int(gx - d), int(gy + d), int(gx + d), int(gy - d)], fill=(255, 60, 60), width=3)

    cx, cy, cr = (x - x0) * s, (y - y0) * s, r * s
    dr.ellipse([int(cx - cr), int(cy - cr), int(cx + cr), int(cy + cr)],
               outline=(255, 0, 255), width=3)
    dr.ellipse([int(cx) - 3, int(cy) - 3, int(cx) + 3, int(cy) + 3], fill=(255, 255, 255))

    dr.rectangle([0, CELL - 26, CELL, CELL], fill=(0, 0, 0))
    dr.text((6, CELL - 24), "#%d  t=%ds  r=%d  p=%.2f" % (no, round(t), round(r), p[i]),
            font=F(15), fill=(255, 255, 0))
    for ox, oy in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
        dr.text((10 + ox, 6 + oy), str(no), font=F(30), fill=(0, 0, 0))
    dr.text((10, 6), str(no), font=F(30), fill=(255, 255, 255))
    return cell


def main():
    os.makedirs(JD, exist_ok=True)
    meta, p = load_probs()
    ann = load_all(os.path.join(ROOT, "标注工具"))
    times = sorted(set(np.round(meta[:, 0], 1)))
    val_times = set(times[::4])
    is_val = np.array([round(float(t), 1) in val_times for t in meta[:, 0]])
    matched = matched_flags(meta, ann, times)
    have = {round(float(s.split("f")[-1].split("_")[0]), 1)
            for s in glob.glob(os.path.join(FRAMES, "f*_enh.png"))}
    picked = select(meta, p, matched, is_val, have)
    n_pages = (len(picked) + PER_PAGE - 1) // PER_PAGE
    print("候选 %d 个（p>=0.9 %d 个），共 %d 页"
          % (len(picked), sum(p[i] >= 0.9 for i in picked), n_pages))

    img_cache = {}
    for i in picked:
        t = round(float(meta[i, 0]), 1)
        if t not in img_cache:
            fp = os.path.join(FRAMES, "f%05.1f_enh.png" % t)
            img_cache[t] = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    rows = []
    for page in range(n_pages):
        sheet = Image.new("RGB", (COLS * CELL, 58 + ROWS * CELL), (12, 12, 12))
        dr = ImageDraw.Draw(sheet)
        dr.text((10, 5), "边界样本判读 第 %d/%d 页 · 共 %d 个候选 · 品红圈=程序检出(没对上你的标注)  绿圈=你已标气泡  红叉=虚影"
                % (page + 1, n_pages, len(picked)), font=F(18), fill=(
                    255, 255, 255))
        dr.text((10, 32), "回复格式：编号+算/不算/重复(和绿圈是同一个气泡)/不确定，例：3算 7不算 12重复 15不确定",
                font=F(18), fill=(150, 220, 255))
        for k in range(PER_PAGE):
            gi = page * PER_PAGE + k
            if gi >= len(picked):
                break
            i = picked[gi]
            no = gi + 1
            cell = render_cell(meta, i, p, ann, img_cache, no)
            r_, c_ = divmod(k, COLS)
            sheet.paste(cell, (c_ * CELL, 58 + r_ * CELL))
            dr.rectangle([c_ * CELL, 58 + r_ * CELL, (c_ + 1) * CELL - 1,
                          58 + (r_ + 1) * CELL - 1], outline=(90, 90, 90))
            dst, ra = nearest_ann(meta, i, ann)
            rows.append([no, page + 1, "%d-%d" % (r_ + 1, c_ + 1), round(float(meta[i, 0]), 1),
                         round(float(meta[i, 1])), round(float(meta[i, 2])),
                         round(float(meta[i, 3])), round(float(p[i]), 3),
                         round(dst, 1) if dst >= 0 else "", round(ra, 1)])
        fp = os.path.join(JD, "判读图_第%d页.png" % (page + 1))
        sheet.save(fp)
        print("saved", fp)

    with open(os.path.join(JD, "judge_list.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["编号", "页码", "格位", "时间s", "x", "y", "半径(全分辨率px)", "概率p",
                    "最近标注中心距", "最近标注半径"])
        w.writerows(rows)
    print("saved", os.path.join(JD, "judge_list.csv"))


if __name__ == "__main__":
    main()
