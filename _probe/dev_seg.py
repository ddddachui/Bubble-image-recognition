# -*- coding: utf-8 -*-
"""开发用：检验气泡分割效果（半分辨率检测），输出诊断图。
用法: python dev_seg.py 52 57 70 30
"""
import sys
import numpy as np
import cv2

VIDEO = r"E:\Bubble image recognition\video1.mp4"
OUT = r"E:\Bubble image recognition\_probe"
SCALE = 0.5                      # 检测用半分辨率


def aperture_mask(frame_small):
    """圆形视场掩膜：非黑区域的最大连通域，填充内部空洞后腐蚀。"""
    lit = (frame_small > 45).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lit, 8)
    if n <= 1:
        return np.ones_like(lit)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cnts, -1, 1, -1)
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    return m


def background(frame, k=151):
    """大核闭运算估计局部背景（暗目标->用周围亮背景填充）。"""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, ker)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4.0)
    return bg


def segment(frame, mask, thr=0.10, open_r=9, min_area=900, max_area=60000):
    bg = background(frame)
    rel = (bg.astype(np.float32) - frame.astype(np.float32)) / (bg.astype(np.float32) + 1e-6)
    rel[~mask.astype(bool)] = 0
    binm = (rel > thr).astype(np.uint8) * 255
    if open_r > 0:
        binm = cv2.morphologyEx(
            binm, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1)))
    binm = cv2.morphologyEx(
        binm, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    # 填内部空洞（气泡中心偏亮）
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(binm)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    nn, lab, stats, cent = cv2.connectedComponentsWithStats((filled > 0).astype(np.uint8), 8)
    keep = np.zeros_like(filled)
    objs = []
    for i in range(1, nn):
        a = stats[i, cv2.CC_STAT_AREA]
        if min_area <= a <= max_area:
            keep[lab == i] = 255
            objs.append((float(cent[i][0]), float(cent[i][1]), int(a)))
    return rel, keep, objs


def main():
    times = [float(x) for x in sys.argv[1:]] or [52.0]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    panels = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        if not ok:
            print("read fail", t); continue
        small = cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        m = aperture_mask(g)
        rel, binm, objs = segment(g, m)
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        vis[binm > 0] = (0.45 * vis[binm > 0] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
        for (cx, cy, a) in objs:
            r = np.sqrt(a / np.pi)
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), 1)
            cv2.putText(vis, "%d" % a, (int(cx - r), int(cy - r)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        relv = np.clip(rel * 600, 0, 255).astype(np.uint8)
        relv = cv2.cvtColor(relv, cv2.COLOR_GRAY2BGR)
        board = np.hstack([vis, relv])
        cv2.putText(board, "t=%.1fs  objs=%d" % (t, len(objs)), (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        panels.append(board)
        print("t=%.1fs  found %d objects, areas=%s" % (t, len(objs), sorted([o[2] for o in objs], reverse=True)[:12]))
    cv2.imwrite(OUT + r"\dev_seg.png", np.vstack(panels))
    cap.release()
    print("saved dev_seg.png")


if __name__ == "__main__":
    main()
