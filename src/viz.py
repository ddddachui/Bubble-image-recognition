# -*- coding: utf-8 -*-
"""可视化：把气泡轮廓/ID、实时计数、融合提示画到画面上。"""
import cv2
import numpy as np

# 一组区分度较好的颜色（BGR），按 ID 稳定分配
PALETTE = [
    (0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 220, 255), (255, 0, 255),
    (0, 165, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255), (255, 128, 0),
    (60, 255, 60), (200, 200, 0), (255, 0, 128), (0, 90, 200), (128, 255, 0),
    (255, 200, 200), (80, 80, 255), (0, 255, 200), (150, 100, 255), (255, 60, 60),
]


def color_for(tid):
    return PALETTE[(tid * 7 + 3) % len(PALETTE)]


def text(img, s, org, scale=0.9, color=(255, 255, 255), thick=2):
    """标签文字（OpenCV 内置字体不支持中文，故全部标签用英文/数字）。"""
    return cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                       cv2.LINE_AA)


def draw(frame, tracks, scale, count, fusions_cum, time_s, contact_pairs=None,
         flash=None, aperture=None):
    """
    frame        原始分辨率 BGR 帧（原地修改）
    tracks       本帧确认的气泡轨迹（坐标是半分辨率）
    scale        检测缩放比（0.5）
    flash        本帧发生的融合事件（画高亮圈）
    """
    inv = 1.0 / scale
    if aperture is not None:
        cnts, _ = cv2.findContours(aperture.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnts = [(c.astype(np.float32) * inv).astype(np.int32) for c in cnts]
        cv2.polylines(frame, cnts, True, (90, 90, 90), 2, cv2.LINE_AA)

    if contact_pairs:
        for p in contact_pairs:
            a, b = p["a"], p["b"]
            cv2.line(frame, (int(a.x * inv), int(a.y * inv)),
                     (int(b.x * inv), int(b.y * inv)), (0, 255, 255), 1, cv2.LINE_AA)

    for t in sorted(tracks, key=lambda z: z.id):
        col = color_for(t.id)
        cx, cy, r = int(t.x * inv), int(t.y * inv), int(t.r * inv)
        cv2.circle(frame, (cx, cy), r, col, 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
        label = "#%d" % t.id
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (cx - tw // 2 - 3, cy - r - th - 8),
                      (cx + tw // 2 + 3, cy - r - 2), (0, 0, 0), -1)
        text(frame, label, (cx - tw // 2, cy - r - 6), 0.7, col, 2)

    if flash:
        for f in flash:
            cv2.circle(frame, (int(f["x"] * inv), int(f["y"] * inv)),
                       int(1.6 * (f["r_a"] + f["r_b"]) * inv), (0, 0, 255), 4, cv2.LINE_AA)
            text(frame, "MERGE", (int(f["x"] * inv) - 40, int(f["y"] * inv)),
                 0.9, (0, 0, 255), 2)

    # 左上角统计面板
    lines = ["bubbles: %d" % count,
             "merges(total): %d" % fusions_cum,
             "t = %6.2f s" % time_s]
    for i, s in enumerate(lines):
        cv2.rectangle(frame, (14, 14 + i * 34), (14 + 250, 44 + i * 34), (0, 0, 0), -1)
        text(frame, s, (22, 40 + i * 34), 0.85, (255, 255, 255), 2)
    return frame
