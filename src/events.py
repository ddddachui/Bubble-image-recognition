# -*- coding: utf-8 -*-
"""
融合（聚并）事件判定（M0）

判定口径（已与用户确认）：
  * 主口径 —— "颈部（两气泡之间的凹腰）消失、合并成单一外轮廓"的那一刻算融合完成；
  * 同时记录"首次接触帧"，两种口径都存，可随时切换；
  * 每融合一次，融合计数 +1；破裂不计数。

实现：
  对任意两条在空间上相邻的轨迹，沿两中心连线采样距离变换值，取最小值作为"颈部宽度"。
    neck_ratio = 颈部宽度 / min(r1, r2)
      ≈ 0   -> 两颗刚好接触（甚至还没接触，各自独立）
      -> 1  -> 已经是一个圆，颈部完全填平 => 融合完成
  要求 neck_ratio 连续若干帧超过阈值，才算融合，避免抖动误判。
"""
import numpy as np


def neck_width(dsm, x1, y1, x2, y2, n=41):
    """沿两点连线采样距离图，返回最小值（颈部宽度，单位像素）。"""
    h, w = dsm.shape
    ts = np.linspace(0.0, 1.0, n)
    xs = np.clip(np.round(x1 + (x2 - x1) * ts).astype(int), 0, w - 1)
    ys = np.clip(np.round(y1 + (y2 - y1) * ts).astype(int), 0, h - 1)
    return float(dsm[ys, xs].min())


class FusionDetector:
    """
    跨帧累计"颈部填平"的持续时间，超过阈值即确认融合。

    为避免把"同一个气泡上的两个假峰"误判成融合，要求该配对**曾经确实是两颗**
    （即曾连续若干帧处于"颈部很细"的状态）才允许触发融合。
    """

    def __init__(self, ratio_hi=0.78, ratio_lo=0.45, ratio_contact=0.08,
                 hold=3, sep_frames=8, max_gap_frames=5, min_hits=5):
        self.ratio_hi = ratio_hi            # 判定融合完成的颈部比例阈值
        self.ratio_lo = ratio_lo            # "确实是两颗"的颈部比例上限
        self.ratio_contact = ratio_contact  # 判定"开始接触"的阈值
        self.hold = hold                    # 需要连续满足的帧数
        self.sep_frames = sep_frames        # 需要曾经连续"分明"的帧数
        self.max_gap_frames = max_gap_frames
        self.min_hits = min_hits
        self.state = {}                     # (id1,id2) -> dict

    def step(self, frame, tracks, dsm):
        """
        tracks: 本帧确认的 Track 列表
        dsm:    本帧距离图
        返回 (已确认的融合事件列表, 本帧处于接触但未完成的配对)
        """
        keyed = {}
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = tracks[i], tracks[j]
                if a.hits < self.min_hits or b.hits < self.min_hits:
                    continue
                d = float(np.hypot(a.x - b.x, a.y - b.y))
                if d > 1.35 * (a.r + b.r):
                    continue
                key = (min(a.id, b.id), max(a.id, b.id))
                wneck = neck_width(dsm, a.x, a.y, b.x, b.y)
                ratio = wneck / max(1e-6, min(a.r, b.r))
                st = self.state.setdefault(key, dict(run=0, sep=0, contact=None,
                                                     best=0.0, was_sep=False))
                if frame - st.get("last", -99) > self.max_gap_frames:
                    st["run"] = 0; st["sep"] = 0
                st["last"] = frame
                st["best"] = max(st["best"], ratio)
                if ratio > self.ratio_contact and st["contact"] is None:
                    st["contact"] = frame
                if ratio < self.ratio_lo:
                    st["sep"] += 1
                else:
                    st["sep"] = 0
                if st["sep"] >= self.sep_frames:
                    st["was_sep"] = True
                if ratio >= self.ratio_hi:
                    st["run"] += 1
                else:
                    st["run"] = 0
                keyed[key] = dict(a=a, b=b, ratio=ratio, wall=wneck,
                                  contact=st["contact"], run=st["run"],
                                  best=st["best"], was_sep=st["was_sep"])

        done = []
        for key, info in list(keyed.items()):
            if info["run"] >= self.hold and info["was_sep"]:
                a, b = info["a"], info["b"]
                dur = max(1, info["run"])
                conf = float(np.clip((info["ratio"] - self.ratio_hi) /
                                     max(1e-6, 1.0 - self.ratio_hi), 0, 1))
                conf = float(np.clip(0.5 * conf + 0.5 * min(1.0, dur / 8.0), 0.05, 0.99))
                done.append(dict(
                    frame=frame, id_a=a.id, id_b=b.id,
                    x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0,
                    r_a=a.r, r_b=b.r, neck_ratio=info["ratio"],
                    contact_frame=info["contact"], confidence=conf,
                    area_a=np.pi * a.r ** 2, area_b=np.pi * b.r ** 2))
                self.state.pop(key, None)
                # 该对已合并，其余含这两个 id 的配对一并清理
                for k2 in list(self.state):
                    if a.id in k2 or b.id in k2:
                        self.state.pop(k2, None)
        return done, [v for v in keyed.values() if v["was_sep"]]
