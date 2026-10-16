# -*- coding: utf-8 -*-
"""
跨帧跟踪（M0）

气泡在这个场景里几乎不动（实测漂移 < 0.15 px/帧），所以跟踪策略很直接：
  * 用匀速模型做位置预测；
  * 用"预测位置 -> 检测位置"的距离做匹配代价，匈牙利算法做全局最优关联；
  * 门限随气泡半径缩放，避免把相邻两颗错配；
  * 漏检时允许轨迹"滑行"若干帧（帧内可变暗、被反光吞掉）。

输出：每条轨迹从出生到死亡保持同一个 ID 与颜色。
"""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:                                    # 退化为贪心匹配
    linear_sum_assignment = None


class Track:
    """一条气泡轨迹。"""

    def __init__(self, tid, frame, det):
        self.id = tid
        self.birth = frame
        self.x = det["cx"]
        self.y = det["cy"]
        self.r = det["r"]
        self.vx = 0.0
        self.vy = 0.0
        self.miss = 0                 # 连续未匹配帧数
        self.age = 0                  # 存活帧数
        self.hits = 1                 # 被观测到的帧数
        self.last_frame = frame
        self.max_r = det["r"]
        self.r_sum = det["r"]
        self.confirmed = False
        self.dead = False
        self.death_frame = None
        self.death_reason = None
        self.parent = None            # 融合时记录父轨迹
        # 历史：用于画尺寸-时间曲线
        self.hist = [(frame, det["r"])]

    # ---- 预测 ----
    def predict(self):
        return self.x + self.vx, self.y + self.vy

    # ---- 用一次检测更新 ----
    def update(self, frame, det):
        px, py = self.predict()
        dt = max(1, frame - self.last_frame)
        # 观测与预测的加权（气泡基本静止 -> 平滑系数取小，抗抖动）
        nx = 0.65 * det["cx"] + 0.35 * px
        ny = 0.65 * det["cy"] + 0.35 * py
        self.vx = 0.5 * self.vx + 0.5 * (nx - self.x) / dt
        self.vy = 0.5 * self.vy + 0.5 * (ny - self.y) / dt
        self.x, self.y = nx, ny
        self.r = 0.7 * det["r"] + 0.3 * self.r
        self.max_r = max(self.max_r, det["r"])
        self.r_sum += det["r"]
        self.miss = 0
        self.hits += 1
        self.last_frame = frame
        self.age += 1
        self.hist.append((frame, det["r"]))
        if not self.confirmed and self.hits >= 3:
            self.confirmed = True

    def coast(self, frame):
        """本帧没有匹配到检测：沿用预测位置。"""
        self.x, self.y = self.predict()
        self.miss += 1
        self.age += 1
        self.hist.append((frame, self.r))

    def kill(self, frame, reason):
        self.dead = True
        self.death_frame = frame
        self.death_reason = reason


class Tracker:
    """多目标跟踪器。"""

    def __init__(self, max_miss=40, gate_scale=0.5, gate_min=12.0,
                 min_hits_to_report=3):
        self.tracks = []
        self.all_tracks = []          # 永久保留（供导出 tracks.csv 用）
        self.next_id = 1
        self.max_miss = max_miss
        self.gate_scale = gate_scale
        self.gate_min = gate_min
        self.min_hits = min_hits_to_report
        self.events = []              # (frame, "birth"/"death", id, reason)
        self.suppress = []            # [until_frame, x, y, r] 融合后短暂抑制误新生

    # ------------------------------------------------------------------
    def add_suppression(self, frame, x, y, r, dur=25):
        """融合完成后，在融合位置附近短暂抑制"新轨迹"，避免把残留峰当成新气泡。"""
        self.suppress.append([frame + dur, x, y, max(r * 1.2, 25.0)])

    def _suppressed(self, frame, d):
        for until, x, y, r in self.suppress:
            if frame <= until and np.hypot(d["cx"] - x, d["cy"] - y) < r:
                return True
        return False

    def spawn_merged(self, frame, x, y, r, parents):
        """融合后的新气泡：新建一条轨迹，父轨迹已死。"""
        det = dict(cx=x, cy=y, r=r)
        t = Track(self.next_id, frame, det)
        self.next_id += 1
        t.parent = tuple(parents)
        t.confirmed = True
        self.tracks.append(t)
        self.all_tracks.append(t)
        self.events.append((frame, "birth", t.id, "融合"))
        return t

    # ------------------------------------------------------------------
    def update(self, frame, dets):
        """返回本帧确认存活的轨迹列表。"""
        alive = [t for t in self.tracks if not t.dead]
        if alive and dets:
            C = np.zeros((len(alive), len(dets)), np.float32)
            G = np.zeros_like(C, bool)
            for i, t in enumerate(alive):
                px, py = t.predict()
                for j, d in enumerate(dets):
                    dist = float(np.hypot(d["cx"] - px, d["cy"] - py))
                    C[i, j] = dist
                    G[i, j] = dist <= max(self.gate_min,
                                          self.gate_scale * (t.r + d["r"]))
            big = 1e6
            Cg = np.where(G, C, big)
            if linear_sum_assignment is not None:
                ri, ci = linear_sum_assignment(Cg)
                pairs = [(int(a), int(b)) for a, b in zip(ri, ci) if Cg[a, b] < big]
            else:                                    # 贪心兜底
                order = np.argsort(Cg, axis=None)
                used_r, used_c, pairs = set(), set(), []
                for k in order:
                    a, b = divmod(int(k), len(dets))
                    if a in used_r or b in used_c or Cg[a, b] >= big:
                        continue
                    used_r.add(a); used_c.add(b); pairs.append((a, b))
        else:
            pairs = []

        matched_det = set()
        for a, b in pairs:
            alive[a].update(frame, dets[b])
            matched_det.add(b)

        # 未匹配的轨迹：滑行或死亡
        for t in alive:
            if t.last_frame != frame:
                t.coast(frame)
                if t.miss > self.max_miss and t.confirmed:
                    t.kill(frame, "消失")
                    self.events.append((frame, "death", t.id, "消失"))

        # 未匹配的检测：新建轨迹
        for j, d in enumerate(dets):
            if j in matched_det:
                continue
            if self._suppressed(frame, d):
                continue
            t = Track(self.next_id, frame, d)
            self.next_id += 1
            self.tracks.append(t)
            self.all_tracks.append(t)
            self.events.append((frame, "birth", t.id, None))

        self.suppress = [s for s in self.suppress if frame <= s[0]]
        self.tracks = [t for t in self.tracks if not t.dead or
                       (frame - (t.death_frame or frame)) < 2]
        return [t for t in self.tracks if not t.dead and
                (t.confirmed or t.hits >= self.min_hits)]
