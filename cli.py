# -*- coding: utf-8 -*-
"""
气泡识别命令行入口

用法：
    python cli.py analyze video1.mp4 --out results --start 16.3 --end 80

产出（都在 --out 目录下）：
    annotated.mp4   标注视频：轮廓 + ID + 实时计数 + 累计融合次数
    count.csv       每帧：气泡数、累计融合数、累计新生/消失数
    tracks.csv      每条气泡轨迹：出生/死亡帧、位置、最大半径、消失原因
    events.csv      融合事件：帧号、时间、涉及 ID、颈部比、首次接触帧、置信度
    size_time.png   每条轨迹的半径-时间曲线
"""
import argparse
import csv
import os
import sys
import time as _time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import segment as S            # noqa: E402
from src import viz                     # noqa: E402
from src.track import Tracker           # noqa: E402
from src.events import FusionDetector   # noqa: E402

DETECT_CFG = dict(med_k=151, lo=0.030, hi=0.140, open_r=6, close_r=17,
                  min_r=17.0, min_sep=15, prominence=0.70, max_r=62.0,
                  merge_ratio=0.88)
SCALE = 0.5


def analyze(args):
    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("打不开视频：%s" % args.video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(round(args.start * fps))
    f1 = int(round(args.end * fps)) if args.end else total
    f1 = min(f1, total)
    print("视频 %dx%d @%.2ffps 共%d帧；分析 %d-%d 帧 (%.1f-%.1f s)"
          % (W, H, fps, total, f0, f1, f0 / fps, f1 / fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(os.path.join(args.out, "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    tracker = Tracker(max_miss=int(args.max_miss), gate_scale=0.5, gate_min=12.0)
    fdet = FusionDetector(ratio_hi=args.merge_ratio, hold=args.merge_hold)

    rows_count, rows_events, t_start = [], [], _time.time()
    fusion_cum = 0
    births_cum = 0
    confirmed_ids = set()
    aperture = None

    for idx in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            print("第 %d 帧读取失败，提前结束" % idx)
            break
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if aperture is None:            # 机位固定：视场掩膜只需算一次
            aperture = S.aperture_mask(gray)
        rel, fill, dsm, dets = S.detect(gray, aperture, DETECT_CFG)
        tracks = tracker.update(idx, dets)
        fusions, contacts = fdet.step(idx, tracks, dsm)

        flash = []
        for fu in fusions:
            fusion_cum += 1
            flash.append(fu)
            # 父轨迹结束
            for tid in (fu["id_a"], fu["id_b"]):
                for t in tracker.tracks:
                    if t.id == tid and not t.dead:
                        t.kill(idx, "融合")
                        tracker.events.append((idx, "death", tid, "融合"))
            r_new = float(np.sqrt(fu["r_a"] ** 2 + fu["r_b"] ** 2))
            tracker.spawn_merged(idx, fu["x"], fu["y"], r_new,
                                 (fu["id_a"], fu["id_b"]))
            tracker.add_suppression(idx, fu["x"], fu["y"], r_new, dur=25)
            rows_events.append(dict(
                type="merge", frame=idx, time=round(idx / fps, 3),
                id_a=fu["id_a"], id_b=fu["id_b"],
                new_id=tracker.next_id - 1,
                x=round(fu["x"] / SCALE, 1), y=round(fu["y"] / SCALE, 1),
                r_a=round(fu["r_a"] / SCALE, 1), r_b=round(fu["r_b"] / SCALE, 1),
                neck_ratio=round(fu["neck_ratio"], 3),
                contact_frame=fu["contact_frame"],
                contact_time=(round(fu["contact_frame"] / fps, 3)
                              if fu["contact_frame"] is not None else ""),
                confidence=round(fu["confidence"], 3)))

        alive = [t for t in tracker.tracks if not t.dead]
        shown = [t for t in alive if t.confirmed or t.hits >= 3]
        fresh = [t for t in shown if t.miss <= 2]      # 本帧真正看到的
        for t in alive:
            if t.confirmed and t.id not in confirmed_ids:
                confirmed_ids.add(t.id)
                births_cum += 1
        rows_count.append(dict(frame=idx, time=round(idx / fps, 3),
                               bubble_count=len(fresh),
                               alive_tracks=len(shown),
                               total_seen=births_cum,
                               fusions_cum=fusion_cum,
                               merges_this_frame=len(fusions)))

        if writer is not None:
            viz.draw(frame, shown, SCALE, len(fresh), fusion_cum, idx / fps,
                     contact_pairs=contacts, flash=flash, aperture=aperture)
            writer.write(frame)

        if idx % 200 == 0:
            el = _time.time() - t_start
            done = idx - f0 + 1
            print("  帧 %5d  %.3fs  气泡=%2d  累计融合=%d  用时%.1fs 剩余约%.0fs"
                  % (idx, idx / fps, len(shown), fusion_cum, el,
                     el / max(1, done) * (f1 - idx)))

    cap.release()
    if writer is not None:
        writer.release()
    _write_csvs(args.out, rows_count, rows_events, tracker)
    return 0


def _write_csvs(out, rows_count, rows_events, tracker):
    with open(os.path.join(out, "count.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time", "bubble_count",
                                          "alive_tracks", "total_seen", "fusions_cum",
                                          "merges_this_frame"])
        w.writeheader()
        w.writerows(rows_count)
    with open(os.path.join(out, "events.csv"), "w", newline="", encoding="utf-8-sig") as f:
        fn = ["type", "frame", "time", "id_a", "id_b", "new_id", "x", "y",
              "r_a", "r_b", "neck_ratio", "contact_frame", "contact_time",
              "confidence"]
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_events)
    with open(os.path.join(out, "tracks.csv"), "w", newline="", encoding="utf-8-sig") as f:
        fn = ["id", "birth_frame", "birth_x", "birth_y", "death_frame",
              "death_reason", "lifespan_frames", "max_r_px", "parent"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for t in tracker.all_tracks:
            b = t.hist[0][0] if t.hist else t.birth
            w.writerow(dict(id=t.id, birth_frame=b, birth_x=round(t.x, 1),
                            birth_y=round(t.y, 1),
                            death_frame=t.death_frame if t.death_frame else "",
                            death_reason=t.death_reason or "存活到末尾",
                            lifespan_frames=t.age,
                            max_r_px=round(t.max_r / SCALE, 1),
                            parent=("%s" % (t.parent,)) if t.parent else ""))
    life = np.array([t.age for t in tracker.all_tracks])
    if life.size:
        print("轨迹统计：共 %d 条；寿命 中位=%d 帧, p90=%d 帧, 最长=%d 帧"
              % (life.size, int(np.median(life)), int(np.percentile(life, 90)),
                 int(life.max())))
    # 尺寸-时间曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        for t in tracker.tracks:
            if len(t.hist) < 25 or t.max_r < 12:
                continue
            xs = [h[0] for h in t.hist]
            ys = [h[1] / SCALE for h in t.hist]
            ax.plot(xs, ys, lw=1.0, label="#%d" % t.id)
        ax.set_xlabel("frame"); ax.set_ylabel("equivalent radius / px (full frame)")
        ax.set_title("bubble radius vs time"); ax.grid(alpha=.3)
        if len(ax.lines) <= 12:
            ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "size_time.png"), dpi=110)
    except Exception as e:
        print("画尺寸曲线失败：%s" % e)
    print("已写出 count.csv / tracks.csv / events.csv / size_time.png 到 %s" % out)


def main():
    p = argparse.ArgumentParser(description="气泡图像识别（M0 传统方法）")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="分析一段视频")
    a.add_argument("video")
    a.add_argument("--out", default="results")
    a.add_argument("--start", type=float, default=16.3, help="起始秒（默认跳过片头）")
    a.add_argument("--end", type=float, default=80.0, help="结束秒，0 表示到片尾")
    a.add_argument("--merge-ratio", type=float, default=0.80,
                   help="判定融合完成的颈部比阈值")
    a.add_argument("--merge-hold", type=int, default=3,
                   help="需要连续满足阈值多少帧才算融合")
    a.add_argument("--max-miss", type=int, default=12,
                   help="允许连续漏检多少帧后判定气泡消失")
    a.add_argument("--no-video", action="store_true", help="不写标注视频（只出数据）")
    args = p.parse_args()
    if args.cmd == "analyze":
        if args.end == 0:
            args.end = None
        sys.exit(analyze(args))


if __name__ == "__main__":
    main()
