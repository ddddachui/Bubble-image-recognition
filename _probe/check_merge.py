# -*- coding: utf-8 -*-
"""在指定时间段跑一遍流水线，导出标注帧，用于人工核对检测/跟踪质量。"""
import os
import sys
import numpy as np
import cv2

ROOT = r"E:\Bubble image recognition"
sys.path.insert(0, ROOT)
from src import segment as S, viz                      # noqa: E402
from src.track import Tracker                          # noqa: E402
from src.events import FusionDetector                  # noqa: E402

VIDEO = os.path.join(ROOT, "video1.mp4")
OUT = os.path.join(ROOT, "_probe")
SCALE = 0.5
CFG = dict(med_k=151, lo=0.030, hi=0.140, open_r=6, close_r=17,
           min_r=17.0, min_sep=15, prominence=0.70, max_r=62.0, merge_ratio=0.88)


def main():
    t0s, t1s = float(sys.argv[1]), float(sys.argv[2])
    dump = [float(x) for x in sys.argv[3:]] or []
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    f0, f1 = int(t0s * fps), int(t1s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    tr = Tracker(max_miss=40, gate_scale=0.5, gate_min=12.0)
    fd = FusionDetector(ratio_hi=0.78, hold=3)
    ap = None
    shots = []
    for idx in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if ap is None:
            ap = S.aperture_mask(g)
        rel, fill, dsm, dets = S.detect(g, ap, CFG)
        tracks = tr.update(idx, dets)
        fusions, contacts = fd.step(idx, tracks, dsm)
        alive = [t for t in tr.tracks if not t.dead and (t.confirmed or t.hits >= 3)]
        for fu in fusions:
            for tid in (fu["id_a"], fu["id_b"]):
                for t in tr.tracks:
                    if t.id == tid and not t.dead:
                        t.kill(idx, "融合")
            rn = float(np.sqrt(fu["r_a"] ** 2 + fu["r_b"] ** 2))
            tr.spawn_merged(idx, fu["x"], fu["y"], rn, (fu["id_a"], fu["id_b"]))
            tr.add_suppression(idx, fu["x"], fu["y"], rn, 25)
        tm = idx / fps
        if any(abs(tm - d) < 0.5 / fps for d in dump):
            v = frame.copy()
            viz.draw(v, alive, SCALE, len(alive), 0, tm,
                     contact_pairs=contacts, aperture=ap)
            v = cv2.resize(v, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA)
            cv2.putText(v, "%.2fs  n=%d" % (tm, len(alive)), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            shots.append(v)
            print("dump t=%.2f  气泡=%d  接触对=%d" % (tm, len(alive), len(contacts)))
    if shots:
        rows = []
        ncol = len(shots)
        cv2.imwrite(os.path.join(OUT, "check_merge.png"), np.hstack(shots))
        print("saved check_merge.png (%d 帧并排)" % ncol)
    cap.release()


if __name__ == "__main__":
    main()
