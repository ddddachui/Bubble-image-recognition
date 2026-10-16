# -*- coding: utf-8 -*-
"""读取并合并人工标注（支持多轮标注文件）。"""
import glob
import json
import os


def load_all(tool_dir, prefer="第2轮"):
    """
    读取 tool_dir 下所有 bubble_annotation*.json 并合并。
    同一个时间点出现多次时，优先用文件名里含 prefer 的那份（默认第 2 轮）。
    返回 {time: {"bubbles":[...], "ghosts":[...]}}
    """
    files = sorted(glob.glob(os.path.join(tool_dir, "bubble_annotation*.json")))
    files = [f for f in files if "第1轮" not in os.path.basename(f) or len(files) == 1]
    merged = {}
    order = []
    for f in files:
        prio = 1 if prefer in os.path.basename(f) else 0
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for a in d.get("annotated", []):
            t = round(float(a["time"]), 1)
            cur = merged.get(t)
            if cur is None or prio >= cur["_prio"]:
                merged[t] = {"bubbles": a.get("bubbles", []),
                             "ghosts": a.get("ghosts", []),
                             "_prio": prio,
                             "_src": os.path.basename(f)}
            if t not in order:
                order.append(t)
    return {t: merged[t] for t in sorted(order)}


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ann = load_all(os.path.join(root, "标注工具"))
    nb = sum(len(v["bubbles"]) for v in ann.values())
    ng = sum(len(v["ghosts"]) for v in ann.values())
    print("合并后：%d 帧，气泡点 %d，虚影点 %d" % (len(ann), nb, ng))
    print("时间点：%s" % ", ".join("%.0f" % t for t in ann))
