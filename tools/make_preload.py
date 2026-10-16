# -*- coding: utf-8 -*-
"""
把上一轮标注预填进新的标注工具（时间点相同的帧直接复用）。

用法: python tools/make_preload.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "标注工具")


def main():
    old = os.path.join(TOOL, "bubble_annotation_第1轮_16帧.json")
    if not os.path.exists(old):
        print("找不到上一轮标注：%s" % old)
        return
    d = json.load(open(old, encoding="utf-8"))
    pre = {}
    for a in d["annotated"]:
        key = "f%05.1f" % a["time"]
        pre[key] = {"bubbles": a.get("bubbles", []),
                    "ghosts": a.get("ghosts", [])}
    out = os.path.join(TOOL, "preload.js")
    with open(out, "w", encoding="utf-8") as f:
        f.write("// 自动生成：来自上一轮标注，时间点相同的帧会预填\n")
        f.write("var PRELOAD = " + json.dumps(pre, ensure_ascii=False) + ";\n")
    print("已生成 %s，预填 %d 帧：%s" % (out, len(pre), ", ".join(sorted(pre))))


if __name__ == "__main__":
    main()
