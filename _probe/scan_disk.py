# -*- coding: utf-8 -*-
r"""
磁盘占用扫描（用于排查 C 盘空间）。
用 os.scandir 手动遍历，遇到坏链接/无权限目录直接跳过，不会中断。

用法：python scan_disk.py C:\Users\10340 --big 300 --recent 14
"""
import argparse
import os
import time
from collections import defaultdict


def walk(root):
    """返回 (files, dirsize) —— dirsize 是每个子目录的字节数。"""
    files = []
    dirsize = defaultdict(int)
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            st = e.stat(follow_symlinks=False)
                            files.append((st.st_size, st.st_mtime, e.path))
                            p = os.path.dirname(e.path)
                            while p.startswith(root) and len(p) >= len(root):
                                dirsize[p] += st.st_size
                                if p == root:
                                    break
                                p = os.path.dirname(p)
                    except OSError:
                        pass
        except OSError:
            pass
    return files, dirsize


def hum(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return "%.2f %s" % (n, u)
        n /= 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--big", type=float, default=300, help="列出大于多少 MB 的文件")
    ap.add_argument("--recent", type=int, default=14, help="只列最近多少天内改动的大文件")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    t0 = time.time()
    print("扫描 %s ..." % a.root)
    files, dirsize = walk(a.root)
    total = sum(f[0] for f in files)
    print("共 %d 个文件，合计 %s，用时 %.0fs" % (len(files), hum(total), time.time() - t0))

    print("\n=== 一级子目录占用 (top %d) ===" % a.top)
    first = {}
    for p, s in dirsize.items():
        rel = os.path.relpath(p, a.root)
        k = rel.split(os.sep)[0]
        if k == ".":
            continue
        first[k] = max(first.get(k, 0), s)
    for k, s in sorted(first.items(), key=lambda x: -x[1])[:a.top]:
        print("  %12s  %s" % (hum(s), k))

    cut = time.time() - a.recent * 86400
    big = [f for f in files if f[0] >= a.big * 1024 * 1024 and f[1] >= cut]
    big.sort(key=lambda x: -x[1])
    print("\n=== 最近 %d 天内、大于 %g MB 的文件（按修改时间，最近在前，最多 25 条）===" % (a.recent, a.big))
    for sz, mt, p in big[:25]:
        print("  %10s  %s  %s" % (hum(sz), time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)), p))
    if not big:
        print("  （没有）")


if __name__ == "__main__":
    main()
