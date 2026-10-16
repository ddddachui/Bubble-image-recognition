# -*- coding: utf-8 -*-
"""查 PyTorch 官方索引里有哪些 CUDA 通道、哪些版本支持 cp311 win64。"""
import re
import urllib.request


def versions(ch):
    url = "https://download.pytorch.org/whl/%s/torch/" % ch
    try:
        h = urllib.request.urlopen(url, timeout=45).read().decode("utf-8", "ignore")
    except Exception as e:
        return None, str(e)
    pat = re.compile(r"torch-(\d+\.\d+\.\d+)[^>]*?cp311-cp311-win_amd64\.whl")
    return sorted(set(pat.findall(h))), None


if __name__ == "__main__":
    for ch in ("cu130", "cu129", "cu128", "cu126", "cu124"):
        vs, err = versions(ch)
        if err:
            print("%-6s 失败: %s" % (ch, err))
        else:
            print("%-6s cp311-win64 可用: %s" % (ch, ", ".join(vs) if vs else "无"))
