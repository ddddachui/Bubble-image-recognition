# -*- coding: utf-8 -*-
"""Bubble drift velocity + exposure stability (ASCII output)."""
import numpy as np
from scipy import ndimage as ndi

BASE = r"E:\Bubble image recognition\_probe"
W, H = 675, 540
raw = np.fromfile(BASE + r"\w40_70.raw", dtype=np.uint8)
n = raw.size // (W * H)
B = raw[: n * W * H].reshape(n, H, W).astype(np.float32)
bg = np.median(B, axis=0)

mv = B[0] > 45
lab, nl = ndi.label(mv)
sizes = ndi.sum(mv, lab, range(1, nl + 1))
mv = ndi.binary_erosion(lab == (np.argmax(sizes) + 1), iterations=10)

# 带通：抑制静态背景，只留气泡尺度
band = []
for i in range(n):
    d = bg - B[i]
    d[~mv] = 0
    band.append(ndi.gaussian_filter(d, 8) - ndi.gaussian_filter(d, 30))
band = np.array(band)


def pshift(a, b):
    F = np.fft.rfft2(a); G = np.fft.rfft2(b)
    R = F * np.conj(G); R /= (np.abs(R) + 1e-9)
    rr = np.fft.irfft2(R, s=a.shape)
    i = np.unravel_index(np.argmax(rr), rr.shape)
    dy = i[0] if i[0] <= a.shape[0] // 2 else i[0] - a.shape[0]
    dx = i[1] if i[1] <= a.shape[1] // 2 else i[1] - a.shape[1]
    return dx, dy, rr.max()


dxs, dys, pks = [], [], []
for i in range(1, n):
    a = band[i - 1].copy(); a -= a.mean()
    b = band[i].copy(); b -= b.mean()
    dx, dy, p = pshift(a, b)
    dxs.append(dx); dys.append(dy); pks.append(p)
dxs = np.array(dxs); dys = np.array(dys)
print("bubble-pattern inter-frame shift @5fps (675x540): dx mean=%.2f (sd %.2f), dy mean=%.2f (sd %.2f) px"
      % (dxs.mean(), dxs.std(), dys.mean(), dys.std()))
print("  => drift speed in unit-fullres (x2): dx %.1f px/s, dy %.1f px/s"
      % (dxs.mean() * 5 * 2, dys.mean() * 5 * 2))
print("  max |dy| per frame = %d px (fullres %d)" % (np.abs(dys).max(), np.abs(dys).max() * 2))

# 曝光稳定性
mb = B.reshape(n, -1).mean(1)
print("\nexposure: mean=%.1f sd=%.1f min=%.1f max=%.1f" % (mb.mean(), mb.std(), mb.min(), mb.max()))
print("frames with mean>170 (blown out): %d / %d" % ((mb > 170).sum(), n))
frac_sat = (B > 250).reshape(n, -1).mean(1) * 100
print("saturated-pixel fraction: median=%.2f%% max=%.2f%%" % (np.median(frac_sat), frac_sat.max()))
