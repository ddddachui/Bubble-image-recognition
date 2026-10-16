# -*- coding: utf-8 -*-
"""临时探查脚本：量化分析 video1.mp4（不写回项目目录）"""
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"E:\Bubble image recognition\_probe"

# ---------- Part A: 2fps / 162x130 全片序列：亮度 + 光圈位置 ----------
W, H = 162, 130
raw = np.fromfile(BASE + r"\seq.raw", dtype=np.uint8)
n = raw.size // (W * H)
A = raw[: n * W * H].reshape(n, H, W).astype(np.float32)
t = np.arange(n) / 2.0
mean_b = A.reshape(n, -1).mean(1)
mask = A > 45
area = mask.reshape(n, -1).sum(1).astype(float)
cy = np.zeros(n); cx = np.zeros(n)
for i in range(n):
    if mask[i].any():
        c = ndi.center_of_mass(mask[i]); cy[i], cx[i] = c
    else:
        cy[i] = cx[i] = np.nan
r_ap = np.sqrt(area / np.pi)

print("== A. 全片 (2fps, 162x130) ==")
print("frames=%d  亮度 min/med/max = %.1f / %.1f / %.1f" % (n, mean_b.min(), np.median(mean_b), mean_b.max()))
post = t >= 16.3
print("16.28s 之前亮度中位数 = %.1f ; 之后 = %.1f" % (np.median(mean_b[~post]), np.median(mean_b[post])))
print("光圈面积 之后: min=%.0f max=%.0f (比 %.2f)" % (area[post].min(), area[post].max(), area[post].max()/area[post].min()))
print("光圈中心 之后: cx 范围 %.1f px, cy 范围 %.1f px (图宽162/高130)" %
      (np.nanmax(cx[post]) - np.nanmin(cx[post]), np.nanmax(cy[post]) - np.nanmin(cy[post])))

# ---------- Part B: 40-70s @5fps / 675x540 ----------
W2, H2 = 675, 540
raw2 = np.fromfile(BASE + r"\w40_70.raw", dtype=np.uint8)
n2 = raw2.size // (W2 * H2)
B = raw2[: n2 * W2 * H2].reshape(n2, H2, W2).astype(np.float32)
print("\n== B. 40-70s @5fps 675x540: frames=%d ==" % n2)


def phase_shift(a, b):
    Fa = np.fft.rfft2(a - a.mean())
    Fb = np.fft.rfft2(b - b.mean())
    R = Fa * np.conj(Fb)
    R /= (np.abs(R) + 1e-9)
    rr = np.fft.irfft2(R, s=a.shape)
    idx = np.unravel_index(np.argmax(rr), rr.shape)
    dy = idx[0] if idx[0] <= a.shape[0] // 2 else idx[0] - a.shape[0]
    dx = idx[1] if idx[1] <= a.shape[1] // 2 else idx[1] - a.shape[1]
    return dx, dy, rr.max()


sx = np.zeros(n2); sy = np.zeros(n2); pk = np.zeros(n2)
for i in range(1, n2):
    sx[i], sy[i], pk[i] = phase_shift(B[i - 1], B[i])
print("相邻帧相位相关位移: |dx|<=%.1f px, |dy|<=%.1f px, 累计 dx=%.1f dy=%.1f ; 峰值中位=%.3f"
      % (np.abs(sx).max(), np.abs(sy).max(), sx.sum(), sy.sum(), np.median(pk[1:])))

bg = np.median(B, axis=0)
valid = ndi.binary_erosion(A.any(0) | True, iterations=0)  # placeholder
# 有效区（光圈内）：用面积最大的亮区估计，并对 675x540 做同样处理
mv = (B[0] > 45)
lab, nl = ndi.label(mv)
if nl > 1:
    sizes = ndi.sum(mv, lab, range(1, nl + 1))
    mv = lab == (np.argmax(sizes) + 1)
mv = ndi.binary_erosion(mv, iterations=6)

print("\n每帧气泡候选数（不同阈值）:")
print(" frame  t(s) " + "".join("%7d" % th for th in (6, 10, 14, 18, 24, 30)))
diam_all = []
cnt_curve = []
for i in range(0, n2, 10):
    d = bg - B[i]
    d[~mv] = 0
    sm = ndi.gaussian_filter(d, sigma=8)
    row = []
    base_cnt = None
    for th in (6, 10, 14, 18, 24, 30):
        p = (sm == ndi.maximum_filter(sm, size=41)) & (sm > th)
        row.append(int(p.sum()))
    print("%6d %5.1f " % (i, 40 + i / 5.0) + "".join("%7d" % v for v in row))

# 用固定阈值做 N(t) 曲线 + 尺寸统计
TH = 14
cnt_curve = np.zeros(n2)
for i in range(n2):
    d = bg - B[i]
    d[~mv] = 0
    sm = ndi.gaussian_filter(d, sigma=8)
    p = (sm == ndi.maximum_filter(sm, size=41)) & (sm > TH)
    cnt_curve[i] = p.sum()
    if i % 25 == 0:
        ys, xs = np.nonzero(p)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 60), min(H2, y + 60)
            x0, x1 = max(0, x - 60), min(W2, x + 60)
            patch = sm[y0:y1, x0:x1]
            half = sm[y, x] / 2.0
            ar = (patch > half).sum()
            if ar > 0:
                diam_all.append(2 * np.sqrt(ar / np.pi))
diam_all = np.array(diam_all)
print("\n阈值%d: 计数 min/med/max = %d / %d / %d" % (TH, cnt_curve.min(), np.median(cnt_curve), cnt_curve.max()))
if diam_all.size:
    print("待检目标等效直径(px, 半高宽): p10=%.0f 中位=%.0f p90=%.0f (原图1350宽下约为其2倍)"
          % (np.percentile(diam_all, 10), np.median(diam_all), np.percentile(diam_all, 90)))

fig, ax = plt.subplots(3, 1, figsize=(11, 9))
ax[0].plot(t, mean_b); ax[0].axvline(16.28, color="r", ls="--"); ax[0].set_title("mean brightness (2fps, whole clip)"); ax[0].set_xlabel("t / s")
ax[1].plot(40 + np.arange(n2) / 5.0, cnt_curve); ax[1].set_title("candidate count N(t), th=%d, 40-70s" % TH); ax[1].set_xlabel("t / s")
ax[2].plot(40 + np.arange(n2) / 5.0, np.concatenate([[0], np.cumsum(np.abs(sx[1:]) + np.abs(sy[1:]))]))
ax[2].set_title("cumulative |global shift| (phase corr)"); ax[2].set_xlabel("t / s")
plt.tight_layout(); plt.savefig(BASE + r"\stats.png", dpi=110)
print("\n已保存 stats.png")
