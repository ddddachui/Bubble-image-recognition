# -*- coding: utf-8 -*-
"""Site map + N(t) over the whole post-cut clip (ASCII output only)."""
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"E:\Bubble image recognition\_probe"
W, H = 675, 540
raw = np.fromfile(BASE + r"\whole.raw", dtype=np.uint8)
n = raw.size // (W * H)
B = raw[: n * W * H].reshape(n, H, W).astype(np.float32)
t = 16.3 + np.arange(n)
print("frames=%d  covering %.1f-%.1f s" % (n, t[0], t[-1]))

mv = B[0] > 45
lab, nl = ndi.label(mv)
if nl > 1:
    sizes = ndi.sum(mv, lab, range(1, nl + 1))
    mv = lab == (np.argmax(sizes) + 1)
mv = ndi.binary_erosion(mv, iterations=10)

bg = np.median(B, axis=0)
TH = 14.0
occ = np.zeros((H, W), np.float32)
cnt = np.zeros(n)
cyc = np.zeros(n, int)
for i in range(n):
    d = bg - B[i]
    d[~mv] = 0
    sm = ndi.gaussian_filter(d, sigma=8)
    p = (sm == ndi.maximum_filter(sm, size=41)) & (sm > TH)
    occ += p
    cnt[i] = p.sum()
occ /= n

print("N(t): min=%d median=%d max=%d mean=%.1f" % (cnt.min(), np.median(cnt), cnt.max(), cnt.mean()))
for lo in range(20, 215, 20):
    sel = (t >= lo) & (t < lo + 20)
    if sel.any():
        print("  %3d-%3ds : mean N = %5.1f  max N = %3d" % (lo, lo + 20, cnt[sel].mean(), cnt[sel].max()))

occ_s = ndi.gaussian_filter(occ, 3)
pk = (occ_s == ndi.maximum_filter(occ_s, size=45)) & (occ_s > 0.25)
ys, xs = np.nonzero(pk)
order = np.argsort(-occ_s[ys, xs])
ys, xs = ys[order], xs[order]
print("\npersistent sites (occ>0.25, 675x540 frame): %d" % len(ys))
print("top-20 occ values:", " ".join("%.2f" % v for v in occ_s[ys[:20], xs[:20]]))

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
ax[0].plot(t, cnt, lw=0.8); ax[0].set_title("N(t)  th=%.0f" % TH); ax[0].set_xlabel("t / s"); ax[0].set_ylabel("bubbles")
ax[0].grid(alpha=.3)
ax[1].imshow(occ, cmap="inferno"); ax[1].set_title("occupancy (fraction of frames a bubble is here)")
ax[1].scatter(xs, ys, s=40, facecolors="none", edgecolors="cyan")
plt.tight_layout(); plt.savefig(BASE + r"\sites.png", dpi=110)
print("saved sites.png")
