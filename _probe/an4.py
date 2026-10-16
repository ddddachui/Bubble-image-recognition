# -*- coding: utf-8 -*-
"""Do bubbles recur at fixed sites? Do they drift? (ASCII output + overlay png)"""
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

TH = 14.0
cents = []
FRAME = None
DET = (np.zeros(0), np.zeros(0))
for i in range(n):
    d = bg - B[i]
    d[~mv] = 0
    sm = ndi.gaussian_filter(d, sigma=8)
    p = (sm == ndi.maximum_filter(sm, size=41)) & (sm > TH)
    ys, xs = np.nonzero(p)
    cents.append(np.stack([xs, ys], 1).astype(float))
    if i == 60:
        FRAME = B[i].copy()
        DET = (xs, ys)

# ---- 1) frame-to-frame nearest-neighbour matching: bubble displacement ----
steps = []
for i in range(1, n):
    a, b = cents[i - 1], cents[i]
    if len(a) == 0 or len(b) == 0:
        continue
    D = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    j = D.argmin(1)
    m = D[np.arange(len(a)), j]
    good = m < 35
    if good.sum() >= 3:
        steps.append((b[j[good]] - a[good]).mean(0))
steps = np.array(steps)
print("matched-frame pairs: %d" % len(steps))
print("bubble displacement per 0.2 s (675x540 px): dx=%.2f+-%.2f  dy=%.2f+-%.2f"
      % (steps[:, 0].mean(), steps[:, 0].std(), steps[:, 1].mean(), steps[:, 1].std()))
print("  => approx speed at fullres(1350 wide): %.0f px/s"
      % (np.hypot(steps[:, 0].mean(), steps[:, 1].mean()) * 5 * 2))

# ---- 2) cluster all detections: do bubbles recur at the same places? ----
allpts = np.concatenate(cents) if len(cents) else np.zeros((0, 2))
centers = []
for p in allpts:
    if not centers:
        centers.append(p.copy())
        continue
    C = np.array(centers)
    k = int(np.argmin(np.linalg.norm(C - p, axis=1)))
    if np.linalg.norm(C[k] - p) < 45:
        centers[k] = 0.9 * centers[k] + 0.1 * p
    else:
        centers.append(p.copy())
centers = np.array(centers)
print("\ndetections total=%d ; distinct site clusters (r<45px) = %d ; detections/site = %.2f"
      % (len(allpts), len(centers), len(allpts) / max(1, len(centers))))
if len(centers):
    nn = np.linalg.norm(allpts[:, None, :] - centers[None, :, :], axis=2).min(1)
    print("distance to nearest site: median=%.1f px, p90=%.1f px  (image 675 wide)"
          % (np.median(nn), np.percentile(nn, 90)))

# ---- 3) overlay figure ----
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
ax[0].imshow(FRAME, cmap="gray")
ax[0].scatter(DET[0], DET[1], s=200, facecolors="none", edgecolors="lime", lw=2)
ax[0].set_title("detections, frame t=52s")
ax[1].imshow(FRAME, cmap="gray")
ax[1].scatter(allpts[:, 0], allpts[:, 1], s=6, c="red", alpha=.35, label="all detections")
ax[1].scatter(centers[:, 0], centers[:, 1], s=140, facecolors="none", edgecolors="cyan", lw=1.6, label="site clusters")
ax[1].legend(loc="upper right", fontsize=8)
ax[1].set_title("where bubbles ever appear")
plt.tight_layout()
plt.savefig(BASE + r"\overlay.png", dpi=110)
print("saved overlay.png")
