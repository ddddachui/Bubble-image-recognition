# -*- coding: utf-8 -*-
"""Permutation test: do bubbles recur at the same places? (ASCII output)"""
import numpy as np
from scipy import ndimage as ndi

EPS = 1e-9
BASE = r"E:\Bubble image recognition\_probe"
W, H = 675, 540


def detections(path, step=1):
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // (W * H)
    B = raw[: n * W * H].reshape(n, H, W).astype(np.float32)
    bg = np.median(B, axis=0)
    mv = B[0] > 45
    lab, nl = ndi.label(mv)
    sizes = ndi.sum(mv, lab, range(1, nl + 1))
    mv = ndi.binary_erosion(lab == (np.argmax(sizes) + 1), iterations=10)
    out = []
    for i in range(0, n, step):
        d = bg - B[i]
        d[~mv] = 0
        sm = ndi.gaussian_filter(d, sigma=8)
        p = (sm == ndi.maximum_filter(sm, size=41)) & (sm > 14)
        ys, xs = np.nonzero(p)
        out.append(np.stack([xs, ys], 1).astype(float))
    return out


def mean_nn_to_other_frame(cents):
    """mean distance from each detection to the nearest detection in a DIFFERENT frame"""
    P = np.concatenate(cents)
    fid = np.concatenate([np.full(len(c), i) for i, c in enumerate(cents)])
    rng = np.random.default_rng(0)
    # sample to keep it fast
    idx = rng.choice(len(P), size=min(400, len(P)), replace=False)
    tot = 0.0
    for k in idx:
        D = np.linalg.norm(P - P[k], axis=1)
        D[fid == fid[k]] = 1e9
        tot += D.min()
    return tot / len(idx)


for tag, path in (("40-70s@5fps", r"\w40_70.raw"), ("whole 16-210s@1fps", r"\whole.raw")):
    cents = detections(BASE + path)
    tot = sum(len(c) for c in cents)
    obs = mean_nn_to_other_frame(cents)
    # null: random uniform points inside the aperture
    rng = np.random.default_rng(1)
    P = np.concatenate(cents)
    null = []
    for _ in range(5):
        Q = np.stack([rng.uniform(P[:, 0].min(), P[:, 0].max(), len(P)),
                      rng.uniform(P[:, 1].min(), P[:, 1].max(), len(P))], 1)
        ks = rng.choice(len(Q), size=min(400, len(Q)), replace=False)
        s = 0.0
        for k in ks:
            D = np.linalg.norm(Q - Q[k], axis=1)
            D[k] = 1e9
            s += D.min()
        null.append(s / len(ks))
    print("%s: %d detections in %d frames" % (tag, tot, len(cents)))
    print("   observed mean dist to nearest detection in another frame = %.1f px" % obs)
    print("   random-uniform null                                        = %.1f +- %.1f px"
          % (np.mean(null), np.std(null)))
    print("   ratio obs/null = %.2f  (<0.6 => strong fixed-site recurrence)" % (obs / np.mean(null)))
