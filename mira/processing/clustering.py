"""DBSCAN, so one car with eight bright spots becomes one object.

Written out instead of importing scikit-learn: a radar frame has tens to a
few hundred points, so the O(n^2) distance matrix is nothing.
"""

from __future__ import annotations

import numpy as np


def dbscan(features: np.ndarray, eps: float, min_samples: int = 2) -> np.ndarray:
    """Cluster labels for each row of features, -1 for noise.

    Scale the feature columns beforehand so that eps means the same thing in
    every dimension (for example x and y in metres, velocity divided by a
    m/s-per-metre factor).
    """
    n = len(features)
    labels = np.full(n, -1, int)
    if n == 0:
        return labels
    d2 = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=-1)
    neighbours = d2 <= eps * eps
    core = neighbours.sum(axis=1) >= min_samples
    cluster = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cluster
        stack = [i]
        while stack:
            j = stack.pop()
            if not core[j]:
                continue
            for k in np.nonzero(neighbours[j])[0]:
                if labels[k] == -1:
                    labels[k] = cluster
                    stack.append(k)
        cluster += 1
    return labels
