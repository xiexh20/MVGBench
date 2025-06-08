"""
some simple functions to help evaluation
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

def normalize_percentile(verts, p=10, ratio=0.5):
    "use percentile + median to compute normalization center and scale for given vertices"
    vcen = np.median(verts, axis=0)
    q1 = np.percentile(verts, p, axis=0)  # robust2
    q3 = np.percentile(verts, 100 - p, axis=0)
    obj_size = np.max(q3 - q1)
    return vcen, obj_size * ratio # multiply a ratio to prevent it from too small


def normalize_percentile_v2(verts, p=10, ratio=0.5):
    "percentile on the radius, this is usually smaller given same ratio"
    vcen = np.median(verts, axis=0)
    distances = np.sum((verts - vcen)**2, -1)
    # q1 = np.percentile(distances, p, axis=0)  # robust2
    q3 = np.percentile(distances, 100 - p, axis=0) # simply discard the outer p% points
    obj_size = np.sqrt(q3) # without sqrt, it is quite bad. with sqrt, still worse than original v1
    return vcen, obj_size * ratio


def normalize_adaptive(verts, p=10, ratio=0.5):
    "use noise density to compute normalization center and scale for given vertices"
    vcen = np.median(verts, axis=0)
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(verts)
    distances, _ = nbrs.kneighbors(verts)
    local_densities = 1 / (np.median(distances, axis=1) + 1e-6) # prevent zero
    density_median = np.median(local_densities) # robust to outliers
    density_std = np.abs(local_densities - density_median).mean() # this will be sensitive to outliers
    normalized_noise = density_std / density_median
    # mean is usually much larger than median.
    normalized_noise = np.clip(normalized_noise, 1e-6, 1.)
    q = normalized_noise * p # default 10
    q1 = np.percentile(verts, q, axis=0)  # robust2
    q3 = np.percentile(verts, 100 - q, axis=0)
    obj_size = np.max(q3 - q1) # this is usually larger than v1
    return vcen, obj_size * ratio, q