"""Small thumbnails and similarity search, used to find near-identical slices.

Only needs numpy and Pillow, so it runs quickly on the CPU without PyTorch.
"""
import hashlib
from multiprocessing import Pool

import numpy as np
from PIL import Image

SIZE = 64


def _load(path):
    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()          # fingerprint of the file, to find exact copies
    image = Image.open(path)
    image.draft("L", (SIZE * 2, SIZE * 2))               # fast JPEG decoding at a reduced size
    image = image.convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    return md5, np.asarray(image, dtype=np.uint8)


def load_thumbnails(paths, workers=4):
    """Return (md5 of each file, uint8 array of 64x64 grayscale thumbnails) for a list of image files."""
    with Pool(workers) as pool:
        results = pool.map(_load, [str(p) for p in paths], chunksize=64)
    return [r[0] for r in results], np.stack([r[1] for r in results])


def normalize(thumbs, size=SIZE):
    """Turn thumbnails (optionally shrunk to size x size) into zero-mean, unit-length vectors.

    The dot product of two such vectors is the pixel correlation of the two images (1 = identical).
    """
    n = len(thumbs)
    x = thumbs.astype(np.float32)
    if size != SIZE:
        f = SIZE // size
        x = x.reshape(n, size, f, size, f).mean(axis=(2, 4))
    x = x.reshape(n, -1)
    x -= x.mean(axis=1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x


def nearest_neighbours(x, k=10, chunk=2000):
    """For every vector, the indices and correlations of its k most similar other vectors."""
    n = len(x)
    k = min(k, n - 1)
    idx = np.zeros((n, k), np.int64)
    sim = np.zeros((n, k), np.float32)
    for start in range(0, n, chunk):
        s = x[start:start + chunk] @ x.T
        rows = np.arange(s.shape[0])
        s[rows, rows + start] = -1.0                      # ignore the image itself
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        idx[start:start + chunk] = top
        sim[start:start + chunk] = np.take_along_axis(s, top, axis=1)
    return idx, sim
