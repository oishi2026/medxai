"""The SHARED region layer — the structural core of the region-coherent protocol.

One SLIC partition per image is reused by BOTH paradigms:
  * CNN post-hoc saliency (Grad-CAM++, IG) is aggregated onto these regions, and
  * the GNN uses these same regions as graph nodes.
Because both are then scored with the SAME region-level insertion/deletion on the
SAME regions, "CNN+XAI vs GNN-attention" becomes an apples-to-apples faithfulness
question — which no existing chest-XAI benchmark does.

Partition parameters (n_segments, compactness) come from conf/frozen.yaml so the
partition is identical across methods, images, and teammates.
"""
from __future__ import annotations

import numpy as np
from skimage.graph import RAG
from skimage.segmentation import slic


def compute_superpixels(
    image: np.ndarray, n_segments: int, compactness: float, sigma: float = 1.0
) -> np.ndarray:
    """SLIC partition. `image` is HxWx3 float in [0,1]. Returns an HxW int label
    map with contiguous ids 0..K-1."""
    labels = slic(
        image, n_segments=n_segments, compactness=compactness, sigma=sigma,
        start_label=0, channel_axis=-1,
    )
    return _remap_contiguous(labels)


def _remap_contiguous(labels: np.ndarray) -> np.ndarray:
    """Ensure region ids are 0..K-1 with no gaps (SLIC can occasionally skip)."""
    uniq, inv = np.unique(labels, return_inverse=True)
    return inv.reshape(labels.shape).astype(np.int64)


def n_regions(labels: np.ndarray) -> int:
    return int(labels.max()) + 1


def build_rag(labels: np.ndarray):
    """Region adjacency graph. Returns (edges, centroids, sizes):
      edges:     (E, 2) int, undirected pairs with i<j
      centroids: (K, 2) float, (row, col) per region — for edge/geometry features
      sizes:     (K,)  int, pixel count per region
    """
    rag = RAG(labels)
    edges = np.array(
        sorted({(min(u, v), max(u, v)) for u, v in rag.edges}), dtype=np.int64
    ) if rag.number_of_edges() else np.zeros((0, 2), dtype=np.int64)

    k = n_regions(labels)
    sizes = np.bincount(labels.ravel(), minlength=k)
    rows, cols = np.indices(labels.shape)
    cr = np.bincount(labels.ravel(), weights=rows.ravel(), minlength=k) / np.maximum(sizes, 1)
    cc = np.bincount(labels.ravel(), weights=cols.ravel(), minlength=k) / np.maximum(sizes, 1)
    centroids = np.stack([cr, cc], axis=1)
    return edges, centroids, sizes.astype(np.int64)


def aggregate_map_to_regions(smap: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean of a saliency map within each region -> (K,) per-region importance.
    This is how a pixel-grid CNN saliency map is projected onto the shared
    regions so it can be compared to native GNN region attention."""
    k = n_regions(labels)
    flat_l = labels.ravel()
    sums = np.bincount(flat_l, weights=smap.ravel().astype(np.float64), minlength=k)
    counts = np.bincount(flat_l, minlength=k)
    return (sums / np.maximum(counts, 1)).astype(np.float32)


def regions_to_map(region_values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Paint each region with its scalar value -> HxW map. Used for visualization
    and for region-level insertion/deletion masking."""
    return np.asarray(region_values, dtype=np.float32)[labels]


def edge_features(edges: np.ndarray, centroids: np.ndarray, labels: np.ndarray):
    """Simple geometric edge features for the GNN: normalized centroid distance.
    Returns (E,) float in image-diagonal units."""
    if len(edges) == 0:
        return np.zeros((0,), dtype=np.float32)
    d = centroids[edges[:, 0]] - centroids[edges[:, 1]]
    dist = np.sqrt((d ** 2).sum(axis=1))
    diag = np.sqrt(labels.shape[0] ** 2 + labels.shape[1] ** 2)
    return (dist / diag).astype(np.float32)
