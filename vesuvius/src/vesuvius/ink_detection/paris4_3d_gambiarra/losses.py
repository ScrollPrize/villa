"""Point losses."""
import numpy as np
import torch
from torch.nn import functional as F


def balanced_point_bce(logits, ids, labels):
    """BCE averaged per point id over its occurrences, then per class, then .5 * (no-ink + ink).
    Returns (loss, unique ids)."""
    unique, inverse = torch.unique(ids, return_inverse=True)
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
    sums = torch.zeros(len(unique), device=logits.device).index_add(0, inverse, loss)
    counts = torch.zeros_like(sums).index_add(0, inverse, torch.ones_like(loss))
    yy = torch.zeros_like(sums).index_add(0, inverse, labels.float()) / counts
    assert torch.all((yy == 0) | (yy == 1)) and (yy == 0).any() and (yy == 1).any()
    avg = sums / counts
    return .5 * (avg[yy == 0].mean() + avg[yy == 1].mean()), unique


def weak_bce(logits, ids, labels):
    """balanced_point_bce for weak sites (a class may be missing). Returns (loss, number of unique sites)."""
    unique, inverse = torch.unique(ids, return_inverse=True)
    values = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
    counts = torch.zeros(len(unique), device=logits.device).index_add(0, inverse, torch.ones_like(values))
    means = torch.zeros_like(counts).index_add(0, inverse, values) / counts
    y = torch.zeros_like(counts).index_add(0, inverse, labels.float()) / counts
    assert torch.all((y == 0) | (y == 1))
    terms = [means[y == v].mean() for v in (0, 1) if (y == v).any()]
    return .5 * sum(terms), len(unique)


def point_weights(ids, labels):
    """Per-occurrence weights of the student's click BCE: each class sums to .5, each unique point gets an equal
    share of its class, split over its occurrences. Returns (weights, unique ids)."""
    ids = np.asarray(ids, np.int64)
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    y = labels[unique]
    assert set(y) == {0, 1}
    per_id = np.where(y == 0, .5 / np.sum(y == 0), .5 / np.sum(y == 1))
    return per_id[inverse] / counts[inverse], unique
