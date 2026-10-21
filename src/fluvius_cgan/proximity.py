"""Exact same-condition proximity to a fixed training reference."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _profiles(profiles, name):
    message = f"{name} profiles must be a nonempty finite real matrix"
    try:
        values = np.asarray(profiles)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if (values.ndim != 2 or not all(values.shape)
            or values.dtype.kind not in "iuf"):
        raise ValueError(message)
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(message)
    return values


def _labels(labels, length):
    message = "condition labels must be a matching vector of integers in [0, 63]"
    try:
        values = np.asarray(labels)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if (values.ndim != 1 or len(values) != length
            or values.dtype.kind not in "iuf"):
        raise ValueError(message)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or np.any(values > 63) or np.any(values != np.floor(values))):
        raise ValueError(message)
    return values.astype(np.int64)


class ConditionalNearestNeighbors:
    """Find exact RMS distances to training profiles of the same condition.

    Training profiles are copied into read-only condition groups. Inputs must
    be finite, nonempty real numeric matrices with aligned condition vectors.
    Labels may be integer arrays or exactly integral floating-point arrays;
    booleans, strings, complex values, and labels outside 0..63 are rejected.

    No normalization or dimensionality reduction is fitted here. Distances
    use the supplied profile coordinates and remain in their original units.
    """

    def __init__(self, training, labels):
        training = _profiles(training, "training")
        labels = _labels(labels, len(training))
        self.training_rows, self.profile_dim = training.shape
        self.conditions = tuple(int(label) for label in np.unique(labels))
        self._groups = {}
        for label in self.conditions:
            values = training[labels == label].copy()
            values.setflags(write=False)
            self._groups[label] = values

    @property
    def counts(self):
        """Training profile count for each supported condition."""
        return {label: len(values) for label, values in self._groups.items()}

    def distances(self, query, labels):
        """Return a float64 vector of nearest same-condition RMS distances.

        The exact squared Euclidean search uses at most 128 query rows per
        distance matrix, so singleton and batch calls use the same metric.
        Query rows and labels are never modified.

        Raises FloatingPointError if a query's nearest squared distance
        overflows, even though the supplied profile values are finite.
        """
        query = _profiles(query, "query")
        if query.shape[1] != self.profile_dim:
            raise ValueError("query profile dimension must match training")
        labels = _labels(labels, len(query))
        missing = set(labels.tolist()) - self._groups.keys()
        if missing:
            raise ValueError(f"unsupported condition labels: {sorted(missing)}")
        result = np.empty(len(query), dtype=np.float64)
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            training = self._groups[int(label)]
            for start in range(0, len(indices), 128):
                selected = indices[start:start + 128]
                squared = cdist(query[selected], training, metric="sqeuclidean")
                nearest = squared.min(axis=1)
                if not np.isfinite(nearest).all():
                    raise FloatingPointError("nearest squared distance overflowed")
                result[selected] = np.sqrt(nearest / self.profile_dim)
        return result
