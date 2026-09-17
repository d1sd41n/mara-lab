from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.score_flux2_identity import (
    build_reference_centroid,
    cosine_similarity,
    normalize_embedding,
    portable_path,
)


def embedding(first: float, second: float = 0.0) -> np.ndarray:
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = first
    vector[1] = second
    return vector


def test_reference_centroid_normalizes_each_reference_before_averaging() -> None:
    centroid = build_reference_centroid([embedding(2.0), embedding(0.0, 4.0)])

    assert centroid[0] == pytest.approx(2**-0.5)
    assert centroid[1] == pytest.approx(2**-0.5)
    assert cosine_similarity(embedding(1.0, 1.0), centroid) == pytest.approx(1.0)


def test_normalize_embedding_rejects_invalid_vectors() -> None:
    with pytest.raises(ValueError, match="512-dimensional"):
        normalize_embedding(np.zeros(512, dtype=np.float32))

    with pytest.raises(ValueError, match="512-dimensional"):
        normalize_embedding(np.ones(511, dtype=np.float32))


def test_portable_path_prefers_a_path_relative_to_the_working_directory() -> None:
    assert portable_path(Path.cwd() / "experiments" / "run") == "experiments/run"
