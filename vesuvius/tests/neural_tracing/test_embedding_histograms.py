import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.embedding_histograms import cosine_groups, histogram


def test_distinct_pairs_and_presence_filter():
    case = dict(scroll='a', blended_embeddings=[[1, 0], [1, 0], [0, 1]],
                negative_cosines=[-.5, .2, .8], negative_presences=[0, .5, 1])
    groups = cosine_groups([case])['a']
    np.testing.assert_allclose(groups['positive'], [1, 0, 0])
    np.testing.assert_allclose(groups['high'], [.2, .8])
    np.testing.assert_allclose(histogram([-1, 0, 1], [-1, 0, 1]), [1/3, 2/3])
    assert histogram([], [-1, 0, 1]) is None


def test_zero_vectors_rejected():
    with pytest.raises(ValueError, match='zero GT'):
        cosine_groups([dict(scroll='a', blended_embeddings=[[0, 0], [1, 0]])])
