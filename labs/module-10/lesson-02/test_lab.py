import numpy as np
import torch

from frontierlab.datax import quality
from frontierlab.labkit import load_target

lab = load_target(__file__)

TEXTS = ["The cell divides by mitosis, then each daughter cell grows.", "", "Buy now!!! cheap cheap login",
         "Photosynthesis converts light energy into chemical energy " * 30]


def test_features_match_reference():
    for t in TEXTS:
        assert np.array_equal(lab.bucket_features(t), quality.features(t))
    assert np.array_equal(lab.bucket_features("a b", buckets=7), quality.features("a b", buckets=7))


def test_linear_score_matches_embeddingbag():
    torch.manual_seed(0)
    m = quality.BagClassifier(buckets=1 << 10)
    torch.nn.init.normal_(m.bag.weight)
    with torch.no_grad():
        m.bias.fill_(0.3)
    feats = [quality.features(t, 1 << 10) for t in TEXTS]
    ref = quality.predict(m, feats=feats)
    w = m.bag.weight.detach().numpy()[:, 0]
    for f, r in zip(feats, ref):
        assert abs(lab.linear_score(f, w, 0.3) - r) < 1e-5


def test_f1_and_top():
    pred, true = np.array([3.5, 2.0, 4.0, 3.1, 1.0]), np.array([4, 3, 5, 1, 0])
    # predicted positive {0, 2, 3}, true positive {0, 1, 2}: P = R = 2/3
    assert abs(lab.f1_at(pred, true) - 2 / 3) < 1e-12
    s = np.array([0.1, 0.9, 0.5, 0.9, 0.2])
    assert lab.keep_top(s, 0.4).tolist() == [1, 3]
    assert np.array_equal(lab.keep_top(s, 0.6), quality.top_fraction(s, 0.6))
    assert lab.keep_top(s, 0.01).tolist() == [1]


def test_epochs_and_decide():
    assert lab.epochs_needed(1.0e6, 2.5e5) == 4.0
    assert lab.decide((-0.03, -0.01)) == "adopt"
    assert lab.decide((0.002, 0.02)) == "reject"
    assert lab.decide((-0.02, 0.01)) == "inconclusive"
