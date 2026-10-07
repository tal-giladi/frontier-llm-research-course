import numpy as np
import torch

from frontierlab.interp import graphs as G
from frontierlab.labkit import load_target
from frontierlab.model import LM, toy

lab = load_target(__file__)


def chain():
    # nodes: 0 emb, 1 feature, 2 error, 3 logit ; edges emb->feat (2.0), feat->logit (3.0), err->logit (-1.0)
    A = np.zeros((4, 4))
    A[1, 0], A[3, 1], A[3, 2] = 2.0, 3.0, -1.0
    return A


def test_normalise_rows_by_hand():
    Ah = lab.normalise_rows(chain())
    assert np.allclose(Ah[3], [0, 0.75, 0.25, 0]) and np.allclose(Ah[1], [1, 0, 0, 0]) and np.allclose(Ah[0], 0)


def test_total_influence_by_hand():
    B = lab.total_influence(lab.normalise_rows(chain()))
    # embedding reaches the logit only through the feature: 0.75 * 1.0
    assert np.isclose(B[3, 0], 0.75) and np.isclose(B[3, 1], 0.75) and np.isclose(B[3, 2], 0.25)


def test_prune_matches_course_on_a_real_graph():
    torch.manual_seed(0)
    m = LM(toy(vocab_size=50).with_(num_hidden_layers=2)).double().eval()
    idx = torch.randint(0, 50, (1, 5))
    tcs = [G.Transcoder(128, 128, "topk", k=6, seed=l).double() for l in range(2)]
    g = G.attribute(m, idx, tcs, n_logits=2)
    w = g.logit_probs / g.logit_probs.sum()
    inf = w @ lab.total_influence(lab.normalise_rows(g.A))
    assert np.allclose(inf, G.influence(g))
    is_logit = np.array([nd[0] == "logit" for nd in g.nodes])
    for th in (0.5, 0.8, 0.95):
        assert lab.prune_nodes(inf, is_logit, th) == G.prune(g, th)


def test_feature_write_is_what_intervention_removes():
    torch.manual_seed(1)
    m = LM(toy(vocab_size=50).with_(num_hidden_layers=2)).double().eval()
    idx = torch.randint(0, 50, (1, 6))
    tc = G.Transcoder(128, 128, "topk", k=6, seed=0).double()
    tcs = [tc, G.Transcoder(128, 128, "topk", k=6, seed=1).double()]
    rec = G.trace(m, idx)
    z = tc.encode(rec["layers"][0]["mlp_in"])
    p, f = 3, int(z[0, 3].argmax())
    a = float(z[0, p, f])
    d = lab.feature_write(tc.W_dec.detach(), float(tc.out_scale), f, a)
    from frontierlab.interp import hooks as HK

    def edit(y):
        y = y.clone()
        y[:, p] = y[:, p] - d
        return y
    ours = HK.run_with(m, idx, {"mlp_out.0": edit})
    assert torch.allclose(ours, G.intervene_feature(m, idx, tcs, 0, p, f, 0.0), atol=1e-10)
