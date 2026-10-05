from frontierlab.datax import rephrase
from frontierlab.labkit import load_target

lab = load_target(__file__)

SRC = "In 1999 the council approved 40,000 dollars for 3 new libraries in Springfield."
OUT = "Springfield's council approved $40000 in 1999 to build 4 libraries, said Mayor Lee in 2001."


def test_generation_flops():
    assert lab.generation_flops(10, 1, 2, 3, 2) == 192
    for args in ((440_000_000, 28, 2048, 300, 250, 2 * 151936 * 1024), (1, 2, 3, 0, 5), (7, 1, 1, 9, 0)):
        assert abs(lab.generation_flops(*args) - rephrase.generation_flops(*args)) < 1e-6


def test_number_check():
    kept, invented = lab.number_check(SRC, OUT)
    assert abs(kept - 2 / 3) < 1e-12 and invented == 2       # 3 lost; 4 and 2001 invented
    assert lab.number_check("no numbers here", "1 2") == (None, 2)


def test_novelty_and_diversity():
    assert lab.novel_ngram_share(SRC, SRC) == 0.0
    assert lab.novel_ngram_share(SRC, "too short") is None
    f = rephrase.fidelity(SRC, OUT)
    assert abs(lab.novel_ngram_share(SRC, OUT) - f["novel_4grams"]) < 1e-12
    outs = ["the cat sat", "the cat ran", "a dog sat"]
    assert abs(lab.distinct_n(outs, 2) - 5 / 6) < 1e-12
    assert lab.distinct_n([], 2) == 0.0


def test_cost_and_decide():
    assert lab.cost_ratio(5e13, 1e13) == 5.0
    assert lab.decide((-0.03, -0.001)) == "adopt" and lab.decide((0.001, 0.01)) == "reject"
    assert lab.decide((-0.01, 0.01)) == "inconclusive"
