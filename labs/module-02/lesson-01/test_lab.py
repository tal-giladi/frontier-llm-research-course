import pytest

from frontierlab.labkit import load_target
from frontierlab.model import baseline0, toy
from frontierlab.perf import roofline as ref

lab = load_target(__file__)


def same(op, got):
    assert got[0] == pytest.approx(op.flops) and got[1] == pytest.approx(op.bytes)


def test_gemm():
    for shape in ((4, 4, 4), (8192, 2816, 768), (1, 2816, 768)):
        same(ref.gemm(*shape, bytes_per=2), lab.gemm_cost(*shape, bytes_per=2))
    same(ref.gemm(3, 5, 7, bytes_per=4), lab.gemm_cost(3, 5, 7, bytes_per=4))


def test_memory_bound_ops():
    same(ref.rmsnorm(8192, 768, 2), lab.rmsnorm_cost(8192, 768, 2))
    same(ref.softmax(96, 1024, 4), lab.softmax_cost(96, 1024, 4))


def test_decode_attention_intensity_is_group_size():
    for B, H, KV, S in ((1, 12, 4, 1024), (64, 12, 4, 32768), (2, 8, 8, 100)):
        f, b = lab.decode_attention_cost(B, H, KV, S, 64, 2)
        same(ref.decode_attention(B, H, KV, S, 64, 2), (f, b))
        assert f / b == pytest.approx(H / KV)


def test_roofline_time():
    hw = ref.HARDWARE["H100-SXM"]
    for f, b in ((35.4e9, 63e6), (25e6, 25e6), (1.0, 1.0)):
        assert lab.roofline_time(f, b, hw.peak_flops, hw.mem_bw) == pytest.approx(ref.roofline_time(f, b, hw))


def test_predict_step_time():
    hw = ref.HARDWARE["H100-SXM"]
    for cfg, B, T, ga in ((baseline0(), 32, 1024, 8), (toy(), 16, 128, 1)):
        assert lab.predict_step_time(cfg, B, T, hw.peak_flops, 0.3, ga) == pytest.approx(
            ref.predicted_step_time(cfg, B, T, hw, 0.3, ga))
