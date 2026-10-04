import pytest

from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_torus_hops():
    dims = (4, 4, 4)
    assert lab.torus_hops((0, 0, 0), (3, 3, 3), dims) == 3            # one wraparound hop per axis
    assert lab.torus_hops((0, 0, 0), (3, 3, 3), dims, wrap=False) == 9
    assert lab.torus_hops((0, 0, 0), (2, 2, 2), dims) == 6
    assert max(lab.torus_hops((0, 0, 0), (x, y, z), (16, 16, 16)) for x in range(16) for y in range(16)
               for z in range(16)) == 24                               # 8 per axis in a 16^3 torus


def test_bisection_links():
    assert lab.bisection_links((4, 4, 4)) == 32
    assert lab.bisection_links((4, 4, 4), wrap=False) == 16
    assert lab.bisection_links((16, 20, 28)) == 2 * 16 * 20


def test_collective_time():
    # 1 GB array over a 4-chip ring at 45 GB/s one-way per link: (3/4) GB / 90 GB/s
    t = lab.collective_time("allgather", 1e9, 4, 4.5e10)
    assert t == pytest.approx(0.75e9 / 9e10)
    assert lab.collective_time("reducescatter", 1e9, 4, 4.5e10) == pytest.approx(t)
    assert lab.collective_time("allreduce", 1e9, 4, 4.5e10) == pytest.approx(2 * t)


def test_matmul_comm():
    assert lab.matmul_comm(("X", ""), ("", "Y")) == "none"           # A[I_X, J] B[J, K_Y]
    assert lab.matmul_comm(("", "X"), ("", "")) == "allgather"       # A[I, J_X] B[J, K]
    assert lab.matmul_comm(("", "X"), ("X", "")) == "allreduce"      # A[I, J_X] B[J_X, K]
    assert lab.matmul_comm(("X", ""), ("", "X")) == "invalid"        # A[I_X, J] B[J, K_X]
