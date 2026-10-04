"""Reference solution for lab 05.3."""

from __future__ import annotations

import math


def fit_exponent(contexts, times):
    xs = [math.log(c) for c in contexts]
    ys = [math.log(t) for t in times]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def crossover(contexts, t_base, t_new):
    r = [math.log(n / b) for n, b in zip(t_new, t_base)]         # < 0 where the new method is faster
    if r[0] < 0:
        return contexts[0]
    for i in range(1, len(r)):
        if r[i] < 0:
            x0, x1 = math.log(contexts[i - 1]), math.log(contexts[i])
            x = x0 + (x1 - x0) * r[i - 1] / (r[i - 1] - r[i])      # linear in log-log: where r crosses 0
            return math.exp(x)
    return None


def project_time(flops, nbytes, peak, bandwidth, mfu, bw_eff):
    return max(flops / (peak * mfu), nbytes / (bandwidth * bw_eff))


def decide(speedup_ci, loss_diff_ci, min_speedup, margin):
    if speedup_ci[0] >= min_speedup and loss_diff_ci[1] <= margin:
        return "adopt"
    if speedup_ci[1] < min_speedup or loss_diff_ci[0] > margin:
        return "keep dense"
    return "inconclusive"
