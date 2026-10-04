"""Lab 06.4, step 2: how many distinct N-grams does Data-v0 have, and how often do they share a hash bucket?

    python labs/module-06/lesson-04/collisions.py                 # first 2M training tokens, about a minute

For n = 2 and 3 and table sizes from 4,093 to 1,000,003 rows: distinct n-grams, the measured share that share
their bucket with another n-gram (one hash head), your lab's prediction under uniform hashing, and the share
that is alone in at least one of K = 2, 4, 8 independent heads (the reason Engram uses several, section 2.2).
"""

import os
import sys
from pathlib import Path

import torch

from frontierlab.blocks.engram import collision_rate
from frontierlab.data.loader import TokenData
from frontierlab.labkit import load_path

lab = load_path(str(Path(__file__).parent / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
N_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000


def main():
    ids = TokenData("train").window(0, N_TOKENS)
    print(f"Data-v0 train, first {N_TOKENS:,} tokens (vocab 8,192)")
    print(f"{'n':>2s} {'table':>9s} {'distinct':>9s} {'colliding (measured)':>21s} {'alone, 1 head (pred.)':>22s} "
          f"{'alone in >=1 of K=2/4/8 heads':>30s}")
    for n in (2, 3):
        for table in (4093, 65521, 1_000_003):
            st = collision_rate(ids, n, table)
            pred = [lab.collision_free_share(st["distinct_ngrams"], st["table"], k) for k in (1, 2, 4, 8)]
            print(f"{n:2d} {st['table']:9,d} {st['distinct_ngrams']:9,d} {st['share_colliding']:21.3f} {pred[0]:22.3f} "
                  f"{pred[1]:10.3f} {pred[2]:9.3f} {pred[3]:9.3f}")


if __name__ == "__main__":
    main()
