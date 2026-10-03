"""Compare two runs' run cards and say whether they can be compared.

    python -m frontierlab.record runs/b0-s0 runs/mla-s0 --changed config.attention --axis tokens
    python -m frontierlab.record runs/b0-s0 runs/b0-s1 --replicates
Exit code 1 if any difference invalidates the comparison.
"""

import argparse
import sys

from frontierlab.record.diff import comparable, diff_cards


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--changed", nargs="*", default=[], help="dotted keys of the declared changed variable(s)")
    ap.add_argument("--axis", default="tokens", choices=["tokens", "flops", "wallclock", "params"])
    ap.add_argument("--replicates", action="store_true", help="the two runs are seed replicates")
    ap.add_argument("--all", action="store_true", help="also print ignored differences")
    a = ap.parse_args(argv)
    findings = diff_cards(a.a, a.b, changed=a.changed, axis=a.axis, seeds_are_replicates=a.replicates)
    for f in findings:
        if a.all or f.severity != "ignore":
            print(f)
    ok = comparable(findings)
    print("COMPARABLE" if ok else "NOT COMPARABLE: fix or declare the differences marked INVALIDATES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
