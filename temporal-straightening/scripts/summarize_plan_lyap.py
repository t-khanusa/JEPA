"""Summarize logs/plan_lyap/*.jsonl: success per (model, objective), paired exact McNemar.

  python scripts/summarize_plan_lyap.py [--ref_model aggmlpcos1e-1]
"""
import argparse
import glob
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def short(run):
    return run.split("_")[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "logs" / "plan_lyap"))
    ap.add_argument("--ref_model", default="aggmlpcos1e-1")
    args = ap.parse_args()

    S = defaultdict(dict)  # (model, obj) -> {(seed, i): success}
    D = defaultdict(dict)
    TERMS = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{args.dir}/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            key = (short(r["run"]), r["objective"])
            for i, s in enumerate(r["success"]):
                S[key][(r["seed"], i)] = s
                if r["state_dist"]:
                    D[key][(r["seed"], i)] = r["state_dist"][i]
            for k, v in r["terms"].items():
                TERMS[key][k] += v

    models = sorted({m for m, _ in S}, key=lambda m: (m != args.ref_model, m))
    objs = sorted({o for _, o in S}, key=lambda o: (o != "last", o))
    print(f"{'model':18s} {'objective':18s} {'succ':>7s} {'vs last (L/W, p)':>20s} "
          f"{'vs ref-last (L/W, p)':>22s} {'last_norm':>10s} {'lyap':>8s} {'dist':>6s}")
    ref = S.get((args.ref_model, "last"), {})
    for m in models:
        base = S.get((m, "last"), {})
        for o in objs:
            cur = S.get((m, o))
            if not cur:
                continue
            ks = sorted(cur)
            succ = sum(cur[k] for k in ks)

            def pair(other):
                common = [k for k in ks if k in other]
                if not common or other is cur:
                    return "-"
                lose = sum(other[k] and not cur[k] for k in common)
                win = sum(cur[k] and not other[k] for k in common)
                return f"{lose}/{win}, p={mcnemar(lose, win):.3f}"

            t = TERMS[(m, o)]
            dist = np.mean([D[(m, o)][k] for k in ks]) if D[(m, o)] else float("nan")
            print(f"{m:18s} {o:18s} {succ:3d}/{len(ks):<3d} {pair(base):>20s} {pair(ref):>22s} "
                  f"{np.mean(t['last_norm']):10.3f} {np.mean(t['lyap']):8.4f} {dist:6.2f}")
        print()


if __name__ == "__main__":
    main()
