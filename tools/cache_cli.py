#!/usr/bin/env python3
import argparse
from tools.cache import Cache

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("stats"); ps.add_argument("--ns", default="default"); ps.add_argument("--maxgb", type=float, default=2.0)
    pp = sub.add_parser("prune"); pp.add_argument("--ns", default="default"); pp.add_argument("--targetgb", type=float, required=True)
    sub.add_parser("vacuum")
    a = p.parse_args()
    if a.cmd == "stats":
        with Cache(namespace=a.ns, max_bytes=int(a.maxgb*1e9)) as c:
            cnt, tot = c.stats(); print(f"[cache:{a.ns}] items={cnt}, size={tot/1e6:.1f}MB")
    elif a.cmd == "prune":
        with Cache(namespace=a.ns) as c:
            freed = c.prune(int(a.targetgb*1e9)); print(f"[cache:{a.ns}] freed={freed/1e6:.1f}MB")
    elif a.cmd == "vacuum":
        with Cache(): pass; print("[cache] vacuum/pragma done")

if __name__ == "__main__": main()
