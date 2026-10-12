#!/usr/bin/env python3
"""Join a debug-logged KV-router run to its requests: router belief vs actual reuse, per decision.

Needs a run made with DYN_LOG=info,dynamo_kv_router=debug and a client that records response_id.
Usage: python scripts/routing_diagnose.py RUN_DIR [--examples N]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as s
from collections import defaultdict
from pathlib import Path

FORMULA = re.compile(
    r"Formula for worker_id=(\d+) dp_rank=\d+ with ([\d.]+) effective cached blocks: ([\d.]+) = .*?"
    r"= ([\d.]+) \* ([\d.]+) \+ ([\d.]+) \+ ([\d.]+) \(raw_prefill_blocks: ([\d.]+), "
    r"overlap_credit_blocks: ([\d.]+), overlap_credit_decay: ([\d.]+)\) request_id=\"([0-9a-f-]+)\"")
FIELDS = ("cached", "logit", "scale", "adj_prefill", "decode", "active", "raw_prefill", "credit", "decay")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--examples", type=int, default=4)
    a = p.parse_args()
    wmap = {int(k): v for k, v in json.load((a.run / "worker_map.json").open()).items()}
    log = re.sub(r"\x1b\[[0-9;]*m", "", (a.run / "frontend.log").read_text())
    form: dict[str, dict[int, dict]] = defaultdict(dict)
    for m in FORMULA.finditer(log):
        form[m.group(11)][wmap[int(m.group(1))]] = dict(zip(FIELDS, map(float, m.groups()[1:10])))
    rows = [json.loads(l) for l in (a.run / "requests.jsonl").open()]
    rid = lambda r: (r.get("response_id") or "").replace("chatcmpl-", "")
    print(f"decisions logged: {len(form)}; requests joined: {sum(rid(r) in form for r in rows)}/{len(rows)}")

    by = defaultdict(dict)
    for r in rows:
        by[r["session"]][r["turn"]] = r
    moved, stayed, shown = [], [], 0
    for sid, turns in sorted(by.items()):
        for t in range(1, len(turns)):
            prev, cur = turns[t - 1], turns[t]
            f = form.get(rid(cur))
            if not f:
                continue
            pw, cw = wmap[prev["worker_id"]], wmap[cur["worker_id"]]
            rec = dict(prev=f[pw], served=f[cw], actual_blocks=cur["cached_tokens"] / 64, prompt_blocks=cur["prompt_tokens"] / 64)
            (moved if pw != cw else stayed).append(rec)
            if pw != cw and shown < a.examples:
                shown += 1
                print(f"\nsession {sid} turn {t}: moved w{pw} -> w{cw}; prompt {cur['prompt_tokens']} tokens, "
                      f"actually cached {cur['cached_tokens']}")
                for w in (pw, cw):
                    x = f[w]
                    print(f"  w{w}: believed cached blocks {x['cached']:.0f}, overlap credit {x['credit']:.2f}, "
                          f"raw prefill {x['raw_prefill']:.2f}, adjusted prefill {x['adj_prefill']:.2f}, "
                          f"decode {x['decode']:.2f}, active {x['active']:.2f} -> logit {x['logit']:.2f}")

    def show(name, xs):
        print(f"\n{name}: n={len(xs)}")
        if xs:
            print(f"  router-believed cached blocks: previous worker {s.median(x['prev']['cached'] for x in xs):.1f}, "
                  f"serving worker {s.median(x['served']['cached'] for x in xs):.1f}; "
                  f"actually cached on serving worker {s.median(x['actual_blocks'] for x in xs):.1f} "
                  f"of ~{s.median(x['prompt_blocks'] for x in xs):.0f} blocks")
    show("stayed on same worker", stayed)
    show("moved to other worker", moved)
    if moved:
        prefix_left = sum(x["prev"]["cached"] >= 0.5 * x["prompt_blocks"] for x in moved)
        stale = sum(x["served"]["cached"] >= 0.5 * x["prompt_blocks"] and x["actual_blocks"] < 0.5 * x["prompt_blocks"] for x in moved)
        print(f"  moves away from a worker the router believed still held >=50% of the prefix: {prefix_left}/{len(moved)}")
        print(f"  moves where the router believed the destination held the prefix but it missed (stale index): {stale}/{len(moved)}")


if __name__ == "__main__":
    main()
