#!/usr/bin/env python3
"""Deterministic closed-loop multi-turn KV reuse benchmark.

Run against a real vLLM HTTP server; the script itself does not need CUDA.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path

import httpx

MAX_OUTPUT_TOKENS = 32
ASSISTANT_REPLY = "Acknowledged; continuing with the next check."


def messages_for_session(sid: int, lines: int, system_tag: str = "") -> list[dict[str, str]]:
    document = "\n".join(
        f"Session {sid:04d} private record {i:04d}: "
        f"asset {sid * 43 + i}; score {(sid * 137 + i * 7) % 997}; "
        "dependency tracing, risk review, latency budget, hardware resources."
        for i in range(lines)
    )
    system = "You are an infrastructure analyst. Answer concisely."
    if system_tag:
        system += " " + system_tag
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Analyze this fixed incident log:\n" + document},
    ]


def append_turn(messages: list[dict[str, str]], turn: int) -> None:
    messages.extend([
        {"role": "assistant", "content": ASSISTANT_REPLY},
        {"role": "user", "content": f"Tool check {turn}: summarize dependency group {turn}."},
    ])


def prompt_set_digest(args: argparse.Namespace) -> str:
    # Hash the exact intended message transcripts, not just the base document.
    transcripts = []
    for local_sid in range(args.sessions):
        session_id = local_sid + args.session_offset
        messages = messages_for_session(session_id, args.lines, args.system_tag)
        requests = []
        for turn in range(args.turns):
            if turn:
                append_turn(messages, turn)
            requests.append(json.loads(json.dumps(messages)))
        transcripts.append({"session": session_id, "requests": requests})
    payload = {"model": args.model, "max_tokens": getattr(args, "max_output_tokens", MAX_OUTPUT_TOKENS), "transcripts": transcripts}
    if getattr(args, "ignore_eos", False):
        payload["ignore_eos"] = True  # only when set, so historical digests are unchanged
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def timing_schedule(args: argparse.Namespace) -> dict[int, dict]:
    """Seeded per-session start delay and per-turn pauses. With the defaults, every session starts
    at once and pauses exactly --pause, which is the previous main-cohort timing."""
    schedule = {}
    hot_count = getattr(args, "hot_session_count", 0)
    # Select even local session IDs first, then odd ones. The chosen cohort is deterministic.
    hot_order = list(range(0, args.sessions, 2)) + list(range(1, args.sessions, 2))
    hot_ids = set(hot_order[:hot_count])
    for local_sid in range(args.sessions):
        sid = local_sid + args.session_offset
        rng = random.Random(f"{args.schedule_seed}-{sid}")
        start = rng.uniform(0, args.stagger_s) if args.stagger_s > 0 else 0.0
        hi = max(args.pause, args.pause_max)
        pauses = [rng.uniform(args.pause, hi) if hi > args.pause else args.pause for _ in range(args.turns - 1)]
        if hot_count:
            is_hot = local_sid in hot_ids
            factor = (getattr(args, "hot_pause_scale", 1.0) if is_hot
                      else getattr(args, "cold_pause_scale", 1.0))
            pauses = [p * factor for p in pauses]
        schedule[sid] = {"start_delay_s": round(start, 4), "pauses_s": [round(x, 4) for x in pauses]}
        if hot_count:
            schedule[sid]["is_hot"] = is_hot
    return schedule


def get_cached_tokens(usage: dict) -> int | None:
    details = usage.get("prompt_tokens_details")
    value = details.get("cached_tokens") if isinstance(details, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def get_worker_id(event: dict) -> int | None:
    # Dynamo returns the serving worker when the request sets nvext.extra_fields=["worker_id"].
    worker = (event.get("nvext") or {}).get("worker_id")
    if isinstance(worker, dict):
        worker = worker.get("decode_worker_id")
    return worker if isinstance(worker, int) and not isinstance(worker, bool) else None


async def query(client: httpx.AsyncClient, base_url: str, model: str,
                messages: list[dict[str, str]], sem: asyncio.Semaphore,
                session_id: str | None = None, request_worker_id: bool = False,
                max_output_tokens: int = MAX_OUTPUT_TOKENS, ignore_eos: bool = False) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_output_tokens,
            "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    if request_worker_id:
        body["nvext"] = {"extra_fields": ["worker_id"]}
    if ignore_eos:
        body["ignore_eos"] = True  # generate exactly max_tokens (Dynamo CommonExt)
    headers = {"X-Dynamo-Session-ID": session_id} if session_id else None
    request_ready_unix = time.time()
    async with sem:
        admitted_unix = time.time()
        start_mono = time.perf_counter()
        first_mono = None
        usage: dict = {}
        worker_id = None
        response_id = None
        response_status = None
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=body,
                                 headers=headers) as response:
            response_status = response.status_code
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if not payload:
                    continue
                event = json.loads(payload)
                if event.get("usage"):
                    usage = event["usage"]
                if (wid := get_worker_id(event)) is not None:
                    worker_id = wid
                response_id = response_id or event.get("id")
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") and first_mono is None:
                        first_mono = time.perf_counter()
        end_mono = time.perf_counter()
        end_unix = time.time()
    return {
        "request_ready_unix": request_ready_unix,
        "start_unix": admitted_unix,
        "end_unix": end_unix,
        "harness_wait_ms": round((admitted_unix - request_ready_unix) * 1000, 3),
        "ttft_ms": round((first_mono - start_mono) * 1000, 3) if first_mono is not None else None,
        "user_ttft_ms": round((admitted_unix - request_ready_unix + first_mono - start_mono) * 1000, 3) if first_mono is not None else None,
        "e2e_ms": round((end_mono - start_mono) * 1000, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": get_cached_tokens(usage),
        "http_status": response_status,
        "worker_id": worker_id,
        "response_id": response_id,
        "success": first_mono is not None,
    }


async def scrape(client: httpx.AsyncClient, url: str, out: Path) -> None:
    response = await client.get(url, timeout=30)
    response.raise_for_status()
    out.write_text(response.text)


def metric_targets(args: argparse.Namespace, phase: str) -> list[tuple[str, Path]]:
    # Default (baseline): one server, metrics_{phase}.prom. With --metrics name=url, one file per endpoint.
    if not args.metrics:
        return [(f"{args.base_url}/metrics", args.out_dir / f"metrics_{phase}.prom")]
    return [(url, args.out_dir / f"{name}_metrics_{phase}.prom")
            for name, url in (m.split("=", 1) for m in args.metrics)]


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--scenario", required=True)
    p.add_argument("--sessions", type=int, default=4)
    p.add_argument("--turns", type=int, default=12)
    p.add_argument("--lines", type=int, default=80)
    p.add_argument("--pause", type=float, default=0.5)
    p.add_argument("--max-inflight", type=int, default=2)
    p.add_argument("--session-offset", type=int, default=0)
    p.add_argument("--system-tag", default="")
    p.add_argument("--out-dir", type=Path, required=True)
    # routing (Dynamo routing). Off by default, so baseline requests are unchanged.
    p.add_argument("--route-arm", default="")
    p.add_argument("--session-header", action="store_true",
                   help="send a stable X-Dynamo-Session-ID per session on every turn")
    p.add_argument("--request-worker-id", action="store_true",
                   help="ask Dynamo to report the serving worker (nvext.extra_fields=['worker_id'])")
    p.add_argument("--metrics", action="append", default=[], metavar="NAME=URL",
                   help="metrics endpoint to snapshot before/after; repeatable")
    # Sensitivity timing (routing guide 8.3). Defaults reproduce the fixed, synchronized schedule.
    p.add_argument("--stagger-s", type=float, default=0.0,
                   help="each session starts after a seeded uniform delay in [0, STAGGER_S]")
    p.add_argument("--pause-max", type=float, default=0.0,
                   help="if > --pause, each tool pause is seeded uniform in [--pause, PAUSE_MAX]")
    p.add_argument("--schedule-seed", default="day2-sensitivity")  # Historical default retained
    p.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    p.add_argument("--ignore-eos", action="store_true",
                   help="ask the server to ignore end-of-sequence and generate exactly --max-output-tokens")
    p.add_argument("--hot-session-count", type=int, default=0,
                   help="select this many hot sessions in deterministic even-ID-first order")
    p.add_argument("--hot-pause-scale", type=float, default=1.0)
    p.add_argument("--cold-pause-scale", type=float, default=1.0)
    args = p.parse_args()
    if min(args.sessions, args.turns, args.max_inflight, args.lines) < 1:
        p.error("sessions, turns, lines, max-inflight must be positive")
    if not 0 <= args.hot_session_count <= args.sessions:
        p.error("hot-session-count must be between 0 and sessions")
    if args.max_output_tokens < 1 or args.hot_pause_scale <= 0 or args.cold_pause_scale <= 0:
        p.error("output-token cap and pause scale factors must be positive")
    if any("=" not in m for m in args.metrics):
        p.error("--metrics takes NAME=URL")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.max_inflight)
    write_lock = asyncio.Lock()
    digest = prompt_set_digest(args)
    schedule = timing_schedule(args)
    schedule_digest = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()
    rows: list[dict] = []
    started = time.time()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=httpx.Limits(max_connections=100)) as client:
        for url, out in metric_targets(args, "before"):
            await scrape(client, url, out)
        with (args.out_dir / "requests.jsonl").open("w", buffering=1) as output:
            async def run_session(local_sid: int) -> None:
                sid = local_sid + args.session_offset
                session_id = f"ailab-session-{sid:04d}" if args.session_header else None
                messages = messages_for_session(sid, args.lines, args.system_tag)
                if schedule[sid]["start_delay_s"]:
                    await asyncio.sleep(schedule[sid]["start_delay_s"])
                for turn in range(args.turns):
                    if turn:
                        append_turn(messages, turn)
                    row = {"scenario": args.scenario, "session": sid, "turn": turn,
                           "prompt_set_sha256": digest}
                    if args.hot_session_count:
                        row["hot_session"] = schedule[sid]["is_hot"]
                    if args.route_arm:
                        row["route_arm"] = args.route_arm
                    if session_id:
                        row["session_id"] = session_id
                    try:
                        row.update(await query(client, args.base_url, args.model, messages, sem,
                                               session_id, args.request_worker_id, args.max_output_tokens,
                                               args.ignore_eos))
                    except Exception as exc:
                        row.update({"success": False, "error": f"{type(exc).__name__}: {exc}",
                                    "end_unix": time.time()})
                    rows.append(row)
                    async with write_lock:
                        output.write(json.dumps(row, separators=(",", ":")) + "\n")
                        output.flush()
                    if turn < args.turns - 1:
                        await asyncio.sleep(schedule[sid]["pauses_s"][turn])
            await asyncio.gather(*(run_session(s) for s in range(args.sessions)))
        for url, out in metric_targets(args, "after"):
            await scrape(client, url, out)
    manifest = {
        "scenario": args.scenario, "model": args.model, "prompt_set_sha256": digest,
        "route_arm": args.route_arm, "session_header": args.session_header,
        "request_worker_id": args.request_worker_id, "metrics_endpoints": args.metrics,
        "sessions": args.sessions, "turns": args.turns, "lines": args.lines,
        "pause_s": args.pause, "pause_max_s": args.pause_max, "stagger_s": args.stagger_s,
        "max_output_tokens": args.max_output_tokens, "ignore_eos": args.ignore_eos,
        "hot_session_count": args.hot_session_count,
        "hot_pause_scale": args.hot_pause_scale, "cold_pause_scale": args.cold_pause_scale,
        "schedule_seed": args.schedule_seed, "schedule_sha256": schedule_digest,
        "max_inflight": args.max_inflight,
        "session_offset": args.session_offset, "system_tag": args.system_tag,
        "measured_start_unix": started, "measured_end_unix": time.time(),
        "failures": sum(not row.get("success", False) for row in rows),
        "expected_requests": args.sessions * args.turns,
        "workload": "deterministic scripted history; client-throttled closed-loop sessions; not a real agent",
    }
    (args.out_dir / "workload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if manifest["failures"] or len(rows) != manifest["expected_requests"]:
        raise SystemExit("Run had failed/missing responses; keep data but do not count as valid benchmark run")


if __name__ == "__main__":
    asyncio.run(main())
