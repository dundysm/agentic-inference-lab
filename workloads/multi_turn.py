#!/usr/bin/env python3
"""Deterministic closed-loop multi-turn KV reuse benchmark.

Run against a real vLLM HTTP server; the script itself does not need CUDA.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
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
    payload = {"model": args.model, "max_tokens": MAX_OUTPUT_TOKENS, "transcripts": transcripts}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def get_cached_tokens(usage: dict) -> int | None:
    details = usage.get("prompt_tokens_details")
    value = details.get("cached_tokens") if isinstance(details, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


async def query(client: httpx.AsyncClient, base_url: str, model: str,
                messages: list[dict[str, str]], sem: asyncio.Semaphore) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    request_ready_unix = time.time()
    async with sem:
        admitted_unix = time.time()
        start_mono = time.perf_counter()
        first_mono = None
        usage: dict = {}
        response_status = None
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=body) as response:
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
        "success": first_mono is not None,
    }


async def scrape(client: httpx.AsyncClient, base_url: str, out: Path) -> None:
    response = await client.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()
    out.write_text(response.text)


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
    args = p.parse_args()
    if min(args.sessions, args.turns, args.max_inflight, args.lines) < 1:
        p.error("sessions, turns, lines, max-inflight must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.max_inflight)
    write_lock = asyncio.Lock()
    digest = prompt_set_digest(args)
    rows: list[dict] = []
    started = time.time()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=httpx.Limits(max_connections=100)) as client:
        await scrape(client, args.base_url, args.out_dir / "metrics_before.prom")
        with (args.out_dir / "requests.jsonl").open("w", buffering=1) as output:
            async def run_session(local_sid: int) -> None:
                sid = local_sid + args.session_offset
                messages = messages_for_session(sid, args.lines, args.system_tag)
                for turn in range(args.turns):
                    if turn:
                        append_turn(messages, turn)
                    row = {"scenario": args.scenario, "session": sid, "turn": turn,
                           "prompt_set_sha256": digest}
                    try:
                        row.update(await query(client, args.base_url, args.model, messages, sem))
                    except Exception as exc:
                        row.update({"success": False, "error": f"{type(exc).__name__}: {exc}",
                                    "end_unix": time.time()})
                    rows.append(row)
                    async with write_lock:
                        output.write(json.dumps(row, separators=(",", ":")) + "\n")
                        output.flush()
                    if turn < args.turns - 1:
                        await asyncio.sleep(args.pause)
            await asyncio.gather(*(run_session(s) for s in range(args.sessions)))
        await scrape(client, args.base_url, args.out_dir / "metrics_after.prom")
    manifest = {
        "scenario": args.scenario, "model": args.model, "prompt_set_sha256": digest,
        "sessions": args.sessions, "turns": args.turns, "lines": args.lines,
        "pause_s": args.pause, "max_inflight": args.max_inflight,
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
