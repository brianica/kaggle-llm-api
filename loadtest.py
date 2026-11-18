#!/usr/bin/env python3
"""
Concurrency load test for an OpenAI-compatible chat endpoint (vLLM, llama.cpp, or this repo's gateway).

It ramps through concurrency levels. At each level, N clients send requests back to back
(each sends its next request as soon as the last one finishes), and it reports what a coding agent feels:

  * success rate and error types
  * TTFT        time to first streamed token (how long an agent waits before anything happens)
  * latency     full request time
  * tok/s/user  streaming speed each client sees
  * total tok/s throughput of the whole server

Agent-like workloads:
  --prompt-tokens 6000     send a long, code-heavy prompt (like an agent's repo context)
  --shared-prefix          every request shares the same long prefix (tests prefix caching, like
                           one agent's multi-turn session); without it each prefix is unique (worst case)
  --tools                  attach tool definitions and check that the model answers with a tool call

Examples:
  pip install aiohttp
  export OPENAI_API_KEY=sk-...
  python loadtest.py --base-url https://xxxx.ngrok-free.app/v1
  python loadtest.py --base-url https://xxxx.ngrok-free.app/v1 -c 1,8,32,64 --prompt-tokens 6000 --shared-prefix
  python loadtest.py --base-url https://xxxx.ngrok-free.app/v1 --tools --json results.json

Inside the notebook (skips the tunnel, so you measure the GPU alone):
  !python loadtest.py --base-url http://127.0.0.1:8000/v1 --api-key "$API_KEY"
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict

try:
    import aiohttp
except ImportError:
    sys.exit("This script needs aiohttp: pip install aiohttp")


# ----------------------------------------------------------------------------- prompts

TASKS = [
    "Write a Python function that parses an ISO-8601 duration string into seconds. Include a docstring and tests.",
    "Refactor the code above to remove duplication and explain each change briefly.",
    "Find the most likely bug in the code above and propose a minimal fix as a unified diff.",
    "Write a TypeScript debounce function with proper generic types and a short usage example.",
    "Implement an LRU cache class in Python with O(1) get and put, and explain the design.",
    "Write a SQL query that returns each customer's three most recent orders, with an explanation.",
    "Add type hints and error handling to the code above. Return only the updated code.",
    "Write a bash script that finds the 10 largest files under a directory, excluding .git.",
]

CODE_LINES = [
    "def {f}(items, key=None):",
    "    result = []",
    "    for i, item in enumerate(items):",
    "        if key is not None and key(item) in seen:",
    "            continue",
    "        result.append(transform_{f}(item, index=i))",
    "    return sorted(result, key=lambda r: r.priority)",
    "class {C}Repository:",
    "    def __init__(self, db, cache_ttl=300):",
    "        self.db = db",
    "        self._cache = {{}}",
    "    async def get_{f}(self, id_: int) -> dict | None:",
    "        row = await self.db.fetch_one('SELECT * FROM {f} WHERE id = $1', id_)",
    "        return dict(row) if row else None",
    "# TODO({f}): handle pagination when the upstream API returns a cursor",
    "export function {f}(input: string): Result<{C}, ParseError> {{",
    "  const parts = input.split(',').map((p) => p.trim());",
    "  return parts.length ? ok(build{C}(parts)) : err(new ParseError(input));",
    "}}",
]


def make_context(approx_tokens: int, seed: int) -> str:
    """Fake repository context of roughly `approx_tokens` tokens (about 3.5 characters per token for code)."""
    if approx_tokens <= 0:
        return ""
    rnd = random.Random(seed)
    names = ["user", "order", "invoice", "session", "token", "widget", "report", "payload", "event", "job"]
    out, chars, target = [], 0, int(approx_tokens * 3.5)
    file_no = 0
    while chars < target:
        file_no += 1
        f = rnd.choice(names) + str(rnd.randint(1, 999))
        header = f"\n### File: src/{f}.py (seed {seed})\n"
        out.append(header)
        chars += len(header)
        for _ in range(rnd.randint(15, 40)):
            line = rnd.choice(CODE_LINES).format(f=f, C=f.capitalize()) + "\n"
            out.append(line)
            chars += len(line)
    return "".join(out)[:target]


TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the repository and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
TOOL_TASKS = [
    "Open src/main.py so you can review it. Use a tool.",
    "Run the test suite with pytest. Use a tool.",
    "Show me what's in README.md. Use a tool.",
    "List the files in the repository root. Use a tool.",
]


def build_messages(args, req_id: int, shared_context: str) -> list:
    if args.tools:
        task = TOOL_TASKS[req_id % len(TOOL_TASKS)]
    else:
        task = TASKS[req_id % len(TASKS)]
    if args.prompt_tokens > 0:
        context = shared_context if args.shared_prefix else make_context(args.prompt_tokens, seed=req_id + 1)
        content = f"Here is part of the repository:\n{context}\n\nTask #{req_id}: {task}"
    else:
        content = f"Task #{req_id}: {task}"
    return [
        {"role": "system", "content": "You are a senior software engineer working in the user's repository."},
        {"role": "user", "content": content},
    ]


# ----------------------------------------------------------------------------- one request

@dataclass
class Result:
    ok: bool
    error: str = ""
    ttft: float | None = None          # seconds to first token
    latency: float = 0.0               # seconds for the whole request
    out_tokens: int = 0
    prompt_tokens: int = 0
    tokens_estimated: bool = False     # True when the server sent no usage and chunks were counted
    tool_called: bool | None = None


async def one_request(session, args, req_id, shared_context) -> Result:
    body = {
        "model": args.model,
        "messages": build_messages(args, req_id, shared_context),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": not args.no_stream,
    }
    if not args.no_stream:
        body["stream_options"] = {"include_usage": True}
    if args.tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    if args.ignore_eos:
        body["ignore_eos"] = True      # vLLM extension: always generate max_tokens (steadier comparisons)

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    t0 = time.perf_counter()
    try:
        async with session.post(f"{args.base_url}/chat/completions", json=body, headers=headers) as resp:
            if resp.status != 200:
                text = (await resp.text())[:200].replace("\n", " ")
                return Result(False, f"HTTP {resp.status}: {text}", latency=time.perf_counter() - t0)

            if args.no_stream:
                data = await resp.json()
                lat = time.perf_counter() - t0
                usage = data.get("usage") or {}
                msg = (data.get("choices") or [{}])[0].get("message") or {}
                return Result(True, ttft=None, latency=lat,
                              out_tokens=usage.get("completion_tokens", 0),
                              prompt_tokens=usage.get("prompt_tokens", 0),
                              tool_called=bool(msg.get("tool_calls")) if args.tools else None)

            ttft, chunks, usage, tool_called, buf = None, 0, None, False, b""
            async for raw in resp.content:
                buf += raw
                if not buf.endswith(b"\n"):
                    continue
                line, buf = buf.decode("utf-8", "replace").strip(), b""
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ev.get("error"):
                    return Result(False, f"stream error: {str(ev['error'])[:150]}",
                                  latency=time.perf_counter() - t0)
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chunks += 1
                    if d.get("tool_calls"):
                        tool_called = True
            lat = time.perf_counter() - t0
            if ttft is None:
                return Result(False, "empty response (no tokens streamed)", latency=lat)
            out = (usage or {}).get("completion_tokens") or chunks
            return Result(True, ttft=ttft, latency=lat, out_tokens=out,
                          prompt_tokens=(usage or {}).get("prompt_tokens", 0),
                          tokens_estimated=not (usage or {}).get("completion_tokens"),
                          tool_called=tool_called if args.tools else None)
    except asyncio.TimeoutError:
        return Result(False, f"timeout after {args.timeout:.0f}s", latency=time.perf_counter() - t0)
    except aiohttp.ClientError as e:
        return Result(False, f"{type(e).__name__}: {str(e)[:120]}", latency=time.perf_counter() - t0)


# ----------------------------------------------------------------------------- one level

def pct(values, p):
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


@dataclass
class LevelStats:
    concurrency: int
    requests: int
    ok: int
    wall_s: float
    success_rate: float
    req_per_s: float
    total_tok_s: float
    ttft_p50: float
    ttft_p95: float
    latency_p50: float
    latency_p95: float
    user_tok_s_p50: float
    user_tok_s_p5: float
    avg_prompt_tokens: float
    avg_out_tokens: float
    tool_call_rate: float | None
    errors: dict = field(default_factory=dict)


async def run_level(session, args, concurrency, shared_context, id_offset) -> LevelStats:
    n_requests = args.requests if args.requests else max(concurrency * args.rounds, concurrency)
    counter = iter(range(n_requests))
    results: list[Result] = []
    deadline = time.perf_counter() + args.duration if args.duration else None
    done = 0

    async def worker():
        nonlocal done
        for i in counter:
            if deadline and time.perf_counter() > deadline:
                return
            r = await one_request(session, args, id_offset + i, shared_context)
            results.append(r)
            done += 1
            if not args.quiet:
                print(f"\r  c={concurrency:<4} {done}/{n_requests} done", end="", flush=True)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - t0
    if not args.quiet:
        print("\r" + " " * 40 + "\r", end="")

    ok = [r for r in results if r.ok]
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok:
            key = r.error.split(":")[0] if r.error.startswith("HTTP") else r.error[:60]
            errors[key] = errors.get(key, 0) + 1
    user_speeds = [(r.out_tokens - 1) / (r.latency - r.ttft)
                   for r in ok if r.ttft is not None and r.out_tokens > 1 and r.latency > r.ttft]
    if args.no_stream:
        user_speeds = [r.out_tokens / r.latency for r in ok if r.latency > 0]
    tool_rate = None
    if args.tools and ok:
        tool_rate = sum(1 for r in ok if r.tool_called) / len(ok)
    return LevelStats(
        concurrency=concurrency, requests=len(results), ok=len(ok), wall_s=wall,
        success_rate=len(ok) / max(len(results), 1),
        req_per_s=len(ok) / wall if wall else 0,
        total_tok_s=sum(r.out_tokens for r in ok) / wall if wall else 0,
        ttft_p50=pct([r.ttft for r in ok if r.ttft is not None], 50),
        ttft_p95=pct([r.ttft for r in ok if r.ttft is not None], 95),
        latency_p50=pct([r.latency for r in ok], 50),
        latency_p95=pct([r.latency for r in ok], 95),
        user_tok_s_p50=pct(user_speeds, 50),
        user_tok_s_p5=pct(user_speeds, 5),
        avg_prompt_tokens=sum(r.prompt_tokens for r in ok) / max(len(ok), 1),
        avg_out_tokens=sum(r.out_tokens for r in ok) / max(len(ok), 1),
        tool_call_rate=tool_rate,
        errors=errors,
    )


# ----------------------------------------------------------------------------- report

def fmt(x, nd=1, unit=""):
    return "-" if x is None or x != x else f"{x:.{nd}f}{unit}"


HEADER = (f"{'conc':>5} {'reqs':>5} {'ok%':>6} {'req/s':>6} {'tok/s':>7} "
          f"{'TTFT p50':>9} {'TTFT p95':>9} {'lat p50':>8} {'lat p95':>8} "
          f"{'tok/s/user':>10} {'slowest5%':>9}")


def row(s: LevelStats) -> str:
    line = (f"{s.concurrency:>5} {s.requests:>5} {s.success_rate*100:>5.1f}% {s.req_per_s:>6.2f} "
            f"{s.total_tok_s:>7.0f} {fmt(s.ttft_p50,2,'s'):>9} {fmt(s.ttft_p95,2,'s'):>9} "
            f"{fmt(s.latency_p50,1,'s'):>8} {fmt(s.latency_p95,1,'s'):>8} "
            f"{fmt(s.user_tok_s_p50):>10} {fmt(s.user_tok_s_p5):>9}")
    if s.tool_call_rate is not None:
        line += f"  tools {s.tool_call_rate*100:.0f}%"
    if s.errors:
        line += "  errors: " + ", ".join(f"{k} ×{v}" for k, v in s.errors.items())
    return line


def summarize(levels: list[LevelStats], args):
    print("\nSummary")
    good = [s for s in levels
            if s.success_rate >= args.min_success and (s.ttft_p95 != s.ttft_p95 or s.ttft_p95 <= args.max_ttft)
            and (s.user_tok_s_p50 != s.user_tok_s_p50 or s.user_tok_s_p50 >= args.min_user_tok_s)]
    if good:
        best = max(good, key=lambda s: s.concurrency)
        print(f"  Highest concurrency meeting the targets (≥{args.min_success*100:.0f}% success, "
              f"TTFT p95 ≤ {args.max_ttft:g}s, ≥{args.min_user_tok_s:g} tok/s per user): {best.concurrency}")
    else:
        print("  No level met the targets "
              f"(≥{args.min_success*100:.0f}% success, TTFT p95 ≤ {args.max_ttft:g}s, "
              f"≥{args.min_user_tok_s:g} tok/s per user).")
    if levels:
        peak = max(levels, key=lambda s: s.total_tok_s)
        print(f"  Peak throughput: {peak.total_tok_s:.0f} output tok/s at concurrency {peak.concurrency}")
        healthy = [s for s in levels if s.success_rate >= args.min_success]
        for prev, cur in zip(healthy, healthy[1:]):
            if prev.total_tok_s > 0 and cur.total_tok_s < prev.total_tok_s * 1.10:
                print(f"  Throughput stopped growing between {prev.concurrency} and {cur.concurrency} clients: "
                      "the GPU is saturated, and extra clients mostly wait in the queue (TTFT goes up).")
                break
        failing = [s for s in levels if s.success_rate < 1]
        if failing:
            print(f"  Errors first appeared at concurrency {failing[0].concurrency}: {failing[0].errors}")
    with_ok = [s for s in levels if s.ok]
    if with_ok:
        print(f"  Average prompt: {with_ok[-1].avg_prompt_tokens:.0f} tokens, "
              f"average output: {with_ok[-1].avg_out_tokens:.0f} tokens")


# ----------------------------------------------------------------------------- main

async def detect_model(session, args):
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    async with session.get(f"{args.base_url}/models", headers=headers) as r:
        if r.status != 200:
            sys.exit(f"GET {args.base_url}/models returned HTTP {r.status}: {(await r.text())[:200]}")
        data = await r.json()
    models = [m["id"] for m in data.get("data", [])]
    if not models:
        sys.exit("The server listed no models; pass --model.")
    return models[0]


async def main(args):
    args.base_url = args.base_url.rstrip("/")
    levels_to_run = [int(x) for x in args.concurrency.split(",") if x.strip()]
    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_connect=15)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    results: list[LevelStats] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if not args.model:
            args.model = await detect_model(session, args)
        shared = make_context(args.prompt_tokens, seed=0) if args.shared_prefix else ""
        print(f"Target: {args.base_url}  model: {args.model}")
        print(f"Workload: max_tokens={args.max_tokens}, prompt≈{args.prompt_tokens} tokens"
              f"{' (shared prefix)' if args.shared_prefix and args.prompt_tokens else ''}"
              f"{', tool calling' if args.tools else ''}, {'non-streaming' if args.no_stream else 'streaming'}")

        print("Warming up...", end=" ", flush=True)
        w = await one_request(session, args, 10**6, shared)
        if not w.ok:
            sys.exit(f"\nWarm-up request failed: {w.error}")
        print(f"ok ({w.latency:.1f}s){' (server sent no usage; counting chunks as tokens)' if w.tokens_estimated else ''}\n")

        print(HEADER)
        id_offset = 0
        try:
            for c in levels_to_run:
                s = await run_level(session, args, c, shared, id_offset)
                id_offset += s.requests
                results.append(s)
                print(row(s), flush=True)
                if s.success_rate < args.stop_below_success:
                    print(f"  Stopping: success rate fell below {args.stop_below_success*100:.0f}%.")
                    break
                if args.pause:
                    await asyncio.sleep(args.pause)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nInterrupted; reporting completed levels.")

    summarize(results, args)
    print("\nColumns: TTFT = wait before the first token; lat = whole request; tok/s/user = streaming speed per client "
          "(median and slowest 5%); tok/s = whole-server output throughput.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"config": {k: v for k, v in vars(args).items() if k != "api_key"},
                       "levels": [asdict(s) for s in results]}, f, indent=2)
        print(f"Saved {args.json}")


def parse_args():
    p = argparse.ArgumentParser(description="Concurrency load test for an OpenAI-compatible chat endpoint.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
                   help="API base URL ending in /v1 (env OPENAI_BASE_URL)")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="API key (env OPENAI_API_KEY)")
    p.add_argument("--model", default="", help="model name (default: first model from /v1/models)")
    p.add_argument("-c", "--concurrency", default="1,2,4,8,16,32,64", help="comma-separated levels to ramp through")
    p.add_argument("--rounds", type=int, default=3, help="requests per client at each level (total = conc × rounds)")
    p.add_argument("--requests", type=int, default=0, help="fixed number of requests per level (overrides --rounds)")
    p.add_argument("--duration", type=float, default=0, help="stop starting new requests after this many seconds per level")
    p.add_argument("--max-tokens", type=int, default=256, help="max output tokens per request")
    p.add_argument("--prompt-tokens", type=int, default=0, help="approximate extra context per prompt (e.g. 6000)")
    p.add_argument("--shared-prefix", action="store_true", help="all requests share the same context (prefix caching)")
    p.add_argument("--tools", action="store_true", help="send tool definitions and check for tool calls")
    p.add_argument("--no-stream", action="store_true", help="use non-streaming requests (no TTFT)")
    p.add_argument("--ignore-eos", action="store_true", help="vLLM only: always generate max_tokens")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--timeout", type=float, default=600, help="per-request timeout in seconds")
    p.add_argument("--pause", type=float, default=2, help="seconds to pause between levels")
    p.add_argument("--min-success", type=float, default=0.99, help="target success rate for the summary")
    p.add_argument("--max-ttft", type=float, default=10, help="target p95 TTFT in seconds for the summary")
    p.add_argument("--min-user-tok-s", type=float, default=5, help="target median tok/s per user for the summary")
    p.add_argument("--stop-below-success", type=float, default=0.5, help="stop the ramp when success drops below this")
    p.add_argument("--json", default="", help="write full results to this JSON file")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress counter")
    return p.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        pass
