"""LLM call layer. Supports claude -p and codex exec backends.

Pipeline.py imports llm() from here. The backend is selected per-role
via the config (backend: claude or backend: codex). Default is claude.

When watch=True, streams events to stderr so you can see tool calls,
thinking, and progress in real time. The return value is the same either way.

llm() returns (response, session_id). Pass resume=session_id on a later
call to continue the same conversation.
"""

from __future__ import annotations

import asyncio
import contextvars
import gzip
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone


SUBPROCESS_STREAM_LIMIT = 64 * 1024 * 1024  # 64 MiB

TOOL_CALL_INPUT_LIMIT = 500    # chars per tool input (usually a query)
TOOL_CALL_OUTPUT_LIMIT = 2000  # chars per tool output (can be a web search result)


# ── Per-call logging via contextvar ─────────────────────────────
#
# When set to a list (by the harness, per problem), every successful
# llm() call appends a metadata dict to it. When None (the default),
# llm() behaves identically to before — no side effects. The list is
# shared across child asyncio tasks via contextvars.copy_context(),
# so parallel judge calls all append to the same list without locks.

call_log: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "llm_call_log", default=None,
)


# ── Artifact directory via contextvar ───────────────────────────
#
# When set (by the harness, per problem), every llm() call writes the
# exact inputs and raw outputs to <artifact_dir>/call_NNN_<role>/.
# That directory is the source of truth; the existing call_log entries
# keep their truncated previews for scannability.

artifact_dir: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_artifact_dir", default=None,
)


# ── Sandbox container via contextvar ────────────────────────────
#
# When set (by the harness, per problem), every llm() call runs the
# claude/codex binary via `docker exec <container_id>` instead of on
# the host. The container is created/destroyed by the harness; llm()
# just reads this value and wraps the cmd accordingly.

sandbox_container: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_sandbox_container", default=None,
)

# Image digest for whichever image the sandbox container was built
# from. Optional — informational only, stored in call_meta so post-hoc
# analysis can tell which build produced a given call.

sandbox_image_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_sandbox_image_id", default=None,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# C0 control chars + DEL, except tab/newline/CR which prompts use legitimately.
# LLMs occasionally emit \x00 (and other control bytes) in their output;
# passing those to subprocess argv raises "embedded null byte" (CPython
# #111656). Sanitize at the LLM-output boundary so the same text is safe
# whether downstream uses argv, file content, or JSON.
_BAD_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _sanitize_llm_output(obj):
    """Strip C0 controls (except \\t \\n \\r) and DEL from LLM-generated
    text. Recurses into dicts/lists so structured outputs are cleaned too.
    Non-string scalars pass through unchanged."""
    if isinstance(obj, str):
        return _BAD_CTRL.sub("", obj)
    if isinstance(obj, dict):
        return {k: _sanitize_llm_output(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_llm_output(v) for v in obj]
    return obj


# ── Hang detection ──────────────────────────────────────────────
#
# A subprocess is hung when its per-process CPU ticks (utime + stime
# from /proc/<pid>/stat inside the container) stop accumulating for
# WATCHDOG_SILENCE_SECS. A working LLM call always uses CPU — even
# during server-side reasoning, the client receives streaming bytes,
# parses keepalive pings, schedules tokio timers — all of which
# generate ticks. Literal 0 ticks for many minutes means the HTTP
# connection is silently dead and the client is stuck waiting on it
# forever.
#
# Per-process beats container-wide CPU: when many judge subprocesses
# share one container, container CPU stays nonzero while siblings
# work, masking individual hangs. Per-PID sampling is judge-specific.
#
# Plus a hard wall-time backstop for unanticipated failure modes
# (e.g. busy-spin hangs that consume CPU without producing output).

WATCHDOG_SILENCE_SECS = 90             # 90s of zero CPU ticks → hung
WATCHDOG_HARD_WALL_SECS = 4 * 60 * 60  # 4h absolute ceiling
WATCHDOG_POLL_SECS = 30                # how often the watchdog wakes
WATCHDOG_RETRY_MAX = 3                 # retry attempts after kill


class WatchdogKilled(RuntimeError):
    """Raised when the watchdog killed the subprocess for being hung.
    Distinguished from generic RuntimeError so the retry loop in llm()
    can treat it as a transient failure (codex CLI hang on a specific
    HTTP connection) rather than a real error."""
    pass


async def _find_subprocess_pid(
    container_id: str, comm: str, marker: str | None,
) -> int | None:
    """Find the PID of the actual LLM binary inside `container_id`.

    Always filters by `comm` (the short command name — "codex" or
    "claude"). This is critical: the launch chain is
    bash → node → codex, and all three have the schema file in argv.
    Without the comm filter we'd match the node wrapper, whose CPU
    activity tells us nothing about the underlying Rust client.

    `marker` (e.g. a per-call schema file path) further disambiguates
    when many same-comm processes run in parallel (codex judges).
    Without a marker we just take the first matching comm.
    """
    if marker:
        # Filter both by comm and by argv-contains-marker.
        # ps -eo pid,comm,args puts pid first, comm second, full argv
        # rest. Awk uses index() to substring-match the marker
        # anywhere in the full record (which includes args).
        cmd_str = (
            "ps -eo pid,comm,args | awk -v c=" + shlex.quote(comm)
            + " -v m=" + shlex.quote(marker)
            + " '$2 == c && index($0, m) > 0 {print $1; exit}'"
        )
    else:
        cmd_str = f"pgrep -x {shlex.quote(comm)} | head -1"
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", container_id, "sh", "-c", cmd_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        text = out.decode().strip()
        if not text:
            return None
        return int(text.split()[0])
    except (asyncio.TimeoutError, ValueError, OSError):
        return None


async def _process_cpu_ticks(container_id: str, pid: int) -> int | None:
    """Read utime+stime (CPU ticks) for `pid` inside `container_id`
    via /proc/<pid>/stat. Fields 14 and 15 of /proc/<pid>/stat per
    proc(5). Returns None if the process is gone or read fails."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", container_id, "sh", "-c",
            f"awk '{{print $14, $15}}' /proc/{pid}/stat",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        parts = out.decode().split()
        if len(parts) != 2:
            return None
        return int(parts[0]) + int(parts[1])
    except (asyncio.TimeoutError, ValueError, OSError):
        return None


async def _watchdog(proc, container_id, kill_flag, *,
                     backend: str, marker: str | None = None,
                     label: str = ""):
    """Kill `proc` if its corresponding LLM subprocess inside
    `container_id` accumulates no CPU ticks for SILENCE_SECS, OR after
    HARD_WALL_SECS regardless. `kill_flag` is a one-element [bool] set
    to True before kill so the caller can tell a watchdog kill apart
    from other failures.

    `backend` is "codex" or "claude" — used for the comm fallback when
    `marker` is None.
    """
    started = time.monotonic()
    pid: int | None = None
    last_active = started
    last_ticks: int | None = None
    comm = "codex" if backend == "codex" else "claude"
    tag = f":{label}" if label else ""
    try:
        while proc.returncode is None:
            await asyncio.sleep(WATCHDOG_POLL_SECS)
            now = time.monotonic()
            if now - started > WATCHDOG_HARD_WALL_SECS:
                print(
                    f"[watchdog{tag}] hard wall-time "
                    f"{WATCHDOG_HARD_WALL_SECS}s exceeded — killing",
                    file=sys.stderr,
                )
                kill_flag[0] = True
                try: proc.kill()
                except ProcessLookupError: pass
                return

            # Lazily look up the in-container PID. The subprocess may
            # take a moment to start — if not found yet, wait the
            # next poll. Reset to None if the process has exited so
            # we re-discover (relevant for retries / proc lifecycle).
            if pid is None:
                pid = await _find_subprocess_pid(container_id, comm, marker)
                if pid is None:
                    continue
                last_ticks = await _process_cpu_ticks(container_id, pid)
                last_active = now  # baseline once we have a PID
                continue

            ticks = await _process_cpu_ticks(container_id, pid)
            if ticks is None:
                # Could be transient (docker exec timeout, scheduling
                # delay) or the process is genuinely gone. Don't exit
                # on a single failure — verify the PID is gone by
                # re-running the lookup. If still findable, treat as
                # transient and skip this sample. If missing, exit.
                still_there = await _find_subprocess_pid(
                    container_id, comm, marker,
                )
                if still_there is None:
                    return  # Process really did exit
                # Transient: don't update last_active, don't update
                # last_ticks, just wait for next poll.
                continue

            if last_ticks is not None and ticks != last_ticks:
                last_active = now  # any tick movement = activity
            last_ticks = ticks

            silent_secs = int(now - last_active)
            if silent_secs > WATCHDOG_SILENCE_SECS:
                print(
                    f"[watchdog{tag}] no CPU activity from pid {pid} "
                    f"for {silent_secs}s — killing",
                    file=sys.stderr,
                )
                kill_flag[0] = True
                try: proc.kill()
                except ProcessLookupError: pass
                return
    except asyncio.CancelledError:
        pass


# ── Resume from cache ───────────────────────────────────────────
#
# When the same call_dir already contains a successful response (from
# a prior run that we're resuming), reuse it — but ONLY after verifying
# the saved prompt.txt matches the current prompt. Idempotency check
# prevents silent corruption when pipeline code or prompt templates
# have drifted between runs.

def _try_resume_from_cache(
    call_dir: str, prompt: str, system: str | None, schema: dict | None,
):
    """Returns (response, session_id, call_meta) if call_dir has a
    complete and idempotent prior result. Returns None if nothing
    cached. Raises if the cached prompt doesn't match the current
    prompt — that's a state-drift bug, not a fallback case."""
    response_path = os.path.join(call_dir, "response.txt")
    meta_path = os.path.join(call_dir, "meta.json")
    prompt_path = os.path.join(call_dir, "prompt.txt")
    if not (os.path.exists(response_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if meta.get("failed"):
        return None
    rc = meta.get("returncode")
    if rc not in (0, None):
        return None
    # Idempotency check: saved prompt must match what we'd send now.
    if os.path.exists(prompt_path):
        try:
            with open(prompt_path) as f:
                cached_prompt = f.read()
        except OSError:
            return None
        if cached_prompt != prompt:
            raise RuntimeError(
                f"resume idempotency check failed for {call_dir}: "
                f"saved prompt.txt ({len(cached_prompt)} chars) does "
                f"not match the current prompt ({len(prompt)} chars). "
                f"Pipeline state has drifted from the original run. "
                f"Either delete {call_dir} to force a fresh LLM call, "
                f"or revert the change that caused the drift."
            )
    # Read response
    try:
        with open(response_path) as f:
            response_text = f.read()
    except OSError:
        return None
    if schema:
        try:
            response = json.loads(response_text)
        except json.JSONDecodeError:
            return None
    else:
        response = response_text
    session_id = meta.get("session_id")
    cache_meta = dict(meta)
    cache_meta["resumed_from_cache"] = True
    return response, session_id, cache_meta


# ── Artifact helpers ────────────────────────────────────────────

def _write_artifact(path: str, data: bytes | str, *, gzip_it: bool = False) -> str:
    """Write raw bytes/text to `path`. With gzip_it=True, the stored
    file has a `.gz` suffix. Returns the actual written path.

    No atomic dance — artifacts are written once per call, never
    overwritten by other callers (call index + role make the directory
    name unique), so a plain write is fine. Crash safety here is not
    worth the extra complexity.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    if gzip_it:
        out = path + ".gz"
        with gzip.open(out, "wb") as f:
            f.write(data)
        return out
    with open(path, "wb") as f:
        f.write(data)
    return path


def _cmd_hash(cmd: list[str]) -> str:
    """Short hash of an argv list, for grouping identical invocations."""
    return hashlib.sha256("\x1f".join(cmd).encode()).hexdigest()[:16]


def _effective_model(cmd: list[str]) -> str | None:
    """Pull the model name out of the cmd (whatever was actually sent).

    Both backends accept `--model X`. This sidesteps a subtle issue
    where the config says `model: opus` but the codex cmd builder
    translates that to `gpt-5.5`; `settings["model"]` still reads the
    pre-translation value. Recording the effective model avoids
    misleading metadata in the saved call_meta."""
    try:
        i = cmd.index("--model")
        return cmd[i + 1]
    except (ValueError, IndexError):
        return None


def _extract_claude_metadata(result_event: dict) -> dict:
    """Pull per-call tokens/cost/tool-use from a claude result event."""
    usage = result_event.get("usage") or {}
    server_tools = usage.get("server_tool_use") or {}
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        "total_cost_usd": result_event.get("total_cost_usd"),
        "num_turns": result_event.get("num_turns"),
        "duration_api_ms": result_event.get("duration_api_ms"),
        "stop_reason": result_event.get("stop_reason"),
        "is_error": result_event.get("is_error", False),
        "web_search_requests": server_tools.get("web_search_requests", 0),
        "web_fetch_requests": server_tools.get("web_fetch_requests", 0),
    }


def _extract_codex_metadata(events: list) -> dict:
    """Pull per-call tokens from codex turn.completed events.

    Codex can emit multiple turn.completed events per call (tool calls
    produce extra turns), so we sum across all of them.
    """
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0
    for event in events:
        if event.get("type") != "turn.completed":
            continue
        usage = event.get("usage") or {}
        input_tokens += usage.get("input_tokens", 0) or 0
        output_tokens += usage.get("output_tokens", 0) or 0
        cached_input_tokens += usage.get("cached_input_tokens", 0) or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "total_cost_usd": None,  # codex doesn't expose cost
    }


def _truncate(s, limit: int) -> str:
    """Truncate with a marker so partial content is obvious in logs."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = json.dumps(s) if isinstance(s, (dict, list)) else str(s)
    if len(s) <= limit:
        return s
    return f"{s[:limit]}\n...[TRUNCATED: {len(s) - limit} of {len(s)} chars]"


def _maybe_truncate(s, limit: int, *, truncate: bool):
    """Apply _truncate(s, limit) when truncate=True; otherwise stringify
    without cutting. For the artifact copy we want the full payload."""
    if not truncate:
        if s is None:
            return ""
        if isinstance(s, str):
            return s
        if isinstance(s, (dict, list)):
            return json.dumps(s)
        return str(s)
    return _truncate(s, limit)


def _extract_claude_tool_calls(events: list, *, truncate: bool = True) -> list:
    """Pair claude tool_use events with their tool_result events.

    With truncate=False, tool inputs/outputs are kept in full — used for
    writing the untruncated tool_calls.json artifact.
    """
    uses, order = {}, []
    for e in events:
        for b in ((e.get("message") or {}).get("content") or []):
            if b.get("type") == "tool_use":
                uses[b["id"]] = {
                    "tool_name": b.get("name"),
                    "input": _maybe_truncate(
                        b.get("input"), TOOL_CALL_INPUT_LIMIT, truncate=truncate,
                    ),
                }
                order.append(b["id"])
            elif b.get("type") == "tool_result":
                tid = b.get("tool_use_id")
                if tid in uses:
                    uses[tid]["output"] = _maybe_truncate(
                        b.get("content"), TOOL_CALL_OUTPUT_LIMIT,
                        truncate=truncate,
                    )
    return [uses[tid] for tid in order]


def _extract_codex_tool_calls(events: list, *, truncate: bool = True) -> list:
    """Pair codex function_call events with function_call_output events."""
    calls, order = {}, []
    for e in events:
        if e.get("type") != "item.completed":
            continue
        item = e.get("item") or {}
        cid = item.get("call_id")
        if item.get("type") == "function_call" and cid:
            calls[cid] = {
                "tool_name": item.get("name"),
                "input": _maybe_truncate(
                    item.get("arguments"), TOOL_CALL_INPUT_LIMIT,
                    truncate=truncate,
                ),
            }
            order.append(cid)
        elif item.get("type") == "function_call_output" and cid in calls:
            calls[cid]["output"] = _maybe_truncate(
                item.get("output"), TOOL_CALL_OUTPUT_LIMIT, truncate=truncate,
            )
    return [calls[cid] for cid in order]


# ── Failure diagnostics ─────────────────────────────────────────

def _format_failure(backend, returncode, stderr_bytes, stdout_bytes=None):
    """Build an informative subprocess failure message.

    Includes returncode and any stderr content. If stdout is also passed
    (batch mode only), includes its tail too. The previous error format
    was just `f"{backend} failed: {stderr}"`, which produced messages like
    `"claude failed: "` (no payload) when stderr was empty — making the
    failure undebuggable.
    """
    parts = [f"{backend} failed (returncode={returncode})"]
    stderr_str = (stderr_bytes or b"").decode(errors="replace").strip()
    if stderr_str:
        parts.append(f"stderr: {stderr_str[:1500]}")
    if stdout_bytes is not None:
        stdout_str = (stdout_bytes or b"").decode(errors="replace").strip()
        if stdout_str:
            tail = stdout_str[-1000:]
            parts.append(f"stdout tail: ...{tail}")
    if len(parts) == 1:
        parts.append("(no stderr or stdout captured)")
    return " | ".join(parts)


# ── Claude ──────────────────────────────────────────────────────

def _build_claude_cmd(prompt, settings, schema, system, watch, resume, *,
                       sandboxed: bool = False):
    model = settings.get("model", "opus")
    effort = settings.get("effort", "max")
    tools = settings.get("tools")

    # Defense-in-depth: even if upstream missed sanitizing, never let a
    # null byte reach subprocess argv (raises "embedded null byte").
    prompt = _BAD_CTRL.sub("", prompt) if isinstance(prompt, str) else prompt
    system = _BAD_CTRL.sub("", system) if isinstance(system, str) else system

    cmd = ["claude", "--model", model, "--effort", effort, "-p", prompt]

    if resume:
        cmd += ["--resume", resume]

    if tools is not None:
        cmd += ["--tools", tools]

    if watch:
        # Stream-json for live events — works with or without schema
        cmd += ["--output-format", "stream-json", "--verbose"]
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]
    else:
        # Always use JSON format so we can capture session_id
        cmd += ["--output-format", "json"]
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]

    if system:
        cmd += ["--append-system-prompt", system]

    # Inside a Docker container, the container IS the sandbox — skip
    # claude's native permission prompts/Seatbelt. Outside the container
    # we leave behavior unchanged (print mode already doesn't prompt
    # interactively).
    if sandboxed:
        cmd += ["--dangerously-skip-permissions"]

    return cmd


def _parse_claude_output(stdout, schema):
    """Parse non-streaming claude output.

    Returns (response, session_id, metadata, events). The events list is
    returned so callers can persist the full untruncated event stream
    without re-parsing.

    With `--output-format json` (what the batch path uses), claude emits
    a single result object — not a list of events. Older versions, and
    some wrappers, emitted a JSON array of events instead, so we accept
    both shapes.
    """
    parsed = json.loads(stdout)

    if isinstance(parsed, dict):
        # Single result object (current `--output-format json` shape).
        # There are no intermediate tool-use events in this mode — tool
        # calls can only be captured via streaming.
        result_event = parsed
        events = [parsed]
    elif isinstance(parsed, list):
        events = parsed
        try:
            result_event = next(
                e for e in reversed(events)
                if isinstance(e, dict) and e.get("type") == "result"
            )
        except StopIteration:
            raise RuntimeError(
                "claude output contained no result event. "
                f"got {len(events)} items; last={events[-1] if events else None!r}"
            )
    else:
        raise RuntimeError(
            f"unexpected claude output shape: {type(parsed).__name__} "
            f"(first 500 chars: {stdout[:500]!r})"
        )
    session_id = result_event.get("session_id")
    metadata = _extract_claude_metadata(result_event)
    metadata["tool_calls"] = _extract_claude_tool_calls(events)
    if schema:
        if "structured_output" not in result_event:
            # Diagnostic: dump everything we know about the failed result
            raise RuntimeError(
                "claude returned a result event without 'structured_output'. "
                f"is_error={result_event.get('is_error')!r} "
                f"subtype={result_event.get('subtype')!r} "
                f"stop_reason={result_event.get('stop_reason')!r} "
                f"result={(result_event.get('result') or '')[:500]!r} "
                f"keys={list(result_event.keys())}"
            )
        return _sanitize_llm_output(result_event["structured_output"]), session_id, metadata, events
    return _sanitize_llm_output(result_event.get("result", "")), session_id, metadata, events


def _print_claude_event(event):
    """Print a claude stream-json event to stderr."""
    t = event.get("type")

    if t == "assistant":
        content = event.get("message", {}).get("content", [])
        for block in content:
            if block.get("type") == "tool_use":
                print(f"      [tool] {block['name']}({json.dumps(block.get('input', {}))[:100]})", file=sys.stderr)
            elif block.get("type") == "text":
                text = block["text"][:200]
                if text.strip():
                    print(f"      [text] {text}", file=sys.stderr)

    elif t == "result":
        cost = event.get("total_cost_usd")
        turns = event.get("num_turns", 0)
        if cost is not None:
            print(f"      [done] {turns} turns, ${cost:.4f}", file=sys.stderr)


async def _run_claude_streaming(proc, schema, *, last_event_ref=None):
    """Read claude stream-json line by line, print events, return
    (response, session_id, metadata, events, raw_stdout).

    `raw_stdout` is the concatenated bytes we read — kept so callers
    can save the untouched provider output as an artifact. `events` is
    the parsed per-line list, for the same reason.
    """
    result_event = None
    session_id = None
    events = []
    raw_chunks: list[bytes] = []

    async for raw_line in proc.stdout:
        if last_event_ref is not None:
            last_event_ref[0] = time.monotonic()
        raw_chunks.append(raw_line)
        line = raw_line.decode().strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        events.append(event)
        _print_claude_event(event)

        if session_id is None and event.get("session_id"):
            session_id = event["session_id"]

        if event.get("type") == "result":
            result_event = event
            # claude's protocol emits exactly one terminal "result" event
            # per CLI invocation. Without this break, the loop continues
            # waiting for stdout EOF — which through `docker exec` can
            # hang indefinitely even after claude itself has exited
            # (manifests as the watch output showing [done] but the
            # pipeline never recording formalizer_returned).
            break

    await proc.wait()
    raw_stdout = b"".join(raw_chunks)

    if proc.returncode != 0:
        stderr = await proc.stderr.read()
        raise RuntimeError(_format_failure("claude", proc.returncode, stderr))

    if result_event is None:
        raise RuntimeError("claude stream ended without result event")

    metadata = _extract_claude_metadata(result_event)
    metadata["tool_calls"] = _extract_claude_tool_calls(events)
    if schema:
        if "structured_output" not in result_event:
            raise RuntimeError(
                "claude returned a result event without 'structured_output'. "
                f"is_error={result_event.get('is_error')!r} "
                f"subtype={result_event.get('subtype')!r} "
                f"stop_reason={result_event.get('stop_reason')!r} "
                f"result={(result_event.get('result') or '')[:500]!r} "
                f"keys={list(result_event.keys())}"
            )
        return _sanitize_llm_output(result_event["structured_output"]), session_id, metadata, events, raw_stdout
    return _sanitize_llm_output(result_event.get("result", "")), session_id, metadata, events, raw_stdout


# ── Codex ───────────────────────────────────────────────────────

# Codex roles that are invoked sequentially and can be resumed on
# a later call. These must share CODEX_HOME so that `codex exec
# resume <session_id>` can locate the rollout file written by the
# initial call. Roles not in this set are either always-initial or
# run in parallel, and get per-call CODEX_HOME isolation to avoid
# concurrent processes corrupting each other's SQLite state.
_RESUMABLE_CODEX_ROLES = {"solver", "formalizer"}


def _build_codex_cmd(prompt, settings, schema_file, system, resume, *,
                      sandboxed: bool = False, role: str | None = None):
    model = settings.get("model", "gpt-5.5")
    sandbox = settings.get("sandbox", "read-only")
    effort = settings.get("effort", "xhigh")

    # Defense-in-depth: never let a null byte reach subprocess argv.
    prompt = _BAD_CTRL.sub("", prompt) if isinstance(prompt, str) else prompt
    system = _BAD_CTRL.sub("", system) if isinstance(system, str) else system

    # Translate claude's "max" to codex's "xhigh" (both mean highest reasoning)
    if effort == "max":
        effort = "xhigh"

    # claude uses "opus"/"sonnet"/"haiku" aliases; if the config has a claude
    # model but backend is codex, fall back to a codex model
    if model in ("opus", "sonnet", "haiku"):
        model = "gpt-5.5"

    if resume:
        # `codex exec resume` only accepts a subset of flags. It does NOT
        # support --sandbox or --output-schema. The sandbox setting is
        # inherited from the original session. Schema-on-resume is not
        # supported at all (separate code path handles this).
        #
        # It DOES require --skip-git-repo-check and
        # --dangerously-bypass-approvals-and-sandbox when running
        # inside our container: codex re-runs its per-exec trust and
        # approval checks on every resume call, not just the first
        # session, so without these flags a resumed call fails with
        # "Not inside a trusted directory" and hangs waiting for tool
        # approval. The container is our isolation boundary; these
        # flags are safe here for the same reason as on the initial call.
        cmd = ["codex", "exec", "resume", resume]
        cmd += ["--model", model]
        if sandboxed:
            cmd += ["--dangerously-bypass-approvals-and-sandbox"]
            cmd += ["--skip-git-repo-check"]
        cmd += ["-c", f"model_reasoning_effort={effort}"]
        cmd += ["--json"]
    else:
        cmd = ["codex", "exec"]
        cmd += ["--model", model]
        # Inside a Docker container we trust the container as the
        # sandbox and drop codex's internal Seatbelt/bubblewrap +
        # approval checks. Outside, keep the native sandbox (default
        # read-only) so codex can't accidentally rampage on the host.
        if sandboxed:
            cmd += ["--dangerously-bypass-approvals-and-sandbox"]
            # /workspace inside the container isn't a git repo; codex
            # refuses to run otherwise. The flag is safe here because
            # the container is the isolation boundary.
            cmd += ["--skip-git-repo-check"]
        else:
            cmd += ["--sandbox", sandbox]
        cmd += ["-c", f"model_reasoning_effort={effort}"]
        cmd += ["--json"]
        if settings.get("full_auto") and not sandboxed:
            # --full-auto is shorthand for --sandbox workspace-write.
            # Redundant (and conflicting) with the bypass flag above.
            cmd += ["--full-auto"]
        if schema_file:
            cmd += ["--output-schema", schema_file]

    if settings.get("search"):
        cmd += ["-c", "tools.web_search=true"]

    if system and not resume:
        # System prompt only on initial call; resume continues existing context
        cmd.append(f"{system}\n\n{prompt}")
    else:
        cmd.append(prompt)

    # Parallel codex calls (judges, pedantry, state 0 audit) share
    # /home/node/.codex when sandboxed — including sqlite DBs (logs_*.sqlite,
    # state_*.sqlite) that codex mmaps. Concurrent processes truncating or
    # re-initializing these files trigger SIGBUS (exit 135) in the other
    # processes. Give each INITIAL call its own CODEX_HOME under /tmp so
    # their state is fully isolated.
    #
    # EXCEPTION: roles that can be resumed (solver, formalizer) MUST
    # use the shared /home/node/.codex. `codex exec resume <session>`
    # looks up the rollout at $CODEX_HOME/sessions/YYYY/MM/DD/rollout-*.jsonl
    # (confirmed via codex docs + session-storage refs). If the initial
    # call wrote the rollout to a per-call /tmp/codex-XXX, the later
    # resume using the default $HOME can't find it and fails with
    # "no rollout found for thread <id>". Keeping these roles on
    # shared CODEX_HOME preserves session continuity. They're called
    # sequentially per problem so no parallel-state corruption risk.
    is_resumable = role in _RESUMABLE_CODEX_ROLES
    if sandboxed and not resume and not is_resumable:
        uid = uuid.uuid4().hex[:12]
        home = f"/tmp/codex-{uid}"
        inner = " ".join(shlex.quote(a) for a in cmd)
        cmd = [
            "bash", "-c",
            f"mkdir -p {home} && "
            f"cp -f /home/node/.codex/auth.json /home/node/.codex/config.toml "
            f"/home/node/.codex/installation_id {home}/ 2>/dev/null; "
            f"CODEX_HOME={home} {inner}"
        ]

    return cmd


def _parse_codex_output(stdout, schema):
    """Parse non-streaming codex output.

    Returns (response, session_id, metadata, events) — events is the
    parsed JSONL list, kept so callers can save it as an artifact.
    """
    lines = [json.loads(line) for line in stdout.strip().split("\n") if line.strip()]
    metadata = _extract_codex_metadata(lines)
    metadata["tool_calls"] = _extract_codex_tool_calls(lines)

    session_id = None
    for event in lines:
        if event.get("type") == "thread.started":
            session_id = event.get("thread_id")
            break

    for event in reversed(lines):
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", "")
                if schema:
                    try:
                        return _sanitize_llm_output(json.loads(text)), session_id, metadata, lines
                    except json.JSONDecodeError as e:
                        raise RuntimeError(
                            f"codex returned non-JSON when schema was requested. "
                            f"This usually means a session was resumed (codex exec resume "
                            f"does not support --output-schema). "
                            f"text={text[:500]!r} error={e}"
                        )
                return _sanitize_llm_output(text), session_id, metadata, lines

    raise RuntimeError("No response found in codex output")


def _print_codex_event(event):
    """Print a codex JSONL event to stderr."""
    t = event.get("type")

    if t == "item.completed":
        item = event.get("item", {})
        item_type = item.get("type", "")

        if item_type == "function_call":
            print(f"      [tool] {item.get('name', '?')}({item.get('arguments', '')[:100]})", file=sys.stderr)
        elif item_type == "function_call_output":
            output = item.get("output", "")[:200]
            print(f"      [result] {output}", file=sys.stderr)
        elif item_type == "agent_message":
            text = item.get("text", "")[:200]
            if text.strip():
                print(f"      [text] {text}", file=sys.stderr)

    elif t == "turn.completed":
        usage = event.get("usage", {})
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        print(f"      [done] {tokens} tokens", file=sys.stderr)


async def _run_codex_streaming(proc, schema, *, last_event_ref=None):
    """Read codex JSONL line by line, print events, return
    (response, session_id, metadata, events, raw_stdout)."""
    last_message_text = None
    session_id = None
    events = []
    raw_chunks: list[bytes] = []

    async for raw_line in proc.stdout:
        if last_event_ref is not None:
            last_event_ref[0] = time.monotonic()
        raw_chunks.append(raw_line)
        line = raw_line.decode().strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        events.append(event)
        _print_codex_event(event)

        if event.get("type") == "thread.started" and session_id is None:
            session_id = event.get("thread_id")

        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                last_message_text = item.get("text", "")

    await proc.wait()
    raw_stdout = b"".join(raw_chunks)

    if proc.returncode != 0:
        stderr = await proc.stderr.read()
        raise RuntimeError(_format_failure("codex", proc.returncode, stderr))

    if last_message_text is None:
        raise RuntimeError("codex stream ended without agent message")

    metadata = _extract_codex_metadata(events)
    metadata["tool_calls"] = _extract_codex_tool_calls(events)
    if schema:
        return _sanitize_llm_output(json.loads(last_message_text)), session_id, metadata, events, raw_stdout
    return _sanitize_llm_output(last_message_text), session_id, metadata, events, raw_stdout


# ── Schema helper ───────────────────────────────────────────────

def _add_additional_properties(schema):
    """Codex requires additionalProperties: false on all objects."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if out.get("type") == "object":
        out.setdefault("additionalProperties", False)
    for key in ("properties", "items"):
        if key in out:
            val = out[key]
            if isinstance(val, dict):
                if key == "items":
                    out[key] = _add_additional_properties(val)
                else:
                    out[key] = {k: _add_additional_properties(v) for k, v in val.items()}
    return out


# ── SDK backends (API-key path) ─────────────────────────────────
#
# Direct SDK calls to the Anthropic / OpenAI APIs — no subprocess, no
# Docker, no Keychain. Code execution and web search run in the
# providers' own server-side sandboxes.
#
# These backends bypass the entire subprocess/docker/watchdog machinery
# used by the CLI backends. They produce the same artifact shapes and
# call_log entries, so downstream consumers (harness, grade, show)
# don't care which path produced a given call.

# In-process session histories for Anthropic. The SDK is stateless, so
# multi-turn conversations (formalizer repair loop, solver retry) need
# the client to replay full history. Keyed by a Theoria-generated
# session_id; persisted to history.json in the per-call artifact dir so
# `theoria --resume` can reconstruct after a crash.
_anthropic_sessions: dict[str, dict] = {}


def _anthropic_tools(settings: dict) -> list[dict]:
    """Build the server-side tool list for an Anthropic call."""
    tools: list[dict] = []
    if settings.get("search"):
        tools.append({"type": "web_search_20260209", "name": "web_search"})
        tools.append({"type": "web_fetch_20260209", "name": "web_fetch"})
    if settings.get("code_exec"):
        tools.append({"type": "code_execution_20260120", "name": "code_execution"})
    return tools


def _openai_tools(settings: dict) -> list[dict]:
    """Build the server-side tool list for an OpenAI Responses call."""
    tools: list[dict] = []
    if settings.get("search"):
        tools.append({"type": "web_search"})
    if settings.get("code_exec"):
        tools.append({
            "type": "code_interpreter",
            "container": {"type": "auto", "memory_limit": "4g"},
        })
    return tools


def _normalize_effort(effort: str) -> str:
    """Map Theoria's legacy `effort: max` to `xhigh`.

    Both providers support both values, but xhigh is the right default
    for coding/agentic work on the current generation (Opus 4.7,
    gpt-5.5). `max` is still accepted by the providers, so users who
    set it explicitly in a config get what they asked for; only the
    default-named alias gets remapped.
    """
    if effort == "max":
        return "xhigh"
    return effort


def _anthropic_max_tokens(model: str) -> int:
    """Pick a sensible max_tokens ceiling for the model.

    Opus supports 128K output; Sonnet and Haiku cap at 64K. Streaming
    is always used (the SDK refuses non-streaming above ~16K), so a
    large ceiling is free — the model still only generates what it
    needs.
    """
    if "opus" in model.lower():
        return 128000
    return 64000


# Lazy module-level SDK clients. The SDKs do HTTP connection pooling
# per-client, so parallel judges benefit from a single shared instance
# rather than one per call. Imports stay inside the getter so the [api]
# extra remains optional.
_anthropic_client = None
_openai_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "Anthropic backend requested but the `anthropic` package "
                "isn't installed. Run `pip install -e \".[api]\"` from the "
                "repo root and try again."
            )
        # 30 min timeout covers adaptive thinking + xhigh effort + code
        # exec on hard problems. The SDK auto-retries 429 / 5xx.
        _anthropic_client = anthropic.AsyncAnthropic(timeout=30 * 60.0)
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError(
                "OpenAI backend requested but the `openai` package isn't "
                "installed. Run `pip install -e \".[api]\"` from the repo "
                "root and try again."
            )
        _openai_client = AsyncOpenAI(timeout=30 * 60.0)
    return _openai_client


def _serialize_anthropic_content(content) -> list[dict]:
    """Turn an Anthropic Message.content list (TextBlock / ThinkingBlock /
    ToolUseBlock / etc.) into plain dicts suitable for the messages API.

    The SDK accepts dict-shaped content blocks on echoed-back assistant
    turns. Using model_dump() preserves the exact field set the API
    expects for each block type (tool_use ids, citation envelopes,
    thinking signatures, etc.).
    """
    out = []
    for block in content:
        if hasattr(block, "model_dump"):
            out.append(block.model_dump(exclude_none=True))
        elif isinstance(block, dict):
            out.append(block)
        else:
            out.append({"type": "text", "text": str(block)})
    return out


def _extract_anthropic_text(content) -> str:
    """Concatenate all text blocks in an Anthropic response. Schema'd
    calls put valid JSON in the first text block; non-schema'd calls may
    interleave text with tool_use blocks — joining preserves the model's
    final prose."""
    parts = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts)


def _extract_anthropic_tool_use(content, *, truncate: bool = True) -> list[dict]:
    """Extract tool_use blocks from an Anthropic Message for the call
    log. Matches the shape of _extract_claude_tool_calls but pulls
    inputs directly from the final Message rather than from the event
    stream (server-side tools don't surface intermediate events on
    non-streaming responses)."""
    out = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype not in ("tool_use", "server_tool_use"):
            continue
        out.append({
            "tool_name": getattr(block, "name", None),
            "input": _maybe_truncate(
                getattr(block, "input", None),
                TOOL_CALL_INPUT_LIMIT, truncate=truncate,
            ),
        })
    return out


async def _call_anthropic(
    prompt: str,
    role: str,
    settings: dict,
    schema: dict | None,
    system: str | None,
    resume: str | None,
    watch: bool,
    call_dir: str | None,
) -> tuple[object, str, dict, list, bytes]:
    """Call the Anthropic Messages API.

    Returns (response, session_id, metadata, events, raw_stdout) —
    same shape as _run_claude_streaming so the artifact-save path in
    llm() can handle both uniformly.

    Multi-turn continuity: the SDK is stateless, so a Theoria-generated
    session_id keys a history dict (mirrored to history.json in the
    call_dir for crash recovery). On resume the prior history is looked
    up and the new turn is appended.
    """
    client = _get_anthropic_client()

    model = settings.get("model", "claude-opus-4-7")
    effort = _normalize_effort(settings.get("effort", "xhigh"))

    # Restore prior history on resume — first from memory, then from
    # the on-disk history.json a prior cache-hit may have left behind.
    container_id: str | None = None
    if resume and resume in _anthropic_sessions:
        sess = _anthropic_sessions[resume]
        messages = list(sess["messages"])
        container_id = sess.get("container_id")
        session_id = resume
    elif resume:
        # Stale in-memory dict (e.g. after --resume into a fresh
        # process). Loud failure here would break repair loops; start
        # a fresh session and let the caller's prior context get
        # replayed via prompt content instead.
        print(
            f"[anthropic] warning: resume={resume[:16]}... requested but "
            f"no in-memory history. Starting a fresh session.",
            file=sys.stderr,
        )
        messages = []
        session_id = f"anthropic_{uuid.uuid4().hex}"
    else:
        messages = []
        session_id = f"anthropic_{uuid.uuid4().hex}"

    messages.append({"role": "user", "content": prompt})

    tools = _anthropic_tools(settings)

    output_config: dict = {"effort": effort}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}

    kwargs: dict = {
        "model": model,
        # Streaming is always used below, so a high ceiling is free —
        # the model still only generates what it needs. 128K for Opus
        # (the documented cap), 64K for Sonnet/Haiku.
        "max_tokens": _anthropic_max_tokens(model),
        "messages": messages,
        "output_config": output_config,
    }
    if system:
        # Cache the system prefix across calls in the same session
        # (formalizer repair loop is the big winner here). Harmless
        # if the prompt is below the ~1024-token cacheable minimum —
        # the SDK just doesn't write the cache.
        kwargs["system"] = [{
            "type": "text", "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
    if settings.get("thinking") == "adaptive":
        kwargs["thinking"] = {"type": "adaptive"}
    if tools:
        kwargs["tools"] = tools
    if container_id:
        # Reuse the code-execution container across resume calls so
        # Python state (variables, installed packages, files) persists.
        kwargs["container"] = container_id

    # Streaming is always used — both because the SDK's accumulator
    # (`get_final_message`) only fills in as the event loop is
    # iterated, and because `max_tokens` is high enough that
    # non-streaming would risk SDK HTTP timeouts. `watch` only
    # controls whether events get printed to stderr along the way.
    #
    # Server-side tools (web_search, code_execution) auto-loop on the
    # server up to ~10 iterations and then return stop_reason=pause_turn
    # if not done. The documented continuation pattern is to re-send
    # with the assistant turn appended; PAUSE_MAX bounds the loop in
    # case something goes pathological.
    final_message = None
    pause_continuations = 0
    PAUSE_MAX = 5
    while True:
        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if not watch:
                    continue
                if event.type == "content_block_start":
                    block = event.content_block
                    btype = getattr(block, "type", None)
                    if btype in ("tool_use", "server_tool_use"):
                        print(
                            f"      [tool] {getattr(block, 'name', '?')}",
                            file=sys.stderr,
                        )
                    elif btype == "thinking":
                        print("      [thinking...]", file=sys.stderr)
            final_message = await stream.get_final_message()

        if final_message.stop_reason != "pause_turn":
            break
        if pause_continuations >= PAUSE_MAX:
            raise RuntimeError(
                f"anthropic pause_turn loop did not converge after "
                f"{PAUSE_MAX} continuations"
            )
        # Per docs: re-send messages with the assistant turn appended.
        # The server detects the trailing server_tool_use and resumes.
        kwargs["messages"] = messages + [
            {"role": "assistant",
             "content": _serialize_anthropic_content(final_message.content)},
        ]
        pause_continuations += 1

    full_text = _extract_anthropic_text(final_message.content)

    if schema:
        try:
            response = json.loads(full_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"anthropic returned non-JSON for a schema'd call "
                f"(stop_reason={final_message.stop_reason!r}). "
                f"text={full_text[:500]!r} error={e}"
            )
    else:
        response = full_text
    response = _sanitize_llm_output(response)

    # Echo the assistant turn back into the history dict so the next
    # resume call has it. Serializing via model_dump rather than
    # storing the raw content blocks is intentional — the API accepts
    # dict-shaped blocks on echo-back, and dicts also serialize cleanly
    # to history.json below.
    messages.append({
        "role": "assistant",
        "content": _serialize_anthropic_content(final_message.content),
    })
    new_container = getattr(final_message, "container", None)
    new_container_id = getattr(new_container, "id", None) if new_container else None
    _anthropic_sessions[session_id] = {
        "messages": messages,
        "container_id": new_container_id or container_id,
    }
    # Mirror to disk so `theoria --resume` can rebuild the in-memory
    # dict after the process restarts.
    if call_dir is not None:
        try:
            with open(os.path.join(call_dir, "history.json"), "w") as f:
                json.dump(_anthropic_sessions[session_id], f, default=str)
        except OSError:
            pass  # artifact failure shouldn't break the call

    usage = final_message.usage
    metadata = {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "stop_reason": final_message.stop_reason,
        # total_cost_usd: Anthropic SDK doesn't expose per-call cost on
        # the Message object; downstream metrics treat None as "unknown".
        "total_cost_usd": None,
        "tool_calls": _extract_anthropic_tool_use(final_message.content),
    }

    events = [final_message.model_dump(exclude_none=True)]
    raw_stdout = json.dumps(events[0], default=str).encode("utf-8")

    return response, session_id, metadata, events, raw_stdout


async def _call_openai(
    prompt: str,
    role: str,
    settings: dict,
    schema: dict | None,
    system: str | None,
    resume: str | None,
    watch: bool,
    call_dir: str | None,
) -> tuple[object, str, dict, list, bytes]:
    """Call the OpenAI Responses API.

    Multi-turn continuity uses native previous_response_id — no
    client-side history tracking needed. store=True is required for
    previous_response_id to work; it gets set unconditionally.

    Returns (response, session_id, metadata, events, raw_stdout) —
    same shape as _run_codex_streaming.
    """
    client = _get_openai_client()

    model = settings.get("model", "gpt-5.5")
    effort = _normalize_effort(settings.get("effort", "xhigh"))

    kwargs: dict = {
        "model": model,
        "input": prompt,
        "reasoning": {"effort": effort},
        "store": True,
    }
    if system:
        # `instructions` is the documented top-level system-prompt
        # parameter on the Responses API. Unlike a developer message
        # in `input`, it's NOT carried by previous_response_id — so
        # it gets resent on every call (including resumes) to keep
        # behavior stable across the turn.
        kwargs["instructions"] = system
    if resume:
        kwargs["previous_response_id"] = resume

    if schema:
        # OpenAI strict JSON schemas require `additionalProperties: false`
        # on every object — the same helper the codex CLI path already
        # uses to satisfy this.
        oai_schema = _add_additional_properties(schema)
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": "response",
                "strict": True,
                "schema": oai_schema,
            }
        }

    tools = _openai_tools(settings)
    if tools:
        kwargs["tools"] = tools

    if watch:
        # Single hint that a call is in flight — the Responses API
        # streaming surface wasn't well-documented at the time this
        # was written, so the non-streaming create() is used here and
        # per-event progress is skipped. The full response still gets
        # saved to events.json.gz for post-hoc inspection.
        print(f"      [openai {model} working...]", file=sys.stderr)
    final_response = await client.responses.create(**kwargs)

    text = final_response.output_text or ""

    if schema:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"openai returned non-JSON for a schema'd call "
                f"(status={getattr(final_response, 'status', '?')!r}). "
                f"text={text[:500]!r} error={e}"
            )
        result = _sanitize_llm_output(data)
    else:
        result = _sanitize_llm_output(text)

    session_id = final_response.id

    # cached_tokens may live under either input_tokens_details or
    # prompt_tokens_details depending on SDK version — try both.
    usage = getattr(final_response, "usage", None)
    cached_tokens = 0
    if usage is not None:
        for attr in ("input_tokens_details", "prompt_tokens_details"):
            details = getattr(usage, attr, None)
            if details is not None:
                cached_tokens = getattr(details, "cached_tokens", 0) or 0
                if cached_tokens:
                    break
    metadata = {
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        "cached_input_tokens": cached_tokens,
        # OpenAI doesn't expose per-call cost on the response; downstream
        # summarization treats None as "unknown" and skips cost roll-up.
        "total_cost_usd": None,
        "tool_calls": _extract_openai_tool_calls(final_response),
    }

    events = [final_response.model_dump(exclude_none=True)]
    raw_stdout = json.dumps(events[0], default=str).encode("utf-8")

    return result, session_id, metadata, events, raw_stdout


def _extract_openai_tool_calls(response, *, truncate: bool = True) -> list[dict]:
    """Pull tool invocations out of a Responses API result.

    The Responses API returns one item per built-in tool call in
    response.output, with type discriminators like 'web_search_call' and
    'code_interpreter_call'. The shape varies per tool; the per-call
    entry captures just enough for post-hoc inspection without bloating
    the call log.
    """
    out = []
    items = getattr(response, "output", None) or []
    for item in items:
        itype = getattr(item, "type", None)
        if itype not in ("web_search_call", "code_interpreter_call", "function_call"):
            continue
        tool_name = itype.replace("_call", "")
        # Each tool surfaces its inputs under a different attribute —
        # web_search_call has `action` (with `query`), code_interpreter_call
        # has `code`, function_call has `arguments`. Best-effort capture.
        action = getattr(item, "action", None)
        if action is not None and hasattr(action, "model_dump"):
            action = action.model_dump(exclude_none=True)
        code = getattr(item, "code", None)
        arguments = getattr(item, "arguments", None)
        payload = action if action is not None else (code or arguments)
        out.append({
            "tool_name": tool_name,
            "input": _maybe_truncate(payload, TOOL_CALL_INPUT_LIMIT, truncate=truncate),
        })
    return out


async def _call_sdk_backend(
    prompt: str,
    role: str,
    backend: str,
    settings: dict,
    schema: dict | None,
    system: str | None,
    resume: str | None,
    watch: bool,
) -> tuple[str | dict, str | None]:
    """Run an SDK-backed LLM call with the same artifact + call_log
    shape the subprocess backends produce. Handles cache resume,
    pre/post-call artifact writes, and call_log slotting.
    """
    # ── Reserve a call_log slot (matches subprocess path) ────────
    log = call_log.get()
    if log is None:
        call_index = None
    else:
        call_index = len(log)
        log.append(None)

    base_artifact_dir = artifact_dir.get()
    call_dir: str | None = None
    if base_artifact_dir is not None and call_index is not None:
        call_dir = os.path.join(
            base_artifact_dir, f"call_{call_index:03d}_{role}",
        )
        os.makedirs(call_dir, exist_ok=True)

    # ── Cache resume (idempotent) ────────────────────────────────
    if call_dir is not None:
        cached = _try_resume_from_cache(call_dir, prompt, system, schema)
        if cached is not None:
            response, session_id, cache_meta = cached
            # Restore Anthropic session history from disk so subsequent
            # live calls can resume properly after a Theoria --resume.
            if backend == "anthropic" and session_id:
                history_path = os.path.join(call_dir, "history.json")
                if os.path.exists(history_path):
                    try:
                        with open(history_path) as f:
                            _anthropic_sessions[session_id] = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        pass  # next live call will warn
            print(
                f"[resume] cache hit on call_{call_index:03d}_{role}",
                file=sys.stderr,
            )
            if log is not None and call_index is not None:
                log[call_index] = cache_meta
            return response, session_id

    # ── Pre-call artifacts ───────────────────────────────────────
    # "cmd" here is the API call params (not an argv list) — close
    # analog of the subprocess cmd.json for post-hoc inspection.
    cmd_dict = {
        "backend": backend,
        "role": role,
        "model": settings.get("model"),
        "effort": settings.get("effort"),
        "schema": bool(schema),
        "tools": {
            "search": bool(settings.get("search")),
            "code_exec": bool(settings.get("code_exec")),
        },
        "resume": bool(resume),
        "watch": watch,
    }
    if call_dir:
        _write_artifact(
            os.path.join(call_dir, "cmd.json"),
            json.dumps(cmd_dict, indent=2, ensure_ascii=False),
        )
        _write_artifact(os.path.join(call_dir, "prompt.txt"), prompt)
        if system:
            _write_artifact(os.path.join(call_dir, "system.txt"), system)

    # ── Call ─────────────────────────────────────────────────────
    started_at = _utc_now_iso()
    started = time.perf_counter()
    raw_stdout = b""
    events: list[dict] = []
    try:
        if backend == "anthropic":
            response, session_id, provider_meta, events, raw_stdout = \
                await _call_anthropic(
                    prompt, role, settings, schema, system, resume,
                    watch, call_dir,
                )
        else:  # openai
            response, session_id, provider_meta, events, raw_stdout = \
                await _call_openai(
                    prompt, role, settings, schema, system, resume,
                    watch, call_dir,
                )
    except Exception as e:
        # Failure path — write a meta.json so the call_dir is
        # self-describing for post-mortem.
        if call_dir:
            try:
                _write_artifact(
                    os.path.join(call_dir, "meta.json"),
                    json.dumps({
                        "role": role,
                        "backend": backend,
                        "model": settings.get("model"),
                        "effort": settings.get("effort"),
                        "cmd": cmd_dict,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "failed": True,
                    }, indent=2, default=str),
                )
            except Exception:
                pass
        raise

    duration_ms = int(round((time.perf_counter() - started) * 1000))

    # ── Post-call artifacts ──────────────────────────────────────
    artifact_paths: dict[str, str] = {}
    if call_dir:
        response_text = response if isinstance(response, str) else json.dumps(response, indent=2)
        artifact_paths["artifact_dir"] = call_dir
        artifact_paths["cmd_path"] = os.path.join(call_dir, "cmd.json")
        artifact_paths["prompt_path"] = os.path.join(call_dir, "prompt.txt")
        if system:
            artifact_paths["system_path"] = os.path.join(call_dir, "system.txt")
        if raw_stdout:
            _write_artifact(
                os.path.join(call_dir, "stdout.jsonl"),
                raw_stdout, gzip_it=True,
            )
            artifact_paths["stdout_path"] = os.path.join(call_dir, "stdout.jsonl.gz")
        _write_artifact(
            os.path.join(call_dir, "response.txt"), response_text,
        )
        artifact_paths["response_path"] = os.path.join(call_dir, "response.txt")
        if events:
            _write_artifact(
                os.path.join(call_dir, "events.json"),
                json.dumps(events, ensure_ascii=False, default=str),
                gzip_it=True,
            )
            artifact_paths["events_path"] = os.path.join(call_dir, "events.json.gz")
        if provider_meta.get("tool_calls"):
            _write_artifact(
                os.path.join(call_dir, "tool_calls.json"),
                json.dumps(provider_meta["tool_calls"], ensure_ascii=False,
                           indent=2, default=str),
            )
            artifact_paths["tool_calls_path"] = os.path.join(call_dir, "tool_calls.json")

    # ── call_log entry ───────────────────────────────────────────
    call_meta = None
    if log is not None:
        response_text = response if isinstance(response, str) else json.dumps(response)
        call_meta = {
            "role": role,
            "backend": backend,
            "model": settings.get("model"),
            "model_config": settings.get("model"),
            "effort": settings.get("effort"),
            "started_at": started_at,
            "ended_at": _utc_now_iso(),
            "duration_ms": duration_ms,
            "session_id": session_id,
            "resumed": bool(resume),
            "has_schema": schema is not None,
            "returncode": 0,
            "argv_hash": hashlib.sha256(
                json.dumps(cmd_dict, sort_keys=True).encode()
            ).hexdigest()[:16],
            "sandboxed": False,  # API path uses provider's sandbox
            "container_id": None,
            "container_cwd": None,
            "image_id": None,
            "prompt": _truncate(prompt, 8000),
            "system": _truncate(system or "", 8000),
            "response": _truncate(response_text, 8000),
            **artifact_paths,
            **provider_meta,
        }
        log[call_index] = call_meta

    if call_dir and call_meta is not None:
        _write_artifact(
            os.path.join(call_dir, "meta.json"),
            json.dumps(call_meta, indent=2, default=str),
        )

    return response, session_id


# ── Main entry point ────────────────────────────────────────────

async def llm(
    prompt: str,
    *,
    role: str = "solver",
    schema: dict | None = None,
    system: str | None = None,
    config: dict = {},
    watch: bool = False,
    resume: str | None = None,
) -> tuple[str | dict, str | None]:
    """Call an LLM. Backend (claude/codex) determined by config for the role.

    Returns (response, session_id). Pass resume=session_id on a later call to
    continue the same conversation.

    When watch=True, streams events to stderr in real time.

    When the `artifact_dir` contextvar is set, every call writes:
        <artifact_dir>/call_NNN_<role>/
            cmd.json              — the exact argv run
            prompt.txt            — the full user prompt
            system.txt            — the full system prompt (if any)
            stdout.jsonl[.gz]     — raw provider stdout
            stderr.txt[.gz]       — raw provider stderr
            response.txt          — parsed response text (or JSON)
            events.json.gz        — parsed provider event stream
            tool_calls.json       — untruncated tool call I/O
            meta.json             — call-level metadata (duration, tokens, etc.)

    The call_log entry keeps its existing truncated fields as previews;
    full source of truth is the files on disk.
    """
    settings = config.get(role, {})
    backend = settings.get("backend", "claude")

    # Apply backend-specific defaults (config values take precedence)
    if backend == "codex":
        settings.setdefault("model", "gpt-5.5")
        settings.setdefault("effort", "xhigh")
        settings.setdefault("sandbox", "read-only")
        settings.setdefault("search", True)
    elif backend == "claude":
        settings.setdefault("model", "opus")
        settings.setdefault("effort", "max")
    elif backend == "anthropic":
        settings.setdefault("model", "claude-opus-4-7")
        settings.setdefault("effort", "xhigh")
    elif backend == "openai":
        settings.setdefault("model", "gpt-5.5")
        settings.setdefault("effort", "xhigh")

    # SDK backends bypass the entire subprocess/docker/watchdog
    # machinery. They produce the same artifact format and call_log
    # entries so downstream consumers don't care which path ran.
    if backend in ("anthropic", "openai"):
        return await _call_sdk_backend(
            prompt, role, backend, settings, schema, system, resume, watch,
        )

    # ── Reserve a slot in the call log ───────────────────────────
    #
    # Parallel judges under asyncio.gather all share the same call_log
    # list via contextvars. If we allocated the index at the END of the
    # call (when we append call_meta), two judges running concurrently
    # would race: both would read len(log) at append time and collide.
    # Reserving a placeholder synchronously at the start — before any
    # await — guarantees a unique index per call. We fill it in later.
    log = call_log.get()
    if log is None:
        call_index = None
    else:
        call_index = len(log)
        log.append(None)  # reserve slot; replaced with call_meta below

    # ── Set up the per-call artifact directory ───────────────────
    base_artifact_dir = artifact_dir.get()
    call_dir: str | None = None
    if base_artifact_dir is not None and call_index is not None:
        call_dir = os.path.join(
            base_artifact_dir, f"call_{call_index:03d}_{role}",
        )
        os.makedirs(call_dir, exist_ok=True)

    # ── Resume from cache (idempotent) ───────────────────────────
    # If we're resuming a prior run, this call_dir may already contain
    # a successful response. Reuse it without making a new LLM call —
    # but only after verifying the saved prompt matches what we'd send
    # now. Mismatch raises (state drift) rather than silently using a
    # stale cached response.
    if call_dir is not None:
        cached = _try_resume_from_cache(call_dir, prompt, system, schema)
        if cached is not None:
            response, session_id, cache_meta = cached
            print(
                f"[resume] cache hit on call_{call_index:03d}_{role}",
                file=sys.stderr,
            )
            if log is not None and call_index is not None:
                log[call_index] = cache_meta
            return response, session_id

    # ── Docker sandbox wiring ────────────────────────────────────
    # When the harness has started a per-problem container and set the
    # sandbox_container contextvar, every call runs inside that
    # container via `docker exec`.
    #
    # All calls in a problem share cwd = /workspace. This matches the
    # SWE-bench per-task workspace pattern and — crucially — keeps
    # `--resume` working: claude stores per-project session rollouts
    # under ~/.claude/projects/<cwd-encoded>/... so changing cwd
    # between calls makes the resumed session un-findable. We tried
    # per-call /workspace/call_NNN_<role>/ subdirs first; the repair
    # loop broke with "No conversation found with session ID". All
    # agent outputs still get captured via the post-run `docker cp
    # /workspace` snapshot, and the per-call artifact dirs on the
    # host already give us "which call wrote which bytes" attribution.
    container_id = sandbox_container.get()
    image_id = sandbox_image_id.get()
    sandboxed = container_id is not None
    container_call_cwd: str | None = "/workspace" if sandboxed else None

    schema_file = None
    raw_stdout: bytes = b""
    raw_stderr: bytes = b""
    events: list[dict] = []
    cmd: list[str] = []
    try:
        # Build command
        if backend == "claude":
            cmd = _build_claude_cmd(
                prompt, settings, schema, system, watch, resume,
                sandboxed=sandboxed,
            )
        elif backend == "codex":
            if schema:
                codex_schema = _add_additional_properties(schema)
                schema_json = json.dumps(codex_schema).encode("utf-8")
                if sandboxed:
                    # Host tempfiles aren't visible to the container.
                    # Write the schema into a per-call path inside
                    # /workspace (cwd is /workspace; the filename is
                    # unique per call so parallel judges don't stomp
                    # on each other's schema files).
                    schema_file = (
                        f"/workspace/.call_{call_index:03d}_{role}_schema.json"
                    )
                    write = await asyncio.create_subprocess_exec(
                        "docker", "exec", "-i", container_id,
                        "sh", "-c", f"cat > {schema_file}",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, w_err = await write.communicate(input=schema_json)
                    if write.returncode != 0:
                        raise RuntimeError(
                            f"failed to write output schema into container: "
                            f"{w_err.decode(errors='replace')[:300]}"
                        )
                else:
                    f = tempfile.NamedTemporaryFile(
                        mode="wb", suffix=".json", delete=False,
                    )
                    f.write(schema_json)
                    f.close()
                    schema_file = f.name
            cmd = _build_codex_cmd(
                prompt, settings, schema_file, system, resume,
                sandboxed=sandboxed, role=role,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Wrap with `docker exec -w <per-call-cwd> <container>` when
        # sandboxed. The CLI's own cwd becomes the per-call subdir, so
        # scratch files land there.
        if sandboxed:
            cmd = [
                "docker", "exec",
                "-w", container_call_cwd,
                container_id,
            ] + cmd

        # ── Save pre-call artifacts ──────────────────────────────
        #
        # Write these BEFORE running the subprocess. That way, if the
        # subprocess hangs or the process is killed, we still know
        # exactly what we asked for.
        if call_dir:
            _write_artifact(
                os.path.join(call_dir, "cmd.json"),
                json.dumps(cmd, indent=2, ensure_ascii=False),
            )
            _write_artifact(
                os.path.join(call_dir, "prompt.txt"), prompt,
            )
            if system:
                _write_artifact(
                    os.path.join(call_dir, "system.txt"), system,
                )

        # Retry loop for watchdog-killed subprocesses. Codex CLI
        # sometimes hangs indefinitely on a specific HTTP connection;
        # the watchdog kills it, and we retry from scratch with a
        # fresh subprocess (and therefore fresh codex/claude session).
        # Up to WATCHDOG_RETRY_MAX retries; non-watchdog failures are
        # raised immediately without retry.
        watchdog_attempts = 0
        call_label = (f"call_{call_index:03d}_{role}"
                      if call_index is not None else role)
        while True:
            started_at = _utc_now_iso()
            started = time.perf_counter()

            # Run
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=SUBPROCESS_STREAM_LIMIT,
            )

            # Hang detection. Sandboxed-only: we monitor per-process
            # CPU ticks via /proc/<pid>/stat inside the container. The
            # marker is a unique-per-call token (the schema file path)
            # so we identify the right subprocess when many judges run
            # in parallel; for calls without a schema (solver) we fall
            # back to comm-name matching, which is fine because such
            # calls don't run in parallel within one container.
            last_event_ref = [time.monotonic()]
            watchdog_killed = [False]
            watchdog_task = None
            if container_id is not None:
                marker = schema_file if schema_file else None
                watchdog_task = asyncio.create_task(_watchdog(
                    proc, container_id, watchdog_killed,
                    backend=backend, marker=marker, label=role,
                ))

            raw_stdout = b""
            raw_stderr = b""
            try:
                if watch:
                    # Stream mode: read line by line, print events
                    if backend == "claude":
                        response, session_id, provider_meta, events, raw_stdout = \
                            await _run_claude_streaming(
                                proc, schema, last_event_ref=last_event_ref,
                            )
                    else:
                        response, session_id, provider_meta, events, raw_stdout = \
                            await _run_codex_streaming(
                                proc, schema, last_event_ref=last_event_ref,
                            )
                else:
                    # Batch mode: collect all output, parse at end
                    stdout_bytes, stderr_bytes = await proc.communicate()
                    raw_stdout = stdout_bytes
                    raw_stderr = stderr_bytes

                    if proc.returncode != 0:
                        raise RuntimeError(
                            _format_failure(backend, proc.returncode,
                                            stderr_bytes, stdout_bytes)
                        )

                    output = stdout_bytes.decode()

                    if backend == "claude":
                        response, session_id, provider_meta, events = \
                            _parse_claude_output(output, schema)
                    else:
                        response, session_id, provider_meta, events = \
                            _parse_codex_output(output, schema)
                # Success — exit retry loop.
                break
            except Exception:
                # Was this a watchdog kill (transient hang) and do we
                # have retries left? If so, swallow and retry.
                if (watchdog_killed[0]
                        and watchdog_attempts < WATCHDOG_RETRY_MAX):
                    watchdog_attempts += 1
                    print(
                        f"[retry] watchdog killed {call_label}; "
                        f"retrying attempt {watchdog_attempts + 1}/"
                        f"{WATCHDOG_RETRY_MAX + 1}",
                        file=sys.stderr,
                    )
                    continue  # finally runs, then loop iterates
                # Non-watchdog failure or out of retries: propagate.
                raise
            finally:
                # Cancel the watchdog so it doesn't leak past this
                # attempt. cancel() is idempotent.
                if watchdog_task is not None:
                    watchdog_task.cancel()
                    try:
                        await watchdog_task
                    except (asyncio.CancelledError, Exception):
                        pass
                # Save raw stdout/stderr for the most recent attempt.
                # On retry the previous attempt's data is overwritten —
                # last attempt wins. The retry message printed above
                # is the only signal that earlier attempts existed.
                if call_dir:
                    if raw_stdout:
                        _write_artifact(
                            os.path.join(call_dir, "stdout.jsonl"),
                            raw_stdout, gzip_it=True,
                        )
                    if raw_stderr:
                        _write_artifact(
                            os.path.join(call_dir, "stderr.txt"),
                            raw_stderr, gzip_it=True,
                        )

        duration_ms = int(round((time.perf_counter() - started) * 1000))

        # ── Save post-parse artifacts ────────────────────────────
        artifact_paths: dict[str, str] = {}
        if call_dir:
            response_text = response if isinstance(response, str) else json.dumps(response, indent=2)
            artifact_paths["artifact_dir"] = call_dir
            artifact_paths["cmd_path"] = os.path.join(call_dir, "cmd.json")
            artifact_paths["prompt_path"] = os.path.join(call_dir, "prompt.txt")
            if system:
                artifact_paths["system_path"] = os.path.join(call_dir, "system.txt")
            if raw_stdout:
                artifact_paths["stdout_path"] = os.path.join(call_dir, "stdout.jsonl.gz")
            if raw_stderr:
                artifact_paths["stderr_path"] = os.path.join(call_dir, "stderr.txt.gz")

            _write_artifact(
                os.path.join(call_dir, "response.txt"), response_text,
            )
            artifact_paths["response_path"] = os.path.join(call_dir, "response.txt")

            if events:
                _write_artifact(
                    os.path.join(call_dir, "events.json"),
                    json.dumps(events, ensure_ascii=False, default=str),
                    gzip_it=True,
                )
                artifact_paths["events_path"] = os.path.join(call_dir, "events.json.gz")

            # Full untruncated tool calls — parallel to the truncated
            # ones in call_meta. The truncated list stays in call_meta
            # for scannability; the full list lives on disk.
            if backend == "claude":
                full_tools = _extract_claude_tool_calls(events, truncate=False)
            else:
                full_tools = _extract_codex_tool_calls(events, truncate=False)
            if full_tools:
                _write_artifact(
                    os.path.join(call_dir, "tool_calls.json"),
                    json.dumps(full_tools, ensure_ascii=False,
                               indent=2, default=str),
                )
                artifact_paths["tool_calls_path"] = os.path.join(call_dir, "tool_calls.json")

        # Build call_meta (truncated previews + artifact paths).
        call_meta = None
        if log is not None:
            response_text = response if isinstance(response, str) else json.dumps(response)
            call_meta = {
                "role": role,
                "backend": backend,
                # Record the effective model actually sent to the CLI,
                # not the configured alias — these differ for codex
                # (config model="opus" gets translated to "gpt-5.5").
                "model": _effective_model(cmd) or settings.get("model"),
                "model_config": settings.get("model"),
                "effort": settings.get("effort"),
                "started_at": started_at,
                "ended_at": _utc_now_iso(),
                "duration_ms": duration_ms,
                "session_id": session_id,
                "resumed": bool(resume),
                "has_schema": schema is not None,
                "returncode": proc.returncode,
                "argv_hash": _cmd_hash(cmd),
                # Which sandbox this call ran in (if any). Enables
                # post-hoc reasoning about the container image version.
                "sandboxed": container_id is not None,
                "container_id": container_id,
                "container_cwd": container_call_cwd,
                "image_id": image_id,
                "prompt": _truncate(prompt, 8000),
                "system": _truncate(system or "", 8000),
                "response": _truncate(response_text, 8000),
                **artifact_paths,
                **provider_meta,
            }
            # Fill the slot we reserved at the top.
            log[call_index] = call_meta

        # Also write a per-call meta.json artifact so a single call
        # dir is self-describing without reading the batch result.
        if call_dir and call_meta is not None:
            _write_artifact(
                os.path.join(call_dir, "meta.json"),
                json.dumps(call_meta, indent=2, default=str),
            )

        return response, session_id

    except Exception as e:
        # Failure path: at minimum record enough to reconstruct what
        # happened. Raw stdout/stderr were already saved in the inner
        # finally. Write a failure meta.json so the call dir is
        # self-describing for post-mortem.
        if call_dir:
            try:
                _write_artifact(
                    os.path.join(call_dir, "meta.json"),
                    json.dumps({
                        "role": role,
                        "backend": backend,
                        "model": settings.get("model"),
                        "effort": settings.get("effort"),
                        "argv": cmd,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "failed": True,
                    }, indent=2, default=str),
                )
            except Exception:
                pass
        # Leave the placeholder None in call_log so an operator can
        # spot it (and the partial save will persist it).
        raise

    finally:
        # Only unlink host-side schema tempfiles. When sandboxed the
        # schema lives at an in-container path; it's cleaned up with
        # the per-call workspace dir when the container is destroyed.
        if schema_file and not sandboxed:
            try:
                os.unlink(schema_file)
            except OSError:
                pass
