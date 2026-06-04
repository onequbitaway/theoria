"""Render a saved run JSON into readable markdown.

This module backs `theoria show`. For each problem it writes (under
`dist/problems/` by default):
  <label>_answer.md  — the judge-passed answer + proof (verified runs only)
  <label>_trace.md   — full trace: attempts, proof, judge verdicts, filters

Normal use is `theoria show runs/<run>.json`. It can also be run directly
on a specific run file:
    python3 export_problem.py --run-file runs/hle_<ts>.json --label myrun
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path


def latest_run_file(pnum: int, runs_dir: str = "runs") -> str | None:
    matches = []
    for f in glob.glob(f"{runs_dir}/hle_p{pnum}_*.json"):
        m = re.match(rf"{re.escape(runs_dir)}/hle_p{pnum}_(\d{{8}})_(\d{{6}})\.json$", f)
        if m:
            matches.append((m.group(1) + m.group(2), f))
    if not matches:
        return None
    return sorted(matches)[-1][1]


def load_run(path: str) -> dict | None:
    try:
        data = json.load(open(path))
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    return data[0]


def fmt_state(state) -> str:
    if not state:
        return "(empty)"
    lines = []
    for i, entry in enumerate(state):
        lines.append(f"  [{i}] {entry}")
    return "\n".join(lines)


def _header_id(pnum) -> str:
    return f"p{pnum}" if isinstance(pnum, int) else str(pnum)


def answer_md(pnum, d: dict, basename: str) -> str:
    lines: list[str] = []
    lines.append(f"# {_header_id(pnum)} — Judge-Passed Answer")
    lines.append("")
    lines.append(f"**Problem ID:** `{d.get('id', '?')}`  ")
    lines.append(f"**Category:** {d.get('category', '?')}  ")
    lines.append(f"**Source file:** `runs/{basename}`")
    lines.append("")
    lines.append("## Problem")
    lines.append("")
    lines.append(d.get("problem") or "(no problem text)")
    lines.append("")
    lines.append("## Answer")
    lines.append("")
    lines.append("```")
    lines.append((d.get("answer") or "").strip())
    lines.append("```")
    lines.append("")
    expected = (d.get("expected") or "").strip()
    if expected:
        lines.append(f"**Expected (answer key):** `{expected}`")
        lines.append("")
    lines.append("## Judge review")
    lines.append("")
    lines.append(f"- Judge-passed (every step cleared the judges): **{d.get('verified')}**")
    if expected:
        lines.append(f"- Naive string match vs answer key: `{d.get('correct')}`")
    convs = d.get("verified_under_assumptions") or []
    if convs:
        lines.append("")
        lines.append("**Conventions invoked during verification:**")
        for c in convs:
            lines.append(f"- {c}")
    lines.append("")
    lines.append("## Proof")
    lines.append("")
    proof = d.get("proof") or {}
    steps = proof.get("steps") or []
    init = proof.get("initial_state") or []
    if init:
        lines.append(f"**Initial state:**")
        lines.append("")
        lines.append(f"```\n{fmt_state(init)}\n```")
        lines.append("")
    for i, s in enumerate(steps, 1):
        jt = s.get("justification_type", "?")
        just = s.get("justification", "")
        st = s.get("state", [])
        lines.append(f"### Step {i} — `{jt}`")
        lines.append("")
        lines.append(just)
        lines.append("")
        lines.append("**State after step:**")
        lines.append("")
        lines.append(f"```\n{fmt_state(st)}\n```")
        lines.append("")
    metrics = d.get("metrics") or {}
    if metrics:
        lines.append("## Metrics")
        lines.append("")
        cost = metrics.get("total_cost_usd")
        dur_ms = metrics.get("problem_duration_ms") or 0
        tok = metrics.get("total_tokens") or 0
        n_calls = metrics.get("num_calls") or 0
        lines.append(f"- Duration: {dur_ms / 60000:.1f} min")
        lines.append(f"- LLM calls: {n_calls}")
        lines.append(f"- Total tokens: {tok:,}")
        if cost is not None:
            lines.append(f"- Cost: ${cost:.2f}")
    return "\n".join(lines) + "\n"


def trace_md(pnum, d: dict, basename: str) -> str:
    lines: list[str] = []
    lines.append(f"# {_header_id(pnum)} — Full Reasoning Trace")
    lines.append("")
    lines.append(f"**Problem ID:** `{d.get('id', '?')}`  ")
    lines.append(f"**Category:** {d.get('category', '?')}  ")
    lines.append(f"**Source file:** `runs/{basename}`  ")
    _outcome = f"**Outcome:** judge-passed={d.get('verified')}"
    if (d.get("expected") or "").strip():
        _outcome += f", correct={d.get('correct')}"
    lines.append(_outcome + "  ")
    if d.get("error"):
        lines.append(f"**Error:** `{d['error']}`")
    lines.append("")
    lines.append("## Problem")
    lines.append("")
    lines.append(d.get("problem") or "(no problem text)")
    lines.append("")
    expected = (d.get("expected") or "").strip()
    given = (d.get("answer") or "").strip()
    if expected:
        lines.append("## Final answer vs expected")
        lines.append("")
        lines.append(f"- **Expected:** `{expected}`")
        lines.append(f"- **System:** `{given or '(no answer)'}`")
    else:
        lines.append("## Final answer")
        lines.append("")
        lines.append(f"- **System:** `{given or '(no answer)'}`")
    lines.append("")

    # Attempts timeline
    attempts = d.get("attempts") or []
    lines.append(f"## Attempts ({len(attempts)})")
    lines.append("")
    for i, a in enumerate(attempts, 1):
        phase = a.get("phase", "?")
        lines.append(f"### Attempt {a.get('attempt', i)} — phase=`{phase}`")
        lines.append("")
        if phase == "formalizer_reject":
            reason = a.get("reject_reason", "")
            lines.append("Formalizer rejected the solver's answer.")
            lines.append("")
            lines.append("**Reject reason:**")
            lines.append("")
            lines.append(f"> {reason}")
            lines.append("")
            continue
        # phase == verify
        all_ok = a.get("all_ok")
        lines.append(f"- all_ok: **{all_ok}**")
        proof = a.get("proof") or {}
        steps = proof.get("steps") or []
        init = proof.get("initial_state") or []
        state0 = a.get("state0_verdict") or {}
        verdicts = a.get("verdicts") or []
        pedantry = a.get("pedantry") or []
        conventions = a.get("conventions") or []
        if init:
            lines.append("")
            lines.append(f"**Initial state:**")
            lines.append("")
            lines.append(f"```\n{fmt_state(init)}\n```")
        if state0:
            lines.append("")
            lines.append(f"**State-0 audit:** accepted={state0.get('accepted')}  ")
            reason = (state0.get("reason") or "")[:500]
            lines.append(f"> {reason}")
        for j, s in enumerate(steps, 1):
            jt = s.get("justification_type", "?")
            just = s.get("justification", "")
            st = s.get("state", [])
            v = next((v for v in verdicts if v and v.get("step_number") == j), None)
            lines.append("")
            lines.append(f"#### Step {j} — `{jt}`")
            lines.append("")
            lines.append(f"**Justification:** {just}")
            lines.append("")
            lines.append(f"**State after step:**")
            lines.append("")
            lines.append(f"```\n{fmt_state(st)}\n```")
            if v:
                lines.append("")
                acc = "✓" if v.get("accepted") else "✗"
                role = v.get("role", "?")
                reason = (v.get("reason") or "").strip()
                lines.append(f"**Judge ({role}): {acc}**")
                lines.append("")
                lines.append(f"> {reason}")
            # pedantry for this step
            ped_rows = [p for p in pedantry if p.get("step_number") == j]
            for pr in ped_rows:
                lines.append("")
                pedtag = "PEDANTIC (override)" if pr.get("is_pedantic") else "LEGITIMATE"
                lines.append(f"**Pedantry filter:** {pedtag}")
                lines.append(f"- Original reason: {(pr.get('original_reason') or '')[:300]}")
                if pr.get("pedantry_reason"):
                    lines.append(f"- Filter reasoning: {pr['pedantry_reason'][:300]}")
        if conventions:
            lines.append("")
            lines.append("**Conventions lifted:**")
            for c in conventions:
                lines.append(f"- {c}")

    # Call-level summary
    calls = d.get("calls") or []
    lines.append("")
    lines.append(f"## LLM calls ({len(calls)})")
    lines.append("")
    lines.append("| # | role | model | duration (s) | tokens | cost |")
    lines.append("|---|---|---|---|---|---|")
    for i, c in enumerate(calls):
        role = c.get("role", "?")
        model = c.get("model", "?")
        dur_s = (c.get("duration_ms") or 0) / 1000.0
        tok = (c.get("input_tokens") or 0) + (c.get("output_tokens") or 0)
        cost = c.get("total_cost_usd")
        cost_s = f"${cost:.3f}" if cost is not None else "—"
        lines.append(f"| {i} | {role} | {model} | {dur_s:.1f} | {tok:,} | {cost_s} |")

    metrics = d.get("metrics") or {}
    if metrics:
        lines.append("")
        lines.append("## Totals")
        lines.append("")
        cost = metrics.get("total_cost_usd")
        dur_ms = metrics.get("problem_duration_ms") or 0
        lines.append(f"- Duration: {dur_ms / 60000:.1f} min")
        lines.append(f"- Input tokens: {metrics.get('total_input_tokens') or 0:,}")
        lines.append(f"- Output tokens: {metrics.get('total_output_tokens') or 0:,}")
        lines.append(f"- Cache-read tokens: {metrics.get('total_cache_read_input_tokens') or 0:,}")
        if cost is not None:
            lines.append(f"- Total cost: ${cost:.2f}")

    return "\n".join(lines) + "\n"


def final_md(label, d: dict, basename: str) -> str:
    """Render just the three pieces: final solver output, final formalizer
    proof, and judge rejections on that final proof. No attempt history,
    no pedantry / convention reasoning, no metrics — just the three
    artifacts that constitute the system's output."""
    lines: list[str] = []
    lines.append(f"# {_header_id(label)} — Final Outputs")
    lines.append("")
    lines.append(f"**Source file:** `runs/{basename}`  ")
    lines.append(f"**Judge-passed:** {d.get('verified')}")
    lines.append("")
    lines.append("## 1. Final Solver Output")
    lines.append("")
    lines.append("```")
    lines.append((d.get("solution") or "(no solver output)").strip())
    lines.append("```")
    lines.append("")
    lines.append("## 2. Final Formalizer Output")
    lines.append("")
    proof = d.get("proof") or {}
    init = proof.get("initial_state") or []
    steps = proof.get("steps") or []
    if init:
        lines.append("**Initial state (state 0):**")
        lines.append("")
        lines.append(f"```\n{fmt_state(init)}\n```")
        lines.append("")
    for i, s in enumerate(steps, 1):
        jt = s.get("justification_type", "?")
        just = s.get("justification", "")
        st = s.get("state", [])
        lines.append(f"### Step {i} — `{jt}`")
        lines.append("")
        lines.append(f"**Justification:** {just}")
        lines.append("")
        lines.append("**State after step:**")
        lines.append("")
        lines.append(f"```\n{fmt_state(st)}\n```")
        lines.append("")
    lines.append("## 3. Judge Rejections on Final Proof")
    lines.append("")
    # `verdicts` at the top level is the LAST attempt's verdicts.
    verdicts = d.get("verdicts") or []
    # state0 verdict from the last verify attempt
    last_verify = next(
        (a for a in reversed(d.get("attempts") or [])
         if a.get("phase") == "verify"),
        None,
    )
    s0 = (last_verify or {}).get("state0_verdict") or {}
    if s0 and not s0.get("accepted"):
        lines.append("### State 0 — REJECTED")
        lines.append("")
        lines.append(f"{s0.get('reason', '').strip()}")
        lines.append("")
    n_rejected = 0
    for i, v in enumerate(verdicts, 1):
        if not isinstance(v, dict):
            continue
        if v.get("accepted"):
            continue
        n_rejected += 1
        lines.append(f"### Step {i} — REJECTED")
        lines.append("")
        lines.append(v.get("reason", "").strip())
        lines.append("")
    if n_rejected == 0 and not (s0 and not s0.get("accepted")):
        lines.append("(no rejections — every step accepted)")
        lines.append("")
    return "\n".join(lines) + "\n"


def export(pnum: int, out_dir: Path, verified_only: bool = False) -> tuple[bool, bool]:
    """Returns (wrote_answer, wrote_trace)."""
    path = latest_run_file(pnum)
    if path is None:
        return (False, False)
    d = load_run(path)
    if d is None:
        return (False, False)
    basename = os.path.basename(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrote_answer = False
    wrote_trace = False

    if d.get("verified"):
        (out_dir / f"p{pnum}_answer.md").write_text(answer_md(pnum, d, basename))
        wrote_answer = True

    if not verified_only or d.get("verified"):
        (out_dir / f"p{pnum}_trace.md").write_text(trace_md(pnum, d, basename))
        wrote_trace = True

    return (wrote_answer, wrote_trace)


def export_file(run_path: str, label: str, out_dir: Path,
                verified_only: bool = False,
                final_only: bool = False) -> tuple[bool, bool, bool]:
    """Export a single run JSON by explicit path. Returns
    (wrote_answer, wrote_trace, wrote_final)."""
    d = load_run(run_path)
    if d is None:
        return (False, False, False)
    basename = os.path.basename(run_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrote_answer = False
    wrote_trace = False
    wrote_final = False

    if final_only:
        # Only the 3-section artifact: solver, formalizer, rejections.
        (out_dir / f"{label}_final.md").write_text(final_md(label, d, basename))
        wrote_final = True
        return (wrote_answer, wrote_trace, wrote_final)

    if d.get("verified"):
        (out_dir / f"{label}_answer.md").write_text(answer_md(label, d, basename))
        wrote_answer = True

    if not verified_only or d.get("verified"):
        (out_dir / f"{label}_trace.md").write_text(trace_md(label, d, basename))
        wrote_trace = True

    return (wrote_answer, wrote_trace, wrote_final)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-problem markdown files.")
    parser.add_argument("--pnum", type=int, default=None, help="Single HLE problem number")
    parser.add_argument(
        "--run-file", default=None,
        help="Explicit path to a run JSON (hle or custom). Use "
             "--label to set the filename stem.",
    )
    parser.add_argument(
        "--label", default=None,
        help="Filename stem when using --run-file (default: derived from "
             "the run JSON's problem id).",
    )
    parser.add_argument("--out-dir", default="dist/problems", help="Output directory")
    parser.add_argument(
        "--verified-only", action="store_true",
        help="Only export runs that were verified (default: export traces for all, "
             "answer files only for verified)",
    )
    parser.add_argument(
        "--final", action="store_true",
        help="Only emit <label>_final.md containing exactly three sections: "
             "(1) final solver output, (2) final formalizer proof, (3) judge "
             "rejections on that final proof. Skips trace and answer files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.run_file is not None:
        # Direct-path mode — works for any run JSON.
        d = load_run(args.run_file)
        if d is None:
            print(f"Could not load {args.run_file}")
            return
        label = args.label or d.get("id") or Path(args.run_file).stem
        # Sanitize the label for filesystem use (no slashes, etc.).
        label = re.sub(r"[^\w.-]", "_", str(label))
        wa, wt, wf = export_file(args.run_file, label, out_dir,
                                  verified_only=args.verified_only,
                                  final_only=args.final)
        files = []
        if wa: files.append(f"{label}_answer.md")
        if wt: files.append(f"{label}_trace.md")
        if wf: files.append(f"{label}_final.md")
        if files:
            print(f"Wrote {len(files)} file(s) to {out_dir}/: {', '.join(files)}")
        else:
            print(f"Nothing written (verified_only={args.verified_only}, "
                  f"verified={d.get('verified')}).")
        return

    if args.pnum is not None:
        pnums = [args.pnum]
    else:
        pnums = set()
        for f in glob.glob("runs/hle_p*.json"):
            m = re.match(r"runs/hle_p(\d+)_", f)
            if m:
                pnums.add(int(m.group(1)))
        pnums = sorted(pnums)

    n_ans, n_trace = 0, 0
    for p in pnums:
        wa, wt = export(p, out_dir, verified_only=args.verified_only)
        if wa: n_ans += 1
        if wt: n_trace += 1

    print(f"Wrote {n_ans} answer files and {n_trace} trace files to {out_dir}/")


if __name__ == "__main__":
    main()
