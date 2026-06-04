"""Theoria — verified reasoning.

1. Solve: LLM answers the question (session kept open).
2. Formalize: LLM writes the proof (session kept open).
3. Judge: three LLMs verify each justification in parallel.
4. Repair loop on failure: continue formalizer session with the failed
   verdicts; formalizer either fixes the proof or escalates to the
   solver (continuing solver session) for a new answer.

Usage:
    python pipeline.py "What is 2 + 2?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# ── Models ──────────────────────────────────────────────────────

@dataclass
class Step:
    state: list[str]
    justification_type: str  # citation | problem_given | computation
    justification: str

@dataclass
class Proof:
    initial_state: list[str]
    steps: list[Step]

@dataclass
class Verdict:
    accepted: bool
    reason: str

# ── JSON Schemas ────────────────────────────────────────────────

# The base proof schema used by general (math/science) formalization.
# Each step has state + justification_type + justification.
PROOF_SCHEMA = {
    "type": "object",
    "properties": {
        "initial_state": {"type": "array", "items": {"type": "string"}},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "state": {"type": "array", "items": {"type": "string"}},
                    "justification_type": {"type": "string", "enum": ["citation", "problem_given", "computation"]},
                    "justification": {"type": "string"},
                },
                "required": ["state", "justification_type", "justification"],
            },
        },
    },
    "required": ["initial_state", "steps"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "accepted": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["accepted", "reason"],
}


def _formalizer_decision_schema() -> dict:
    """Wrap the proof schema in the formalizer's action/reject_reason
    envelope: the formalizer either returns a proof or rejects the
    solution with a reason for the solver."""
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["proof", "reject"]},
            "proof": PROOF_SCHEMA,
            "reject_reason": {"type": "string"},
        },
        "required": ["action", "proof", "reject_reason"],
    }

# The pedantry filter's decision on a single failed verdict.
PEDANTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "is_pedantic": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_pedantic", "reason"],
}

# The convention-lift judge's decision on a legitimate rejection.
# Runs after pedantry; if it lifts, the rejected step becomes accepted
# with the added convention recorded as an explicit assumption.
CONVENTION_LIFT_SCHEMA = {
    "type": "object",
    "properties": {
        "can_lift": {"type": "boolean"},
        "convention": {"type": "string"},
        "source": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["can_lift", "convention", "source", "reasoning"],
}


def agent_prompt(role: str) -> str:
    """Return the system prompt for a role, optionally prefixed with
    the shared `_preamble` block when the role sets `preamble: true`.
    Used to factor out environment descriptions or other content that
    should appear at the top of several roles' prompts without
    duplicating the text in each role."""
    p = agent_settings(role).get("prompt", "")
    if agent_settings(role).get("preamble"):
        pre = CONFIG.get("_preamble", "")
        if pre:
            p = f"{pre}\n\n{p}"
    return p

# ── Config ─────────────────────────────────────────────────────

DEFAULTS_PATH = Path(__file__).parent / "configs" / "defaults.yaml"

def load_config(override_paths=None) -> dict:
    """Load defaults.yaml and stack zero or more overrides on top.

    `override_paths` accepts None, a single path (str/Path), or a list
    of paths. When a list, overrides are applied in order — later
    entries win on conflicts. Role entries are dict-merged into
    defaults; top-level scalars like `_preamble` are replaced wholesale
    (can't .update() a string).
    """
    with open(DEFAULTS_PATH) as f:
        config = yaml.safe_load(f)

    if override_paths is None:
        paths = []
    elif isinstance(override_paths, (str, Path)):
        paths = [override_paths]
    else:
        paths = list(override_paths)

    for path in paths:
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        for key, value in overrides.items():
            existing = config.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                existing.update(value)
            else:
                config[key] = value
    return config

CONFIG = {}  # set in main

def agent_settings(role: str) -> dict:
    return CONFIG.get(role, {})

# ── LLM ────────────────────────────────────────────────────────

from llm import llm as _llm_call

WATCH = False  # set in main

async def llm(
    prompt: str,
    *,
    role: str = "solver",
    schema: dict | None = None,
    system: str | None = None,
    resume: str | None = None,
) -> tuple[str | dict, str | None]:
    return await _llm_call(
        prompt, role=role, schema=schema, system=system,
        config=CONFIG, watch=WATCH, resume=resume,
    )

# ── Judges ──────────────────────────────────────────────────────

def _format_proof(proof: Proof) -> str:
    lines = [f"State 0: {proof.initial_state}"]
    for i, step in enumerate(proof.steps):
        lines.append(f"State {i+1}: {step.state}  [{step.justification_type}] {step.justification}")
    return "\n".join(lines)


async def judge(
    step: Step, prev_state: list[str], problem: str, proof: Proof, step_number: int,
    on_complete=None,
) -> tuple[Verdict, dict]:
    role = step.justification_type  # "citation", "problem_given", or "computation"
    prompt_template = agent_prompt(role)

    # Variables available to all judge prompts
    variables = {
        "step_number": str(step_number),
        "prev_state": str(prev_state),
        "new_state": str(step.state),
        "justification": step.justification,
        "justification_type": step.justification_type,
        "problem": problem,
        "proof": _format_proof(proof),
    }

    # Format the prompt template with variables. If a template references an
    # undefined variable we crash loudly — silently falling back to the raw
    # template would send the judge a broken prompt with literal `{step_number}`
    # placeholders and produce garbage verdicts. Fail fast on this class of bug.
    try:
        system = prompt_template.format(**variables)
    except KeyError as e:
        raise RuntimeError(
            f"Prompt template for role {role!r} references undefined variable {e}. "
            f"Available variables: {sorted(variables.keys())}. "
            f"Check configs/defaults.yaml for a stray {{...}} placeholder."
        )

    # User message is always the full context
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Full proof:\n{_format_proof(proof)}\n\n"
        f"Step {step_number} being judged:\n"
        f"  Previous state: {prev_state}\n"
        f"  New state: {step.state}\n"
        f"  Justification type: {step.justification_type}\n"
        f"  Justification: {step.justification}"
    )

    data, _ = await llm(user_msg, role=role, schema=VERDICT_SCHEMA, system=system)
    verdict = Verdict(**data)
    inputs = {"role": role, "user_msg": user_msg, "system": system}
    if on_complete is not None:
        await on_complete(step_number, verdict, inputs)
    return verdict, inputs


async def judge_initial_state(
    proof: Proof, problem: str, on_complete=None,
) -> tuple[Verdict, dict]:
    """Audit state 0 — the proof's initial state, which never gets a
    per-step justification. Catches two failure modes: (1) the
    formalizer drifting off the goal-in-state[0] convention and
    filling state 0 with definitions instead of a goal (p115 pattern),
    and (2) content smuggled into state 0 that isn't supported by the
    problem text and would otherwise flow through the proof unchecked.

    Runs in parallel with the per-step judges from `_judge_proof`.
    """
    prompt_template = agent_prompt("initial_state")
    variables = {
        "initial_state": str(proof.initial_state),
        "problem": problem,
    }
    try:
        system = prompt_template.format(**variables)
    except KeyError as e:
        raise RuntimeError(
            f"Prompt template for role 'initial_state' references undefined "
            f"variable {e}. Available: {sorted(variables.keys())}."
        )
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Initial state (state 0): {proof.initial_state}\n\n"
        f"Full proof for context:\n{_format_proof(proof)}"
    )
    data, _ = await llm(
        user_msg, role="initial_state", schema=VERDICT_SCHEMA, system=system,
    )
    verdict = Verdict(**data)
    inputs = {"role": "initial_state", "user_msg": user_msg, "system": system}
    if on_complete is not None:
        await on_complete(0, verdict, inputs)
    return verdict, inputs


# ── Pipeline ────────────────────────────────────────────────────

def _proof_from_dict(data: dict) -> Proof:
    return Proof(
        initial_state=data["initial_state"],
        steps=[Step(**s) for s in data["steps"]],
    )


def _proof_to_dict(proof: Proof) -> dict:
    """Serialize a Proof back to JSON shape."""
    return {
        "initial_state": proof.initial_state,
        "steps": [
            {
                "state": s.state,
                "justification_type": s.justification_type,
                "justification": s.justification,
            }
            for s in proof.steps
        ],
    }


def _format_failed_verdicts(
    proof: Proof,
    verdicts: list[Verdict],
    state0_verdict: Verdict | None = None,
) -> str:
    lines = []
    if state0_verdict is not None and not state0_verdict.accepted:
        lines.append(
            f"State 0 (initial_state) FAILED: {proof.initial_state}\n"
            f"  Reason: {state0_verdict.reason}"
        )
    for i, (step, v) in enumerate(zip(proof.steps, verdicts)):
        if not v.accepted:
            lines.append(
                f"Step {i+1} [{step.justification_type}] FAILED: {step.justification}\n"
                f"  Reason: {v.reason}"
            )
    return "\n\n".join(lines)


async def _judge_proof(
    proof: Proof, problem: str, on_each=None,
) -> tuple[list[Verdict], list[dict]]:
    prev_states = [proof.initial_state] + [s.state for s in proof.steps[:-1]]
    judge_results = await asyncio.gather(*[
        judge(step, prev, problem, proof, i + 1, on_complete=on_each)
        for i, (step, prev) in enumerate(zip(proof.steps, prev_states))
    ])
    verdicts = [jr[0] for jr in judge_results]
    inputs = [jr[1] for jr in judge_results]
    return verdicts, inputs


async def _pedantry_check(
    verdict, step, prev_state, step_number, problem, proof,
    original_reason=None, on_complete=None,
):
    """Ask the pedantry filter whether this rejection is legitimate or pedantic."""
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Full proof:\n{_format_proof(proof)}\n\n"
        f"Step {step_number} being evaluated:\n"
        f"  Previous state: {prev_state}\n"
        f"  New state: {step.state}\n"
        f"  Justification type: {step.justification_type}\n"
        f"  Justification: {step.justification}\n\n"
        f"A judge rejected this step with the following reason:\n{verdict.reason}\n\n"
        f"Is this rejection legitimate (the proof is actually wrong) or "
        f"pedantic (the proof is correct, the judge is being too strict)?"
    )
    data, _ = await llm(
        user_msg, role="pedantry", schema=PEDANTRY_SCHEMA,
        system=agent_prompt("pedantry"),
    )
    is_pedantic = data["is_pedantic"]
    reason = data["reason"]
    if on_complete is not None:
        await on_complete(step_number, original_reason, is_pedantic, reason)
    return is_pedantic, reason


async def _convention_lift_step(
    verdict, step, prev_state, step_number, problem, proof,
    pedantry_reason, on_complete=None,
):
    """Ask the convention_lift judge whether a legitimately-rejected
    step can be accepted by invoking a standard, citable, domain-level
    convention. If yes, the step becomes accepted-under-assumption.

    Runs only on per-step verdicts that pedantry already confirmed are
    legitimate. Same parallel-gather shape as _pedantry_check.
    """
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Full proof:\n{_format_proof(proof)}\n\n"
        f"Step {step_number} being evaluated:\n"
        f"  Previous state: {prev_state}\n"
        f"  New state: {step.state}\n"
        f"  Justification type: {step.justification_type}\n"
        f"  Justification: {step.justification}\n\n"
        f"Judge's rejection reason:\n{verdict.reason}\n\n"
        f"Pedantry already confirmed this rejection is legitimate:\n"
        f"{pedantry_reason}\n\n"
        f"Is there a single standard, citable, domain-wide convention "
        f"that — if added as an explicit premise — would fully justify "
        f"this step and resolve the judge's objection?"
    )
    data, _ = await llm(
        user_msg, role="convention_lift", schema=CONVENTION_LIFT_SCHEMA,
        system=agent_prompt("convention_lift"),
    )
    if on_complete is not None:
        await on_complete(step_number, data)
    return data


async def _convention_lift_state0(
    verdict, initial_state, problem, pedantry_reason, on_complete=None,
):
    """Convention_lift applied to state 0 (the initial state)."""
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Initial state (state 0): {initial_state}\n\n"
        f"The initial_state judge rejected this state with the following "
        f"reason:\n{verdict.reason}\n\n"
        f"Pedantry already confirmed this rejection is legitimate:\n"
        f"{pedantry_reason}\n\n"
        f"Is there a single standard, citable, domain-wide convention "
        f"that — if added as an explicit premise — would fully justify "
        f"this initial state and resolve the judge's objection?"
    )
    data, _ = await llm(
        user_msg, role="convention_lift", schema=CONVENTION_LIFT_SCHEMA,
        system=agent_prompt("convention_lift"),
    )
    if on_complete is not None:
        await on_complete(0, data)
    return data


async def _pedantry_check_state0(
    verdict, initial_state, problem, on_complete=None,
):
    """Pedantry filter applied to a failed state 0 verdict. Same
    mechanism as _pedantry_check but adapted for state 0 (which has
    no Step/prev_state structure)."""
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Initial state (state 0) being evaluated: {initial_state}\n\n"
        f"The initial_state judge rejected this state with the following "
        f"reason:\n{verdict.reason}\n\n"
        f"Is this rejection legitimate (state 0 actually contains smuggled "
        f"or made-up content not supported by the problem text) or pedantic "
        f"(state 0 is fine and the judge is being too strict)?"
    )
    data, _ = await llm(
        user_msg, role="pedantry", schema=PEDANTRY_SCHEMA,
        system=agent_prompt("pedantry"),
    )
    is_pedantic = data["is_pedantic"]
    reason = data["reason"]
    if on_complete is not None:
        await on_complete(0, verdict.reason, is_pedantic, reason)
    return is_pedantic, reason


async def _filter_pedantic(
    proof: Proof, verdicts: list[Verdict], problem: str,
    state0_verdict: Verdict | None = None,
    on_each=None, on_state0=None,
) -> tuple[list[Verdict], Verdict | None, list[dict]]:
    """Run pedantry checks on every failed verdict in parallel.

    Returns (updated_verdicts, updated_state0_verdict, pedantry_records).
    Updated verdicts have accepted=True (with a [PEDANTRY OVERRIDE] tag)
    for any verdict marked pedantic. Pedantry_records is a list of
    {step_number, original_reason, is_pedantic, pedantry_reason}; state
    0's record (if any) uses step_number=0.
    """
    failed_indices = [i for i, v in enumerate(verdicts) if not v.accepted]
    state0_failed = (
        state0_verdict is not None and not state0_verdict.accepted
    )
    if not failed_indices and not state0_failed:
        return list(verdicts), state0_verdict, []

    prev_states = [proof.initial_state] + [s.state for s in proof.steps[:-1]]

    tasks = []
    if state0_failed:
        tasks.append(_pedantry_check_state0(
            state0_verdict, proof.initial_state, problem,
            on_complete=on_state0,
        ))
    tasks.extend(
        _pedantry_check(
            verdicts[i],
            proof.steps[i],
            prev_states[i],
            i + 1,
            problem,
            proof,
            original_reason=verdicts[i].reason,
            on_complete=on_each,
        )
        for i in failed_indices
    )
    pedantry_results = await asyncio.gather(*tasks)

    # Unpack: state 0 first (if it ran), then step verdicts.
    s0_result: tuple | None = None
    if state0_failed:
        s0_result = pedantry_results[0]
        pedantry_results = pedantry_results[1:]

    updated = list(verdicts)
    records = []
    for idx, (is_pedantic, reason) in zip(failed_indices, pedantry_results):
        original_reason = verdicts[idx].reason
        records.append({
            "step_number": idx + 1,
            "original_reason": original_reason,
            "is_pedantic": is_pedantic,
            "pedantry_reason": reason,
        })
        if is_pedantic:
            updated[idx] = Verdict(
                accepted=True,
                reason=f"[PEDANTRY OVERRIDE] {reason}",
            )

    # Apply state 0 pedantry override too, if applicable.
    updated_state0 = state0_verdict
    if s0_result is not None:
        is_pedantic, reason = s0_result
        records.append({
            "step_number": 0,
            "original_reason": state0_verdict.reason,
            "is_pedantic": is_pedantic,
            "pedantry_reason": reason,
        })
        if is_pedantic:
            updated_state0 = Verdict(
                accepted=True,
                reason=f"[PEDANTRY OVERRIDE] {reason}",
            )

    return updated, updated_state0, records


def _print_proof(proof: Proof):
    print(f"    {len(proof.steps)} steps")
    print(f"    State 0: {proof.initial_state}")
    for i, step in enumerate(proof.steps):
        print(f"    State {i+1}: {step.state}  [{step.justification_type}]")


def _print_verdicts(
    proof: Proof, verdicts: list[Verdict],
    state0_verdict: Verdict | None = None,
):
    if state0_verdict is not None:
        icon = "PASS" if state0_verdict.accepted else "FAIL"
        print(f"    0. [{icon}] initial_state: {proof.initial_state}")
        if not state0_verdict.accepted:
            print(f"       {state0_verdict.reason}")
    for i, (step, v) in enumerate(zip(proof.steps, verdicts)):
        icon = "PASS" if v.accepted else "FAIL"
        print(f"    {i+1}. [{icon}] {step.justification_type}: {step.justification}")
        if not v.accepted:
            print(f"       {v.reason}")


# Per-problem cycle bounds. Defaults shown below; override via a
# `_limits:` block in a stacked --config YAML.
_DEFAULT_MAX_VERIFY_ATTEMPTS = 3   # formalize+judge cycles per problem
_DEFAULT_MAX_SOLVER_ANSWERS = 3    # distinct solver answers per problem


def _limits() -> dict:
    return CONFIG.get("_limits") or {}


def max_verify_attempts() -> int:
    return int(_limits().get("max_verify_attempts", _DEFAULT_MAX_VERIFY_ATTEMPTS))


def max_solver_answers() -> int:
    return int(_limits().get("max_solver_answers", _DEFAULT_MAX_SOLVER_ANSWERS))


async def _solver_call(problem, solver_session, retry_reason=None):
    """First solve or retry. Returns (solution, session_id)."""
    if solver_session is None:
        return await llm(problem, role="solver", system=agent_prompt("solver"))
    msg = (
        "Your previous answer was rejected. Reason:\n\n"
        f"{retry_reason}\n\n"
        "Please provide a new answer."
    )
    return await llm(msg, role="solver", resume=solver_session)


async def _formalizer_call(
    problem,
    solution,
    formalizer_session,
    failed_verdicts_text=None,
    prior_proof=None,
):
    """Ask the formalizer for a decision (proof or reject).

    Two modes:
    - claude backend: uses session resume. Each call only sends new info;
      the formalizer remembers prior proofs and reasoning from the session.
    - codex backend: stateless. Each call passes the full context (prior
      proof + failed verdicts) because codex exec resume doesn't support
      --output-schema, so we can't get structured output on resumed sessions.

    Returns (decision_dict, session_id).
    """
    backend = agent_settings("formalizer").get("backend", "claude")

    if backend == "claude":
        # Session-resume mode
        if formalizer_session is None:
            user_msg = (
                f"Problem: {problem}\n\nSolution: {solution}\n\n"
                "Formalize this into a proof, or reject if it has errors."
            )
            return await llm(
                user_msg, role="formalizer", schema=_formalizer_decision_schema(),
                system=agent_prompt("formalizer"),
            )
        elif failed_verdicts_text:
            user_msg = (
                "Your proof failed verification.\n\n"
                f"Failed steps:\n{failed_verdicts_text}\n\n"
                "Either produce a corrected proof (action='proof') or reject "
                "the underlying solution (action='reject') with a reason for "
                "the solver."
            )
            return await llm(
                user_msg, role="formalizer", schema=_formalizer_decision_schema(),
                resume=formalizer_session,
            )
        else:
            user_msg = (
                f"The solver provided a new answer:\n\n{solution}\n\n"
                "Formalize this into a proof, or reject again if it still has errors."
            )
            return await llm(
                user_msg, role="formalizer", schema=_formalizer_decision_schema(),
                resume=formalizer_session,
            )

    # Stateless mode (codex or any backend that doesn't support resume+schema)
    parts = [f"Problem: {problem}", f"Solution: {solution}"]
    if prior_proof is not None:
        parts.append(
            "Your previous proof attempt:\n" + _format_proof(prior_proof)
        )
    if failed_verdicts_text:
        parts.append(
            "Failed verification steps:\n" + failed_verdicts_text
        )
        parts.append(
            "Either produce a corrected proof (action='proof') or reject "
            "the underlying solution (action='reject') with a reason for "
            "the solver."
        )
    else:
        parts.append("Formalize this into a proof, or reject if it has errors.")
    user_msg = "\n\n".join(parts)
    return await llm(
        user_msg, role="formalizer", schema=_formalizer_decision_schema(),
        system=agent_prompt("formalizer"),
    )


async def run(problem: str, *, pid: str | None = None, partial_save_path: str | None = None):
    """Run the verified-reasoning pipeline on one problem.

    Optional `pid` is the problem id; when set, every print is prefixed
    with `[pid]` so concurrent runs are distinguishable in shared logs.

    Optional `partial_save_path` enables crash-safe per-step persistence.
    The current running state is atomically written to that path after
    every meaningful boundary (solver returns, formalizer returns, each
    individual judge call completes, each individual pedantry call
    completes, and after each attempt is appended). On crash mid-attempt,
    the partial file contains everything that completed before the crash.
    """
    attempts = []  # one entry per formalize+judge cycle

    # ── per-problem logging helper ───────────────────────────────────
    def log(msg: str) -> None:
        prefix = f"[{pid}] " if pid else ""
        print(f"{prefix}{msg}")

    # ── per-step partial save helpers ────────────────────────────────
    state = {
        "pid": pid,
        "problem": problem,
        "solution": None,
        "attempts": attempts,
        "in_progress_attempt": None,
        "last_event": None,
    }
    save_lock = asyncio.Lock()

    async def save_partial(event: str):
        state["last_event"] = event
        if not partial_save_path:
            return
        async with save_lock:
            tmp = partial_save_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp, partial_save_path)

    # ── initial solve ────────────────────────────────────────────────
    log("\n[solve] Solving...")
    solution, solver_session = await _solver_call(problem, None)
    log(f"    {solution[:200]}")
    state["solution"] = solution
    await save_partial("solver_returned")

    formalizer_session = None
    failed_verdicts_text = None  # set after a verification failure
    last_proof = None             # latest proof — used by stateless backends
    verify_attempts = 0
    solver_answers = 1  # initial solve counts as the first answer

    max_verify = max_verify_attempts()
    max_solver = max_solver_answers()

    while True:
        # Ask formalizer for a decision
        log(
            f"\n[formalize] verify={verify_attempts}/{max_verify} "
            f"solver_answers={solver_answers}/{max_solver}..."
        )
        decision, new_session = await _formalizer_call(
            problem, solution, formalizer_session, failed_verdicts_text,
            prior_proof=last_proof,
        )
        if formalizer_session is None:
            formalizer_session = new_session
        action = decision.get("action")
        log(f"    Formalizer chose: {action}")
        await save_partial(f"formalizer_returned:{action}")

        if action == "reject":
            reason = decision.get("reject_reason", "")
            log(f"    Rejected: {reason[:200]}")
            attempts.append({
                "attempt": len(attempts) + 1,
                "phase": "formalizer_reject",
                "reject_reason": reason,
            })
            await save_partial("attempt_appended:formalizer_reject")
            if solver_answers >= max_solver:
                log(f"\n[!] Max solver answers ({max_solver}) reached, giving up")
                break
            # Solver retry
            log(f"\n[solve] Retrying with formalizer's reason...")
            solution, _ = await _solver_call(problem, solver_session, retry_reason=reason)
            log(f"    {solution[:200]}")
            solver_answers += 1
            failed_verdicts_text = None  # reset, this is a fresh formalization
            last_proof = None             # reset, no prior proof for new answer
            state["solution"] = solution
            await save_partial("solver_returned")
            continue

        # action == "proof": judge it
        verify_attempts += 1
        proof = _proof_from_dict(decision["proof"])
        last_proof = proof
        _print_proof(proof)

        # Set up the in-progress attempt with verdict placeholders that
        # get filled in as individual judges complete. state0_verdict
        # is a separate slot — state 0 is audited in parallel with the
        # per-step judges but isn't part of the step-indexed verdicts
        # list (it has no Step associated with it).
        in_progress = {
            "attempt": len(attempts) + 1,
            "phase": "verify",
            "proof": _proof_to_dict(proof),
            "state0_verdict": None,
            "verdicts": [None] * len(proof.steps),
            "pedantry": [],
            "conventions": [],
            "all_ok": None,
        }
        state["in_progress_attempt"] = in_progress
        await save_partial("attempt_started")

        log(f"\n[judge] Verifying...")

        async def on_judge_complete(step_number, verdict, inputs):
            in_progress["verdicts"][step_number - 1] = {
                "step_number": step_number,
                "role": inputs["role"],
                "user_msg": inputs["user_msg"],
                "system": inputs["system"],
                "accepted": verdict.accepted,
                "reason": verdict.reason,
            }
            await save_partial(f"judge_completed:{step_number}")

        async def on_state0_complete(_step_number, verdict, inputs):
            in_progress["state0_verdict"] = {
                "step_number": 0,
                "role": inputs["role"],
                "user_msg": inputs["user_msg"],
                "system": inputs["system"],
                "accepted": verdict.accepted,
                "reason": verdict.reason,
            }
            await save_partial("state0_judge_completed")

        # Run state 0 judge and the per-step judges concurrently.
        # Wasted work on a state 0 failure is trivial; keeping them in
        # one gather means the repair loop gets richer feedback.
        state0_task = judge_initial_state(
            proof, problem, on_complete=on_state0_complete,
        )
        step_judges_task = _judge_proof(
            proof, problem, on_each=on_judge_complete,
        )
        (state0_verdict, state0_input), (verdicts, judge_inputs) = \
            await asyncio.gather(state0_task, step_judges_task)
        _print_verdicts(proof, verdicts, state0_verdict=state0_verdict)

        # Pedantry filter — for each failed verdict (including state 0),
        # decide if the rejection is legitimate or just nitpicking.
        # Pedantic ones get marked accepted with a [PEDANTRY OVERRIDE] tag.
        n_failed_steps = sum(1 for v in verdicts if not v.accepted)
        state0_failed = not state0_verdict.accepted
        n_failed_before = n_failed_steps + (1 if state0_failed else 0)
        if n_failed_before > 0:
            log(f"\n[pedantry] Filtering {n_failed_before} failures"
                f"{' (incl. state 0)' if state0_failed else ''}...")

            async def on_pedantry_complete(step_number, original_reason, is_pedantic, reason):
                in_progress["pedantry"].append({
                    "step_number": step_number,
                    "original_reason": original_reason,
                    "is_pedantic": is_pedantic,
                    "pedantry_reason": reason,
                })
                await save_partial(f"pedantry_completed:{step_number}")

            verdicts, state0_verdict, pedantry_records = await _filter_pedantic(
                proof, verdicts, problem,
                state0_verdict=state0_verdict,
                on_each=on_pedantry_complete,
                on_state0=on_pedantry_complete,
            )
            n_pedantic = sum(1 for r in pedantry_records if r["is_pedantic"])
            n_legit = n_failed_before - n_pedantic
            log(f"    {n_pedantic} marked pedantic, {n_legit} legitimate")
        else:
            pedantry_records = []

        # Convention-lift pass: for each verdict that survived pedantry
        # as a legitimate rejection, ask whether a standard domain
        # convention would fully justify it. If so, override to
        # accepted-under-assumption and record the convention.
        conventions_added: list[dict] = []
        legit_rejections = [
            (r, v, s) for r, v, s in (
                [
                    (r, state0_verdict, "state0") for r in pedantry_records
                    if r["step_number"] == 0 and not r["is_pedantic"]
                ] + [
                    (r, verdicts[r["step_number"] - 1], "step") for r in pedantry_records
                    if r["step_number"] >= 1 and not r["is_pedantic"]
                ]
            )
            if not v.accepted
        ]
        if legit_rejections:
            log(f"\n[convention_lift] Evaluating {len(legit_rejections)} "
                f"legitimate rejections for standard-convention lift...")

            async def on_convention_complete(step_number, data):
                conventions_added.append({
                    "step_number": step_number,
                    "can_lift": data.get("can_lift"),
                    "convention": data.get("convention"),
                    "source": data.get("source"),
                    "reasoning": data.get("reasoning"),
                })
                await save_partial(f"convention_lift_completed:{step_number}")

            prev_states = [proof.initial_state] + [s.state for s in proof.steps[:-1]]
            tasks = []
            for pr, v, kind in legit_rejections:
                sn = pr["step_number"]
                ped_reason = pr["pedantry_reason"]
                if kind == "state0":
                    tasks.append(_convention_lift_state0(
                        v, proof.initial_state, problem, ped_reason,
                        on_complete=on_convention_complete,
                    ))
                else:
                    tasks.append(_convention_lift_step(
                        v, proof.steps[sn - 1], prev_states[sn - 1],
                        sn, problem, proof, ped_reason,
                        on_complete=on_convention_complete,
                    ))
            results = await asyncio.gather(*tasks)

            # Apply accepted-under-assumption overrides.
            n_lifted = 0
            for (pr, v, kind), data in zip(legit_rejections, results):
                if not data.get("can_lift"):
                    continue
                n_lifted += 1
                tag = (
                    f"[ASSUMPTION: {data.get('convention','')}] "
                    f"(source: {data.get('source','')})"
                )
                new_verdict = Verdict(accepted=True, reason=tag)
                if kind == "state0":
                    state0_verdict = new_verdict
                else:
                    verdicts[pr["step_number"] - 1] = new_verdict
            n_unlifted = len(legit_rejections) - n_lifted
            log(f"    {n_lifted} lifted under convention, {n_unlifted} not")

        all_ok = state0_verdict.accepted and all(v.accepted for v in verdicts)
        # Replace any pedantry-overridden verdicts in the in-progress record.
        in_progress["state0_verdict"] = {
            "step_number": 0,
            "role": state0_input["role"],
            "user_msg": state0_input["user_msg"],
            "system": state0_input["system"],
            "accepted": state0_verdict.accepted,
            "reason": state0_verdict.reason,
        }
        for i, v in enumerate(verdicts):
            in_progress["verdicts"][i] = {
                "step_number": i + 1,
                "role": judge_inputs[i]["role"],
                "user_msg": judge_inputs[i]["user_msg"],
                "system": judge_inputs[i]["system"],
                "accepted": v.accepted,
                "reason": v.reason,
            }
        in_progress["pedantry"] = pedantry_records
        in_progress["conventions"] = conventions_added
        in_progress["all_ok"] = all_ok
        attempts.append(in_progress)
        state["in_progress_attempt"] = None
        await save_partial("attempt_appended:verify")

        if all_ok:
            break

        if verify_attempts >= max_verify:
            log(f"\n[!] Max verify attempts ({max_verify}) reached, giving up")
            break

        # Set up for next iteration: formalizer will get the failed verdicts
        # including a state 0 failure if one occurred.
        failed_verdicts_text = _format_failed_verdicts(
            proof, verdicts, state0_verdict=state0_verdict,
        )

    # Build final result from the last attempt that produced a proof
    last_proof_attempt = next(
        (a for a in reversed(attempts) if a.get("phase") == "verify"),
        None,
    )
    if last_proof_attempt:
        verified = last_proof_attempt["all_ok"]
        proof_dict = last_proof_attempt["proof"]
        verdicts_dict = last_proof_attempt["verdicts"]
        answer = (
            proof_dict["steps"][-1]["state"][0]
            if proof_dict["steps"] else None
        )
        # Collect any conventions the last (winning) attempt invoked.
        # A proof is verified_unconditionally iff it's verified AND no
        # assumptions were added.
        verified_under_assumptions = [
            {
                "step_number": c["step_number"],
                "convention": c["convention"],
                "source": c["source"],
            }
            for c in (last_proof_attempt.get("conventions") or [])
            if c.get("can_lift")
        ]
    else:
        # All attempts ended in rejection — no proof was ever produced
        verified = False
        proof_dict = None
        verdicts_dict = []
        answer = None
        verified_under_assumptions = []

    verified_unconditionally = bool(verified and not verified_under_assumptions)

    print(f"\n{'='*40}")
    if verified and verified_under_assumptions:
        print(
            f"JUDGE-PASSED (under {len(verified_under_assumptions)} "
            f"assumption{'s' if len(verified_under_assumptions)!=1 else ''}) "
            f"— answer: {answer}"
        )
    else:
        print(f"{'JUDGE-PASSED' if verified else 'REJECTED'} — answer: {answer}")

    return {
        "problem": problem,
        "answer": answer,
        "verified": verified,
        "verified_unconditionally": verified_unconditionally,
        "verified_under_assumptions": verified_under_assumptions,
        "solution": solution,
        "proof": proof_dict,
        "verdicts": verdicts_dict,
        "attempts": attempts,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Theoria — verified reasoning")
    parser.add_argument("problem", nargs="?", default="What is 2 + 2?")
    # Debug entry point only — the real CLI is `theoria` (cli.py). Backend
    # selection lives there; this just runs one problem against the current
    # config so you can poke at the pipeline directly.
    parser.add_argument("--config", action="append", default=None, help="Path to config YAML. Repeat to stack multiple.")
    parser.add_argument("--watch", action="store_true", help="Stream LLM events to stderr in real time")
    args = parser.parse_args()

    CONFIG.update(load_config(args.config))
    WATCH = args.watch
    asyncio.run(run(args.problem))
