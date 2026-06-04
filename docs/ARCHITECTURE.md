# Architecture

Theoria is small. There are four engine files, three thin layers on top, and a
folder of prompts. If you read `pipeline.py` top to bottom you've understood
the system.

## The files

```
pipeline.py    THE METHOD. The solve → formalize → judge → filter → repair
               loop, plus the proof/verdict data types. Read this first.
llm.py         Backend plumbing. Calls the `claude` or `codex` CLI (optionally
               inside Docker), enforces JSON-schema output, streams events,
               writes per-call artifacts, and survives hangs.
sandbox.py     One hardened Docker container per problem; mounts subscription
               credentials; tears it down afterward.
harness.py     Runs many problems with bounded parallelism, saves incrementally,
               and records reproducibility metadata for each run.

loaders.py     Builds the `problem` dicts: HLE problems or an ad-hoc question.
grade.py       LLM-judge grader (answer vs. expected) — a stronger check than
               the naive `correct` flag, not ground truth. Reads a run JSON.
export_problem.py   Renders a run JSON into readable markdown.
cli.py         The `theoria` command that ties it all together.

configs/       Every role's prompt and model, as data. defaults.yaml is the
               spec of what each judge actually does.
sandbox/, sandbox-sage/   The two Docker images (standard, and Sage for math).
```

## The verification loop

```
                         ┌─────────────────────────────────────────┐
                         │                problem                   │
                         └─────────────────────────────────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │    SOLVE     │  solver answers
                                   └──────────────┘  (web search ok)
                                          │
                                          ▼
                                   ┌──────────────┐   reject
                                   │  FORMALIZE   │──────────────┐
                                   │  → Proof     │              │ (back to
                                   └──────────────┘              │  solver for
                                          │ proof                │  a new answer)
                                          ▼                      │
              ┌───────────────────────────────────────────┐     │
              │   JUDGE  (all steps + state 0, in parallel)│     │
              │   computation · citation · problem_given   │     │
              └───────────────────────────────────────────┘     │
                                          │ any rejections?      │
                            ┌─────────────┴───────────┐          │
                          no│                          │yes       │
                            ▼                          ▼          │
                     ┌────────────┐            ┌──────────────┐   │
                     │JUDGE-PASSED│            │  PEDANTRY    │   │
                     │     ✓      │            │  filter      │   │ over the
                     └────────────┘            └──────────────┘   │ attempt
                                                      │ still bad │ limit?
                                                      ▼           │
                                               ┌──────────────┐   │
                                               │ CONVENTION   │   │
                                               │ lift         │   │
                                               └──────────────┘   │
                                                      │ still bad │
                                                      └───────────┘
                                                       repair loop
```

- **Three justification types**, each with its own judge prompt:
  `computation` (an operation that was actually performed),
  `citation` (a theorem / identity / definition that licenses the step),
  `problem_given` (a fact taken directly from the problem text).
- **State 0** (the proof's initial state) is audited by its own judge in
  parallel with the steps — it catches content smuggled in as a premise.
- **Pedantry filter**: brutal judges over-reject informally-worded problems.
  This pass asks, per rejection, "real error or nitpick?" and overrides
  nitpicks (tagged `[PEDANTRY OVERRIDE]`).
- **Convention lift**: a rejection that's a *real* gap may still be closed by
  one standard, citable convention (e.g. "assume an inertial frame"). If so
  the step is accepted *under that assumption*, which is recorded — such a
  result is `verified` but not `verified_unconditionally`.
- **Repair loop**: failed verdicts go back to the formalizer, bounded by
  `max_verify_attempts` and `max_solver_answers` (defaults 3 and 3).

**A note on terminology.** A pass means *every step's justification was
accepted by an independent LLM judge* — it is **not** formal/axiomatic
verification. So the outward-facing verdict is reported as **JUDGE-PASSED /
REJECTED**, not "verified." The internal result field is still named
`verified` (a boolean), and at the system level the decline behavior is
*abstention*: Theoria withholds an answer rather than ship one it can't justify.

**What it's for.** Theoria verifies questions that have a **definite,
checkable answer** — a value, expression, multiple-choice letter, or other
conclusion a proof can land on. Open-ended or subjective prompts ("what do you
think of…", "why is the sky blue?") don't fit the proof model; the formalizer
is forced into an awkward conclusion and such questions are typically REJECTED.
That's the system working as intended, not a failure.

## Backends

Every role runs on **Codex** except the **formalizer**, which always runs on
**Claude** — it needs session-resume plus structured output together, which
only the Claude path supports (`llm.py` dispatches per role from the config).
This split — Codex solver and judges, Claude formalizer — is the single
audited configuration, and it's the default with no
flags (so a Claude *and* a Codex subscription are both required). Re-wiring
roles to other backends is only possible by writing a custom `--config` YAML,
and is unaudited.

## What a run leaves behind

```
runs/<command>_<tag>_<timestamp>.json     the results (one entry per problem)
runs/partial/<id>.json                    crash-safe live snapshot
runs/artifacts/<run_id>/
    meta.json                             argv, config, model + CLI versions,
                                          git sha, sandbox image digest
    <problem_id>/call_NNN_<role>/         every prompt, response, tool call,
                                          and raw event stream, per LLM call
```

Everything needed to reproduce or audit a run is on disk.
