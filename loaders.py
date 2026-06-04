"""Problem loaders — turn an HLE problem or an ad-hoc question into the
`problem` dicts the pipeline runs on.

Every loader returns a list of dicts. Each dict must have:
    id        — unique string
    question  — the problem text the solver sees
    answer    — expected answer (used only by the naive `correct` check;
                empty string for open-ended questions)
plus any extra fields you want carried through to the saved result.

This module is pure data-wrangling: no LLM calls, no Docker, no argument
parsing. The CLI (cli.py) wires these into the harness.
"""

from __future__ import annotations

import json


# ── Humanity's Last Exam ──────────────────────────────────────────

def load_hle(
    category: str | None = None,
    max_questions: int | None = None,
    skip: int = 0,
    skip_images: bool = True,
    ids: list[str] | None = None,
    subset: str = "Gold subset",
) -> list[dict]:
    """Load HLE-Verified (https://arxiv.org/abs/2602.13964).

    By default, only the Gold subset (668 expert-verified clean items) is
    returned. Pass subset=None to load all 2,500 items, or one of:
      "Gold subset"      — 668 fully-verified items
      "Revision subset"  — 1,143 expert-repaired items
      "Uncertain subset" — 689 items with documented uncertainty

    When `ids` is given, the subset filter is ignored — you always get the
    requested items regardless of which subset they belong to.

    Images are skipped by default (this is a text-only pipeline).
    """
    from datasets import load_dataset  # imported lazily so other CLI
                                       # commands don't need `datasets`

    ds = load_dataset("skylenage/HLE-Verified", split="train")
    problems = []
    id_set = set(ids) if ids else None
    skipped = 0

    for ex in ds:
        # Image flag and answer_type live inside the nested `json` blob.
        meta = json.loads(ex["json"])

        if id_set is not None:
            if ex["id"] not in id_set:
                continue
        else:
            if subset and ex["Verified_Classes"] != subset:
                continue

        if skip_images and meta.get("image"):
            continue
        if category and ex["category"] != category:
            continue
        if id_set is None:
            skipped += 1
            if skipped <= skip:
                continue

        problems.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "answer_type": meta.get("answer_type", ""),
            "category": ex["category"],
            "verified_class": ex["Verified_Classes"],
        })

        if max_questions and len(problems) >= max_questions:
            break

    return problems


# ── Arbitrary questions ───────────────────────────────────────────

def build_question(question: str, problem_id: str = "question") -> list[dict]:
    """A single ad-hoc question. No expected answer."""
    return [{
        "id": problem_id,
        "question": question,
        "answer": "",
    }]
