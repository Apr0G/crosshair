"""
Regression tests from Faig's review loop (P11.2.3). Skeleton: no review has happened
yet, so it holds one placeholder check.

How it grows: every "disagree" in `data/engine/eval/review_log.md` becomes a numbered
item (`python -m engine.evaluation.review --parse <log>`). Each item is traced to a
stage (parse, state, action detection, model, attribution, value) and either fixed or
accepted with a written reason. A fix adds ONE function here, `item_<n>()`, that
rebuilds the case from stored states / actions (copied into `test/engine/fixtures/`
so the test does not need `data/`), runs the stage that was wrong, and asserts the
corrected behaviour; the docstring names the review cycle, round and action id. Call
it from the list at the bottom. Accepted items get no function; their reason goes in
the review log.

    .venv\\Scripts\\python.exe test\\engine\\test_review_items.py
"""
from _harness import check, section, summary
from engine.evaluation import review


def placeholder() -> None:
    """Stands in until the first review cycle produces an item."""
    check("review-item parser is importable (no items yet)", callable(review.parse_review_log))


# ── items (one function per review item, P11.2.3) ─────────────────────────────

ITEMS = [placeholder]

section("review items (P11.2.3)")
for fn in ITEMS:
    fn()

raise SystemExit(summary())
