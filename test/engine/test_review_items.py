"""
Regression tests from Faig's review loop (P11.2.3). No review has happened yet, so it
holds one placeholder: a tiny fixture log parsed into the item a real one would give.

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
import tempfile
from pathlib import Path

from _harness import check, section, summary
from engine.evaluation import review

FIXTURE_LOG = """# review log (fixture)

<!-- case index=1 match_key=syn_trade round_num=3 -->
| rank | action | action_id | V | D | X | panel |
|---|---|---|---|---|---|---|
| 1 | kill | a000001 | agree | disagree | agree | unsure |
| 2 | flash | a000002 | agree | agree | agree | agree |
Comment: D looks too high for a free trade
"""


def placeholder() -> None:
    """Stands in until the first review cycle produces an item: a fixture log with one
    "disagree" parses into exactly that item (LOW-57)."""
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "review_log.md"
        log.write_text(FIXTURE_LOG, encoding="utf-8")
        items = review.parse_review_log(log)
    want = [{"item": 1, "round_index": 1, "match_key": "syn_trade", "round_num": 3, "rank": 1,
             "action_id": "a000001", "aspect": "D", "comment": "D looks too high for a free trade", "line": 6}]
    check("fixture review log -> one item (D on a000001, round 3, with the comment)", items == want, str(items))


# ── items (one function per review item, P11.2.3) ─────────────────────────────

ITEMS = [placeholder]

section("review items (P11.2.3)")
for fn in ITEMS:
    fn()

raise SystemExit(summary())
