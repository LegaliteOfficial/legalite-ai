# Evaluation datasets

A dataset here is a JSON array of test cases run through the **live**
`/ask` pipeline (real Qdrant, real Postgres, real OpenAI — see
`evaluation/runner.py`). Run one with:

```bash
python -m evaluation.runner --dataset evaluation/datasets/sample.json
```

`sample.json` ships 20 cases, only 3 of which are `verified: true` — and
those 3 are refusal cases that need no legal expertise to verify (see
"Why some cases don't need a lawyer" below). The other 17 are **templates**:
placeholder questions like `"What does Section [X] of the [Act Name]
provide regarding [Topic]?"` marking a category and difficulty, with every
`expected_*` field empty. **Nobody has written legal ground truth into this
file.** A legal reviewer needs to replace the placeholders with real
questions and real expected answers before any of this means anything — see
`LEGAL_REVIEWER_GUIDE.md` in this directory for the version of that ask
written for a non-technical reviewer.

The runner always prints two separate sections, and it is important they
stay separate:

- **"ALL CASES (exploratory only)"** — computed over every case, including
  unverified guesses. Useful for confirming the pipeline runs end-to-end.
  **Not a quality signal.**
- **"VERIFIED baseline (trustworthy)"** — computed only over cases with
  `verified: true`. This is the only section that should ever be quoted as
  "LegaLite's accuracy is X%". It prints `BLOCKED` instead of a number when
  there are zero verified cases, rather than showing a misleading `0.0`.

## Case schema

```json
{
  "id": "rent-eviction-1",
  "question": "Can a landlord evict a tenant without notice?",
  "category": "statute_lookup",
  "difficulty": "easy",
  "expected_law": "Rent Act",
  "expected_section": "17",
  "expected_article": null,
  "expected_law_type": "act",
  "expected_citations": [],
  "expected_documents": [],
  "expected_passages": [],
  "expected_answer": null,
  "must_refuse": false,
  "verified": false,
  "notes": null
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Unique, stable |
| `question` | yes | Exactly what a user would type into `/ask` |
| `category` | no | One of the categories below; ungrouped cases report as `uncategorized` |
| `difficulty` | no | `easy` \| `medium` \| `hard` — free text, not enforced |
| `expected_law` / `expected_section` / `expected_article` | for single-citation cases | The original, still-supported fields. `expected_law` matches as a case-insensitive substring against a citation's `law_name`; section/article match exactly |
| `expected_citations` | for multi-citation cases | A **list** of `{"law": ..., "section": ..., "article": ...}` — use this instead of the singular fields when a question genuinely needs more than one source (multi-document questions). When both are given, `expected_citations` wins |
| `expected_documents` | optional | Document/law **titles** that should be surfaced *somewhere* in the response (citations or the broader source list), even without a matching section — a weaker signal than a citation match, useful for "did it pull from the right sources" questions |
| `expected_passages` | optional | Substrings expected to appear in a retrieved passage's excerpt. Approximate — excerpts can be truncated, so a miss here means "check by hand", not "definitely wrong" |
| `expected_answer` | optional | For the reviewer's own reference and for a human reading the report later. **No metric compares this to the generated answer** — see "Why there's no answer-similarity score" below |
| `expected_law_type` | no | Not read by any metric yet; reserved |
| `must_refuse` | yes (default `false`) | `true` when the correct behavior is refusal (out-of-corpus, ambiguous, insufficient evidence) |
| `verified` | yes (default `false`) | Set to `true` only after a person with legal-domain knowledge has confirmed every `expected_*` field against the actual corpus. See the checklist below |
| `notes` | optional | Free-text annotator remark. Never read by any metric |

## Categories

- `statute_lookup` — "what does the law say about X"
- `legal_concept` — explain a legal concept using the corpus
- `case_law_lookup` — "is there a case about X"
- `citation_source` — the important issue is identifying the correct authority/citation
- `multi_document` — needs more than one source pulled together
- `statute_case_interaction` — a case has reinterpreted or narrowed a statute
- `current_law` — depends on the corpus actually containing dated/amended material; drop the case if it doesn't
- `exact_section` — the question has exactly one correct provision as the answer
- `citation_lookup` — "what does [specific case/section] say" (legacy name, kept for existing rows; equivalent to `citation_source`)
- `multi_hop` — legacy name for reasoning across sources; equivalent to `multi_document` + `statute_case_interaction`
- `ambiguous_question` — no reasonable single answer; correct behavior is to ask for clarification or refuse
- `insufficient_evidence` — genuinely outside the corpus; correct behavior is refusal

## The legal-review checklist

For each case, the reviewer provides:

```
Question
Expected answer
Expected authority/citation(s)
Expected document(s)
Expected passage(s)
Whether refusal is expected
Difficulty
Category
```

...and confirms, before setting `verified: true`:

```
[ ] The expected answer is legally accurate
[ ] The cited authority actually supports the answer
[ ] The expected document exists in the LegaLite corpus
[ ] The expected passage supports the expected answer
[ ] The expected refusal status is correct
[ ] The category is appropriate
[ ] The difficulty is reasonable
```

Determine the actual correct citation(s) by checking the corpus directly
(`GET /documents/{id}`, or querying `legal_chunks` directly) — never by
trusting what `/ask` currently returns, since that is exactly what's being
evaluated.

**Do not set `verified: true` on a case you have not personally checked
against the corpus.** An eval score is only as trustworthy as its
least-verified case.

## Why some cases don't need a lawyer

`out-of-corpus-1`, `ambiguous-1`, and `insufficient-evidence-002` are
`verified: true` without any legal content in them at all. Their correct
behavior — refuse, because the question is out-of-corpus or has no single
answer — doesn't depend on any specific law being correct. That's a
narrower, cheaper kind of verification than a content-bearing case, and
it's why the dataset can honestly have 3 verified cases already: refusal
cases, not the harder statute/case-law questions that actually need a legal
reviewer's time.

## Why there's no answer-similarity score

It would be easy to add a metric that compares `expected_answer` to the
generated answer and reports a similarity score. It's deliberately not
here: doing that honestly needs either brittle string/fuzzy matching (two
correct answers can be worded completely differently, producing a false
"miss") or an LLM-as-judge (which makes an LLM the arbiter of legal
correctness — exactly the thing this evaluation harness exists to check,
not assume). `expected_answer` stays as a field for a human to read next to
the actual answer; automating that comparison is future work that needs its
own careful design, not a quiet default.

## What `recall@5` and `citation_accuracy@3` actually measure

Both are computed from `AskResponse.citations` — the citations the LLM's
**final answer** referenced, formatted by `app/rag/citation_formatter.py`.
The evaluator has no access to the raw pre-generation retrieval candidate
set (`QAService.ask()` returns only the finished `AskResponse`). So:

- `recall@5` = "did the expected law end up among the top-5 citations the
  *answer* surfaced" — not classic IR recall against the retriever's
  candidate pool. A retrieval miss and a generation miss (retrieved
  correctly, but the LLM chose not to cite it) both look identical here.
- `citation_accuracy@3` = the same, additionally requiring the cited
  section/article to match exactly, within the top 3.
- Neither checks that the citation's passage text actually *supports* the
  answer — only that a citation with a matching law name (and section,
  for `citation_accuracy`) exists somewhere in the response.
- `document_recall` and `passage_support_rate` narrow this further:
  `document_recall` checks whether an expected source was surfaced *at
  all* (citations **or** the broader `sources_used` list), independent of
  section matching; `passage_support_rate` is the only metric that
  inspects passage *text* (`SourceUsed.excerpt`), via substring match.

If a metric has nothing to measure for the current dataset (e.g. no case
sets `expected_passages`), it reports `None` — rendered as `N/A` — never a
misleading `0.0`.

## Target size

15-20 verified cases is enough for a first trustworthy baseline. 100-200 is
a reasonable target once the process is working and the corpus has grown —
don't block the first baseline on reaching that.
