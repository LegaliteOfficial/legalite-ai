# Ask Legalite — chat agent architecture & scaling roadmap

_As of 2026-11-05._

## Why the old Ask Legalite felt rigid

The v3 pipeline was a **citation engine, not a chatbot**:

| Behaviour | Cause |
| --- | --- |
| "Hi" / "thanks" got canned templates | `ConversationalResponder` — regex intent → fixed strings, no LLM |
| Off-topic questions ("help me write an email", "what is 15% of 2,400") got the refusal sentence | every non-conversational turn went through retrieval → JSON-mode generator → grounding gate |
| Legal questions with weak retrieval got *"I could not confidently determine this…"* | hard pre/post-generation refusal gates in `ConfidenceEngine` |
| Follow-ups lost context | memory flattened to a 400-char-per-turn text block inside the user prompt |
| Typos like "Am i goo?" → capability tour | `CONVERSATIONAL_UNCLEAR` template instead of letting a model interpret it |
| "Hi" (2 chars) → HTTP 422 | `AskRequest.question` had `min_length=3` |
| Document assistant broke on long documents | page context was folded into `question`, which was capped at 2,000 chars |

## What shipped (v4): a tool-calling agent

`app/rag/chat_agent.py` — `ChatAgent`, wired into `QAService` behind
`ASK_AGENT_ENABLED` (auto-on with a real OpenAI key; the legacy strict
pipeline stays available with `ASK_AGENT_ENABLED=false`).

```
system prompt (static, cache-friendly)      ─┐
per-request context (date, tenant, on-screen)│
real multi-turn history (12 turns, 6k tok)   ├─► LLM ─► tool calls? ─► search_legal_library ─► HybridRetriever
user message                                ─┘   ▲                    describe_legal_library     + rerank (tenant-scoped)
                                                 └──── numbered passages [n] ◄─────────────────────┘
                                         (≤ AGENT_MAX_TOOL_ROUNDS; final round streams markdown tokens)
```

Key properties:

- **Answers anything.** The persona is "senior Ghanaian legal researcher who is
  also an excellent general assistant". Non-legal questions are answered fully,
  with no forced redirect to law.
- **The model decides when to search** — and is *forced* to on the first round
  when the planner is confident the question is legal (statute/case/procedure
  intents + legal vocabulary). Grounding never depends on the model remembering.
- **Multi-query retrieval.** The model can issue several focused searches per
  turn (parallel tool calls) — query decomposition for multi-issue questions
  comes for free. Passages are numbered turn-globally and deduplicated.
- **Citation integrity is enforced in code**: every `[n]` in the final answer is
  validated against passages actually returned this turn; invented numbers are
  stripped (`validate_citation_markers`). Only *cited* passages become sources.
- **Honest provenance**: `answer_mode` = `grounded` | `general` | `conversational`.
  General-knowledge legal answers are labelled in the UI ("General knowledge")
  and carry a "verify against current law" note. `general` is never shown as
  high confidence.
- **True token streaming** (plain markdown, no partial-JSON extraction) plus
  `status` SSE events ("Searching the Legalite library: '…'").
- **Tenant isolation unchanged** — the search tool is `MultiDocumentReasoner.gather(query, tenant=…)`.
- **Prompt-injection resistance** — tool results and on-screen context are
  framed as data, never instructions.
- **Graceful degradation** — if the LLM is down, greetings fall back to the old
  templates; everything else returns the existing `error` SSE event.
- **Backward compatible** — `AskResponse` keeps every field; the markdown answer
  is `structured_answer.direct_answer`, cited passages populate
  `applicable_law` / `relevant_public_cases` / `firm_similar_cases` by authority tier.

## Workspace tools (v5): the assistant works with the user's own practice

_Added 2026-11-18._ `app/rag/workspace_tools.py`, `app/integrations/workspace_client.py`.

```
browser ── Authorization: Bearer <user JWT> ──► /ask ──► ChatAgent
                                                       │  read tools ──► NestJS GraphQL (same JWT)
                                                       │                 └─ auth, firm tenancy, permissions enforced there
                                                       │  propose_* ──► action_proposed SSE + AskResponse.proposed_actions
                                                       ▼
UI confirmation card ── user clicks Confirm ──► createCase / createDeadline / createDocument … (GraphQL, user's session)
```

- **Delegated access, not a second data path.** The agent reads cases, clients,
  deadlines, calendar, tasks and documents through the existing GraphQL API
  with the signed-in user's own token. It can never see more than the user can
  in the app, and no access rules are duplicated here. The token is only
  forwarded, never stored or logged.
- **Human-in-the-loop writes.** `propose_case|client|deadline|task|document`
  validate and normalise (ISO dates, priorities, case/client ids resolved to
  names) but never write. The UI shows a card; Confirm runs the normal
  mutation (`legalite-app/lib/ai/actions.ts`). An LLM misreading intent can at
  worst show a card the user dismisses.
- **Personalisation.** Each turn's context message carries the user's name,
  firm, city and practice areas (`me`, `currentFirm`, `myPracticeAreas`).
- **Case inference** = `get_case` (facts, parties, documents, deadlines, tasks)
  + `read_document` + `search_legal_library`, then issue → law → application →
  strengths/weaknesses → next steps (prompt section `WORKSPACE_PROMPT`).
- **Drafting.** "Draft a document" produces the full draft inline and a
  `propose_document` card that saves it (markdown converted to the editor's HTML)
  under the right case.
- `answer_mode="workspace"` labels answers drawn only from the user's own data.
- Anonymous callers get the library-only agent and are told to sign in for
  workspace features. Workspace tools turn off when
  `LEGALITE_BACKEND_GRAPHQL_URL` is blank.

Next steps worth doing: durable per-user memory ("prefers British spelling",
"usually acts for landlords") stored in this service and injected into the
context; a `propose_calendar_event` tool; update/complete actions (mark task
done, reschedule deadline) using the same proposal pattern; and an eval set of
workspace conversations.

## Scaling the intelligence further — prioritised roadmap

Ordered by impact ÷ effort. Each item is independent.

### 1. Model strategy (biggest single lever, config-only)
- Set `AGENT_LLM_MODEL` to the strongest model the budget allows; the agent is
  the user-facing brain, and quality scales almost linearly with the model.
  Keep `AGENT_FAST_LLM_MODEL` on a mini model for small talk.
- **Route by difficulty**: a reasoning-tier model (o-series / GPT-5-class) for
  multi-hop legal analysis (problem questions, statute-vs-case reconciliation),
  the standard model for lookups. The planner's intent + question length is a
  good first router; `_sampling_params` already handles reasoning models
  (no `temperature`, `max_completion_tokens`).
- Make the provider pluggable (OpenAI-compatible API → OpenRouter / Azure /
  Anthropic) so models can be A/B-tested without code changes.

### 2. Evaluation loop (prerequisite for every other change)
- Extend `evaluation/` with an **agent golden set**: ~150 real questions spanning
  grounded legal, general legal, off-topic, follow-up chains, adversarial.
- Score with an LLM judge on: correctness, citation faithfulness (does passage
  [n] support the claim?), helpfulness, refusal appropriateness.
- Feed thumbs-down from `query_feedback` into the set weekly.
- Run on every prompt/model change in CI; ship nothing that regresses.

### 3. Long-conversation memory
- Rolling **summary memory**: when a session exceeds the window, summarise the
  older turns into a `chat_sessions.summary` column (one cheap LLM call, async)
  and prepend it to history. Keeps context unbounded at fixed token cost.
- **User/firm memory**: durable facts ("practices land law in Kumasi", "prefers
  British spelling") extracted with consent and injected into the context message.

### 4. Retrieval quality for the agent
- **Title / citation lookup tool**: `get_document(title | case citation)` via
  Postgres FTS on `legal_documents.title` — "summarise Republic v X" currently
  relies on content similarity.
- **Citation-graph tool**: `find_citing_cases(statute, section)` over
  `case_citations` so the agent can follow authority chains.
- `text-embedding-3-large` + tuned RRF weights, measured on the golden set.
- **Contextual retrieval**: prepend a one-line LLM-generated summary of the
  document/section to each chunk before embedding (large recall gains on legal text).

### 5. Web / live-law tool (with guardrails)
- An allow-listed search tool (e.g. Ghana Legal Information Institute,
  parliament.gh, judicial service) for materials not yet in the library, with
  results marked "external, unverified" and never mixed into library citations.

### 6. Latency & cost at scale
- **Prompt caching**: the static system prompt is first and byte-stable, so
  OpenAI's automatic prefix caching applies (~50% cheaper, faster TTFT).
- **Semantic answer cache** (Redis + embeddings) for repeated general questions
  (keyed per tenant; never for tenant-private answers).
- **Async I/O**: move `QAService` to `AsyncOpenAI` + async SQLAlchemy so one
  worker serves many concurrent streams instead of blocking a thread per turn.
- Run retrieval for the forced first search **in parallel** with the first LLM
  call (speculative prefetch) to cut ~1s off legal answers.
- Per-tenant token budgets and usage metering in `ask_total` / a usage table.

### 7. Product surface
- Clickable inline `[n]` chips that open `SourcePreviewDrawer`.
- "Regenerate", "make it shorter/more formal", and "draft as letter" actions.
- File uploads into a chat (PDF → temporary, session-scoped index).

## Configuration reference

| Env var | Default | Purpose |
| --- | --- | --- |
| `ASK_AGENT_ENABLED` | auto | `true`/`false`/blank (blank = on iff a real key) |
| `AGENT_LLM_MODEL` | `OPENAI_LLM_MODEL` | main model |
| `AGENT_FAST_LLM_MODEL` | `AGENT_LLM_MODEL` | small-talk model |
| `AGENT_TEMPERATURE` | 0.3 | ignored for reasoning models |
| `AGENT_MAX_OUTPUT_TOKENS` | 2000 | per LLM call |
| `AGENT_MAX_TOOL_ROUNDS` | 4 | bounds tool loops |
| `AGENT_SEARCH_TOP_K` | 6 | passages per search |
| `AGENT_PASSAGE_MAX_CHARS` | 1800 | passage text shown to the model |
| `AGENT_MEMORY_WINDOW_TURNS` | 12 | history turns |
| `AGENT_MEMORY_MAX_TOKENS` | 6000 | history budget |
| `ASK_CONTEXT_MAX_CHARS` | 20000 | on-screen context cap |
| `LEGALITE_BACKEND_GRAPHQL_URL` | unset | NestJS GraphQL endpoint; unset disables workspace tools |
| `AGENT_WORKSPACE_TOOLS_ENABLED` | true | kill switch for workspace tools |
| `AGENT_WORKSPACE_MAX_TOOL_ROUNDS` | 6 | tool rounds when workspace tools are active |
| `WORKSPACE_REQUEST_TIMEOUT_SECONDS` | 10 | per GraphQL call |

Tests: `tests/unit/test_chat_agent.py` (fake streaming OpenAI client — tool
calls, citation validation, loop bound, dedup, QAService contract, fallback).
