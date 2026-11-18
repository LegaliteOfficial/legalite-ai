# ruff: noqa: E501, TC001
"""Live evaluation of the Ask Legalite chat agent.

Runs every scenario in ``scenarios.py`` through the real ``ChatAgent`` and
real LLM, with production routing (``route_agent_turn``), the real library
retriever (read-only) unless ``--no-library``, and a fake workspace so no
real firm data is read or written.

    python -m evaluation.agent_chat.run                # all scenarios
    python -m evaluation.agent_chat.run --only legal   # by id or category
    python -m evaluation.agent_chat.run --no-library   # stub library search

Writes ``evaluation/agent_chat/results/<timestamp>.jsonl`` and prints a
pass/fail table. Costs real LLM tokens (~50 conversations).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.tenant_context import TenantContext
from app.core.config import get_settings
from app.rag.chat_agent import AgentResult, AgentStatus, ChatAgent, ChatTurn, clean_history_text
from app.rag.conversation_memory import ConversationSummarizer, build_memory_message
from app.rag.legal_memory import LegalMemory
from app.rag.query_planner import QueryPlanner
from app.rag.retriever import RetrievedPassage
from app.rag.workspace_tools import WorkspaceToolkit
from app.services.qa_service import route_agent_turn
from evaluation.agent_chat.fixtures import (
    DownWorkspaceClient,
    FakeWorkspaceClient,
    InMemoryChatRepo,
)
from evaluation.agent_chat.scenarios import S

RESULTS = Path(__file__).parent / "results"


def _library_search() -> Any:
    from app.db.session import SessionLocal
    from app.ingestion.embedding_service import EmbeddingService
    from app.rag.hybrid_retriever import HybridRetriever
    from app.rag.multi_document_reasoner import MultiDocumentReasoner
    from app.rag.query_rewriter import QueryRewriter
    from app.rag.reranker import build_reranker
    from app.vector_store.qdrant_client import get_vector_store

    embeddings = EmbeddingService()
    store = get_vector_store()
    reranker = build_reranker()
    rewriter = QueryRewriter()

    def search(query: str) -> list[RetrievedPassage]:
        db = SessionLocal()
        try:
            reasoner = MultiDocumentReasoner(
                retriever=HybridRetriever(db=db, embeddings=embeddings, vector_store=store),
                reranker=reranker,
                rewriter=rewriter,
            )
            return reasoner.gather(query, tenant=TenantContext.anonymous()).passages
        finally:
            db.close()

    return search


def _stub_search(query: str) -> list[RetrievedPassage]:
    return []


def run_scenario(
    sc: dict[str, Any], agent: ChatAgent, search: Any, summarizer: Any
) -> dict[str, Any]:
    """Run one conversation through the same memory path as production:
    LegalMemory (in-memory repo) -> load_conversation (+ rolling summary)
    -> build_memory_message -> agent."""
    settings = get_settings()
    planner = QueryPlanner()
    memory = LegalMemory(InMemoryChatRepo())  # type: ignore[arg-type]
    sid = memory.get_or_create_session(None)
    window = int(sc.get("window", settings.agent_memory_window_turns))
    turns_out: list[dict[str, Any]] = []
    signed_in = sc.get("signed_in", False)
    for user_msg in sc["turns"]:
        conv = memory.load_conversation(
            sid,
            window_turns=window,
            max_tokens=settings.agent_memory_max_tokens,
            summarizer=summarizer,
        )
        memory_note = build_memory_message(conv)
        history = [
            ChatTurn(
                role="assistant" if t.role == "assistant" else "user",
                content=clean_history_text(t.content, settings.agent_memory_turn_max_chars),
            )
            for t in conv.recent
        ]
        memory.append_user(sid, user_msg)
        workspace = None
        if signed_in == "down":
            workspace = WorkspaceToolkit(DownWorkspaceClient())  # type: ignore[arg-type]
        elif signed_in:
            workspace = WorkspaceToolkit(FakeWorkspaceClient())  # type: ignore[arg-type]
        plan = planner.plan(user_msg, has_memory=bool(history))
        conversational, force = route_agent_turn(plan, user_msg, page_context=None)
        statuses: list[str] = []
        result: AgentResult | None = None
        t0 = time.perf_counter()
        error = None
        try:
            for item in agent.run_stream(
                user_msg,
                history=history,
                search_fn=search,
                force_search=force,
                conversational=conversational,
                workspace=workspace,
                conversation_memory=memory_note,
            ):
                if isinstance(item, AgentResult):
                    result = item
                elif isinstance(item, AgentStatus):
                    statuses.append(
                        f"{item.kind}:{item.query or (item.action.summary if item.action else '')}"
                    )
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
        answer = result.answer if result else ""
        turns_out.append(
            {
                "user": user_msg,
                "intent": plan.intent.value,
                "conversational": conversational,
                "force_search": force,
                "answer": answer,
                "mode": result.mode if result else None,
                "model": result.model if result else None,
                "tools": [
                    {"name": t.name, "args": t.arguments, "n": t.result_count}
                    for t in (result.tool_calls if result else [])
                ],
                "proposals": [a.to_dict() for a in (result.proposed_actions if result else [])],
                "statuses": statuses,
                "cited": len(result.cited_ids) if result else 0,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "error": error,
            }
        )
        memory.append_assistant(sid, answer or "(no answer)")
    failures = check(sc.get("checks", {}), turns_out[-1])
    return {"id": sc["id"], "cat": sc["cat"], "turns": turns_out, "failures": failures}


def check(checks: dict[str, Any], turn: dict[str, Any]) -> list[str]:
    fails: list[str] = []
    if turn["error"]:
        return [f"error: {turn['error']}"]
    names = {t["name"] for t in turn["tools"]}
    answer = turn["answer"]
    if (want := checks.get("tools_any")) and not names & set(want):
        fails.append(f"expected one of tools {want}, got {sorted(names)}")
    if (bad := checks.get("tools_none")) and names & set(bad):
        fails.append(f"unexpected tools {sorted(names & set(bad))}")
    ptypes = [p["type"] for p in turn["proposals"]]
    if (p := checks.get("proposal")) and p not in ptypes:
        fails.append(f"expected proposal {p}, got {ptypes}")
    if checks.get("no_proposal") and ptypes:
        fails.append(f"unexpected proposals {ptypes}")
    if (rx := checks.get("must_match")) and not re.search(rx, answer, re.I):
        fails.append(f"answer lacks /{rx}/")
    if (rx := checks.get("must_not_match")) and (m := re.search(rx, answer, re.I)):
        fails.append(f"answer contains forbidden '{m.group(0)}'")
    if (n := checks.get("max_chars")) and len(answer) > n:
        fails.append(f"answer too long ({len(answer)} > {n})")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="scenario ids or categories")
    ap.add_argument("--no-library", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    scenarios = [s for s in S if not args.only or s["id"] in args.only or s["cat"] in args.only]
    agent = ChatAgent()
    summarizer = ConversationSummarizer()
    search = _stub_search if args.no_library else _library_search()

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(lambda sc: run_scenario(sc, agent, search, summarizer), scenarios)
        )
    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    failed = [r for r in results if r["failures"]]
    for r in results:
        mark = "FAIL" if r["failures"] else "ok  "
        last = r["turns"][-1]
        print(
            f"{mark} {r['id']:<24} {r['cat']:<12} {last['intent']:<26} mode={last['mode']} tools={[t['name'] for t in last['tools']]}"
        )
        for f in r["failures"]:
            print(f"       - {f}")
    print(f"\n{len(results) - len(failed)}/{len(results)} passed  ->  {out_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
