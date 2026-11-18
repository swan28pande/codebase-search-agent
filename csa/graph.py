"""LangGraph pipeline: plan -> parallel search agents -> verify -> consensus."""
from __future__ import annotations

import asyncio
import operator
from collections import defaultdict
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .index import CodeIndex, tokenize
from .llm import VLLMClient


class Hit(TypedDict):
    ref: str
    text: str
    score: float
    agent: str
    verified: bool


class State(TypedDict, total=False):
    question: str
    queries: list[str]
    hits: Annotated[list[Hit], operator.add]
    answer: list[dict]


class AgentInput(TypedDict):
    question: str
    query: str
    agent: str


def build_graph(index: CodeIndex, llm: VLLMClient | None = None, k: int = 5, min_votes: int = 2):
    async def plan(state: State) -> State:
        q = state["question"]
        queries = [q, " ".join(dict.fromkeys(tokenize(q)))]
        if llm:
            try:
                text = await llm.chat(
                    "Rewrite the question into 3 short code-search queries (identifiers, keywords). One per line.",
                    q,
                )
                queries += [l.strip("-* ").strip() for l in text.splitlines() if l.strip()][:3]
            except Exception:
                pass  # degrade gracefully to heuristic queries
        return {"queries": list(dict.fromkeys(queries))}

    def fan_out(state: State):
        return [
            Send("search_agent", {"question": state["question"], "query": q, "agent": f"agent-{i}"})
            for i, q in enumerate(state["queries"])
        ]

    async def search_agent(inp: AgentInput) -> State:
        results = await asyncio.to_thread(index.search, inp["query"], k)
        hits = [Hit(ref=c.ref, text=c.text, score=s, agent=inp["agent"], verified=False) for c, s in results]

        async def verify(h: Hit) -> Hit:
            if llm is None:  # heuristic self-check: chunk must share a term with the question
                h["verified"] = bool(set(tokenize(inp["question"])) & set(tokenize(h["text"])))
                return h
            try:
                verdict = await llm.chat(
                    "Answer YES or NO: is this code relevant to the question?",
                    f"Question: {inp['question']}\n\nCode ({h['ref']}):\n{h['text'][:2000]}",
                    max_tokens=3,
                )
                h["verified"] = verdict.upper().startswith("YES")
            except Exception:
                h["verified"] = True
            return h

        return {"hits": list(await asyncio.gather(*(verify(h) for h in hits)))}

    def consensus(state: State) -> State:
        votes, score, text = defaultdict(set), defaultdict(float), {}
        for h in state["hits"]:
            if h["verified"]:
                votes[h["ref"]].add(h["agent"])
                score[h["ref"]] += h["score"]
                text[h["ref"]] = h["text"]
        n_agents = len(state["queries"])
        need = min(min_votes, n_agents)
        ranked = sorted(votes, key=lambda r: (-len(votes[r]), -score[r]))
        answer = [
            {"ref": r, "votes": len(votes[r]), "agents": n_agents, "score": round(score[r], 4), "snippet": text[r]}
            for r in ranked if len(votes[r]) >= need
        ]
        return {"answer": answer[:k]}

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("search_agent", search_agent)
    g.add_node("consensus", consensus)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["search_agent"])
    g.add_edge("search_agent", "consensus")
    g.add_edge("consensus", END)
    return g.compile()


async def search(index: CodeIndex, question: str, llm: VLLMClient | None = None, **kw) -> list[dict]:
    graph = build_graph(index, llm, **kw)
    out = await graph.ainvoke({"question": question, "hits": []})
    return out["answer"]
