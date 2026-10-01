"""Complete LangGraph RAG fact-checking pipeline."""

import json
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from critic import critique_claim
from judge import judge_claim
from vector_store import Retriever
from verifier import verify_claim


class FactCheckState(TypedDict, total=False):
    claim: str
    paper_id: str | None
    top_k: int
    retrieved_evidence: list[dict]
    evidence: str
    initial_result: dict
    critic_result: dict
    final_result: dict


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Load the embedding model and vector store only once."""
    print("Loading vector store once...", flush=True)
    return Retriever("data/vector_store")


def retrieve_node(state: FactCheckState) -> dict:
    """Retrieve relevant evidence."""
    retriever = get_retriever()

    documents = retriever.retrieve(
        query=state["claim"],
        top_k=state.get("top_k", 5),
        paper_id_filter=state.get("paper_id"),
    )

    evidence_parts = []

    for index, document in enumerate(documents, start=1):
        evidence_parts.append(
            f"[Evidence {index} | source={document['source_id']} | "
            f"type={document['source_type']} | "
            f"similarity={document['distance']:.4f}]\n"
            f"{document['text']}"
        )

    return {
        "retrieved_evidence": documents,
        "evidence": "\n\n".join(evidence_parts),
    }


def verify_node(state: FactCheckState) -> dict:
    """Run the initial verifier."""
    result = verify_claim(
        claim=state["claim"],
        evidence=state.get("evidence", ""),
    )
    return {"initial_result": result}


def critic_node(state: FactCheckState) -> dict:
    """Run the independent critic."""
    initial = state["initial_result"]

    result = critique_claim(
        claim=state["claim"],
        evidence=state.get("evidence", ""),
        initial_verdict=initial["verdict"],
        initial_reason=initial.get("reason", ""),
    )

    return {"critic_result": result}


def judge_node(state: FactCheckState) -> dict:
    """Produce the final reconciled verdict."""
    result = judge_claim(
        claim=state["claim"],
        evidence=state.get("evidence", ""),
        initial_result=state["initial_result"],
        critic_result=state["critic_result"],
    )

    return {"final_result": result}


def build_fact_check_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(FactCheckState)

    workflow.add_node("retrieve_evidence", retrieve_node)
    workflow.add_node("verify_claim", verify_node)
    workflow.add_node("critic_agent", critic_node)
    workflow.add_node("final_judge", judge_node)

    workflow.add_edge(START, "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "verify_claim")
    workflow.add_edge("verify_claim", "critic_agent")
    workflow.add_edge("critic_agent", "final_judge")
    workflow.add_edge("final_judge", END)

    return workflow.compile()


def fact_check(
    claim: str,
    paper_id: str | None = None,
    top_k: int = 5,
) -> dict:
    """Run the complete fact-checking pipeline."""
    graph = build_fact_check_graph()

    return graph.invoke(
        {
            "claim": claim,
            "paper_id": paper_id,
            "top_k": top_k,
        }
    )


if __name__ == "__main__":
    result = fact_check(
        claim="LoRA reduces trainable parameters by 10,000 times.",
        paper_id="lora",
        top_k=3,
    )

    print(json.dumps(result, indent=2, default=str))