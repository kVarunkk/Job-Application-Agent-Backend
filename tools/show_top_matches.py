from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from typing import Annotated
from utils.types import State


@tool(description="Show top N job matches based on similarity scores.")
def show_top_matches(state: Annotated[State, InjectedState], top_n: int = 3) -> str:
    job_results = state.get("job_results", {})
    scored_jobs = [
        (url, res["score"]) for url, res in job_results.items() if "score" in res and res["score"] is not None
    ]
    scored_jobs.sort(key=lambda x: x[1], reverse=True)
    if not scored_jobs:
        return "No similarity scores available yet."

    top_jobs = scored_jobs[:top_n]
    return "\n".join([f"{i+1}. {url} (Score: {score:.2f})" for i, (url, score) in enumerate(top_jobs)])
