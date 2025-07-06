from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated
from utils.types import State


@tool(description="Filter job descriptions by a given keyword or phrase.")
def filter_jobs_by_keyword(keyword: str, state: Annotated[State, InjectedState]) -> list[str]:
    results = []
    for url, desc in state.get("job_results", {}).items():
        if keyword.lower() in desc.get("description", "").lower():
            results.append(url)
    return results
