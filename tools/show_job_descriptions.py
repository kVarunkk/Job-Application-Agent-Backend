from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from typing import Annotated
from utils.types import State


@tool(description="Show job descriptions that have been fetched so far.")
def show_job_descriptions(state: Annotated[State, InjectedState]) -> str:
    job_results = state.get("job_results", {})
    if not job_results:
        return "No job descriptions have been fetched yet."

    result = []
    for i, (url, entry) in enumerate(job_results.items(), start=1):
        desc = entry.get("description", "")
        snippet = desc[:300].strip() + "..." if len(desc) > 300 else desc
        result.append(f"{i}. {url}\n{snippet}")

    return "\n\n".join(result)