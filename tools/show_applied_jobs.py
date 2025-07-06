from langgraph.prebuilt import InjectedState
from typing import Annotated
from langchain_core.tools import tool
from utils.types import State


@tool(description="Show a list of jobs the user has applied to.")
def show_applied_jobs(state: Annotated[State, InjectedState]) -> str:
    applied = [url for url, desc in state.get("job_results", {}).items() if desc.get("applied") is True]    
    if not applied:
        return "You haven't applied to any jobs yet."
    return "Jobs you've applied to:\n" + "\n".join(applied)
