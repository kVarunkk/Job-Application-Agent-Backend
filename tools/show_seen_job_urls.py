from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from typing import Annotated
from utils.types import State


@tool(description="Return all job URLs fetched so far in this session.")
def show_seen_job_urls(state: Annotated[State, InjectedState]) -> str:
    job_urls = state.get("job_urls_seen", [])
    if not job_urls:
        return "No job URLs have been fetched yet."
    return "\n".join(f"{i+1}. {url}" for i, url in enumerate(job_urls))