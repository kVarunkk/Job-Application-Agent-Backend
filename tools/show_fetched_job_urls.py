from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from typing import Annotated
from utils.types import State


@tool(description="Show the list of job postings and descriptions that have already been fetched.")
def show_fetched_job_urls(state: Annotated[State, InjectedState]) -> str:
    job_urls = state.get("job_urls", [])
    if not job_urls:
        return "No jobs have been fetched yet."

    return "\n\n".join([
        f"{i+1}. {url}"
        for i, url in enumerate(job_urls)
    ])