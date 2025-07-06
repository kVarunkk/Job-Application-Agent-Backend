from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from typing import Annotated
from utils.types import State


@tool(description="Show job descriptions either by index (e.g., 1st, 2nd) from the last fetched jobs, or directly using job URLs.")
def show_job_descriptions_by_index_or_url(
    state: Annotated[State, InjectedState],
    indexes: list[int] = [],
    urls: list[str] = [],
) -> str:
    job_results = state.get("job_results", {})
    job_urls = list(job_results.keys())

    response = []

    # Handle index-based lookup
    for idx in indexes:
        if 1 <= idx <= len(job_urls):
            url = job_urls[idx - 1]
            description = job_results.get(url, {}).get("description", "Description not found.")
            response.append(f"### Job {idx}: {url}\n{description[:800]}")
        else:
            response.append(f"Job {idx} not found.")

    # Handle direct URL lookup
    for url in urls:
        description = job_results.get(url, {}).get("description", "Description not found.")
        response.append(f"### Job: {url}\n{description[:800]}")

    if not response:
        return "No valid job indexes or URLs were provided."

    return "\n\n".join(response)
