from typing_extensions import TypedDict, Optional
from typing import Annotated, Any
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph.message import add_messages

class JobResult(TypedDict, total=False):
    description: str
    score: Optional[float]
    cover_letter: Optional[str]
    apply_decision: Optional[bool]
    applied: Optional[bool]
    suitable: Optional[bool]

def merge_job_results(
    left: dict[str, JobResult], right: dict[str, JobResult]
) -> dict[str, JobResult]:
    """Reducer to merge job_results, allowing overwrite by job URL key."""
    if left is None:
        left = {}
    if right is None:
        right = {}
    return {**left, **right}

# --- Define State Type ---
class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    job_urls_seen: list[str]
    job_results: Annotated[dict[str, JobResult], merge_job_results]
    suitable_jobs_scraped_or_applied_in_current_run: list[str]
    started_at: str
    ended_at: str
    not_enough_urls: bool