from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from utils.types import State
from langchain_core.messages import ToolMessage
from typing import Annotated
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId


@tool(description="Show all suitable jobs identified so far (not applied yet).")
def show_suitable_jobs(
    config: RunnableConfig,
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    job_results = state.get("job_results", {})
    
    suitable_jobs = [
        url for url, job in job_results.items()
        if job.get("suitable", False) and not job.get("applied", False)
    ]
    
    if not suitable_jobs:
        content = "No suitable jobs found yet. Run the workflow to filter jobs."
    else:
        content = "Suitable Jobs:\n" + "\n".join(f"- {url}" for url in suitable_jobs)
    
    return Command(update={
        "messages": [
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id
            )
        ]
    })
