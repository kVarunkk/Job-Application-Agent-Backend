from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from playwright.async_api import async_playwright
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from utils.types import State
from helpers.fetch_desc import fetch_desc

@tool(description="Fetch and store the job description for a given job URL.")
async def fetch_description(
    job_url: str,
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    try:
        description = await fetch_desc(job_url)
        # Update state
        updated_results = state.get("job_results", {})
        if job_url not in updated_results:
            updated_results[job_url] = {}
        updated_results[job_url]["description"] = description
    
        return Command(update={
            "job_results": updated_results,
            "messages": [
                ToolMessage(f"Fetched job description for {job_url}", tool_call_id=tool_call_id)
            ]
        })

    except Exception as e:
        return Command(update={
            "messages": [
                   ToolMessage(
                       content=f"❌ Unexpected error during fetching job description: {str(e)}",
                       tool_call_id=tool_call_id
                   )
               ]
        })
