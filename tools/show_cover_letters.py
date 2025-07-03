from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from typing import Annotated
from utils.types import State


@tool(description="Show cover letters generated for jobs.")
def show_cover_letters(state: Annotated[State, InjectedState]) -> dict:
    return {
        url: data["cover_letter"]
        for url, data in state.get("job_results", {}).items()
        if "cover_letter" in data
    }
