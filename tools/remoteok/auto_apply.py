from langchain_core.tools import tool,InjectedToolCallId
from playwright.async_api import async_playwright
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState
from typing import Annotated
from utils.types import State
from helpers.decrypt import decrypt_aes_key, decrypt_password
from helpers.supabase import supabase
from langgraph.types import Command
from langchain_core.messages import ToolMessage

@tool(description="Give user the application url of the job according to the job URL and let them know that you cannot apply to the jobs for them on the Remote OK platform.")
async def auto_apply_remoteok(
    job_url: str,
    config: RunnableConfig,
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    try:
        job_data = state.get("job_results", {}).get(job_url, {})
        job_application_slug = job_url[-7:]

        if not job_data:
            raise Exception(f"Job data not found for URL: {job_url}")
        # if not job_data.get("cover_letter"):
        #     raise Exception(f"❌ No cover letter found in state for {job_url}. Please generate one first.")
        return f"You can apply to the job at https://remoteok.com/l/{job_application_slug}. Please visit the link and follow the instructions to apply. Note that I cannot apply to jobs on Remote OK platform for you."
    except Exception as e:
        return f"Error: {str(e)}. Please check the job URL and ensure you have a cover letter generated."
