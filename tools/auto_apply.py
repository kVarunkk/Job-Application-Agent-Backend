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
from helpers.auto_apply_to_job import auto_apply_to_job


@tool(description="Auto-apply to a job by logging in with the user's credentials and submitting the cover letter (stored in state).")
async def auto_apply(
    job_url: str,
    config: RunnableConfig,
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    try:
        thread_id = config.get("configurable", {}).get("thread_id") or ""
        resume_path = config.get("configurable", {}).get("resume_path") or ""
        job_data = state.get("job_results", {}).get(job_url, {})

        if not thread_id:
            raise Exception("Missing thread ID.")
        if not job_data or not job_data.get("cover_letter"):
            raise Exception(f"❌ No cover letter found in state for {job_url}. Please generate one first.")
        if job_data.get("applied", False):
            raise Exception(f"Already applied to the job with URL {job_url}.")

        # Fetch and decrypt credentials
        creds_res = supabase.table("encrypted_credentials_yc").select("*").eq("agent_id", thread_id).single().execute()
        if getattr(creds_res, "error", None) or not creds_res.data:
            raise Exception("Failed to fetch credentials from database.")

        creds = creds_res.data
        username = creds.get("username")
        padded_aes_key = decrypt_aes_key(creds["aes_key_enc"])
        aes_key = padded_aes_key[:-padded_aes_key[-1]]
        password = decrypt_password(creds["password_enc"], aes_key)

        if not username or not password:
            raise Exception("Missing decrypted credentials.")

        await auto_apply_to_job(job_url, username, password, str(job_data.get("cover_letter", "")))

        # Update state
        updated_results = state.get("job_results", {})
        updated_results[job_url]["applied"] = True

        return Command(update={
            "job_results": updated_results,
            "messages": [
                ToolMessage(
                    content=f"✅ Successfully applied to job: {job_url}.",
                    tool_call_id=tool_call_id
                )
            ]
        })

    except Exception as e:
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"❌ Error during job application for {job_url}: {str(e)}",
                    tool_call_id=tool_call_id
                )
            ]
        })
