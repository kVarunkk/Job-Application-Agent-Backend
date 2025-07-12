from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from playwright.async_api import async_playwright
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from utils.types import State
from fastapi import HTTPException
from helpers.decrypt import decrypt_aes_key, decrypt_password
from helpers.supabase import supabase
from helpers.remoteok.scrape_jobs_core import scrape_jobs_core_remoteok

@tool(description="Scrape a list of job posting URLs from the Remote OK job board using a URL that contains the job postings from the config that has already been provided by the user. If user does not provide the number of job posting URLs to scrape, then take it as 5. You also have access to the job urls seen by the user in the current session, so you can avoid scraping those again.")
async def scrape_jobs_remoteok(
    config: RunnableConfig,
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    no_jobs: int = 10
) -> Command:
    try:
        filter_url = config.get("configurable", {}).get("filter_url", "")
        seen_urls = list(state.get("job_results", {}).keys())

        new_urls = await scrape_jobs_core_remoteok(filter_url, seen_urls, no_jobs)

        updated_results = state.get("job_results", {})
        for url in new_urls:
            updated_results[url] = {}

        return Command(update={
            "job_results": updated_results,
            "messages": [
                ToolMessage(
                    content=f"✅ Scraped {len(new_urls)} new job(s).",
                    tool_call_id=tool_call_id
                )
            ]
        })

    except Exception as e:
        print(f"[scrape_jobs tool] ERROR: {e}")
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"❌ Error scraping jobs: {str(e)}",
                    tool_call_id=tool_call_id
                )
            ]
        })

