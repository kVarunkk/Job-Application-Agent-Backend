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

@tool(description="Scrape a list of job posting URLs from the Y Combinator job board using user login and a URL that contains the job postings from the config that has already been provided by the user. If user does not provide the number of job posting URLs to scrape, then take it as 5. You also have access to the job urls seen by the user in the current session, so you can avoid scraping those again.")
async def scrape_jobs(config: RunnableConfig, state: Annotated[State, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], no_jobs: int = 5) -> Command:
    
    try:
        creds_res = supabase.table("encrypted_credentials_yc").select("*").eq("agent_id", config.get("configurable", {}).get("thread_id") or "").single().execute()
        if getattr(creds_res, "error", None) is not None or not creds_res.data:
            raise Exception("Could not fetch credentials")
    
        creds = creds_res.data
        username = creds.get("username")
        padded_aes_key = decrypt_aes_key(creds["aes_key_enc"])
        pad_len = padded_aes_key[-1]
        aes_key = padded_aes_key[:-pad_len] 
        password = decrypt_password(creds["password_enc"], aes_key)
    
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await context.route("**/*", lambda route, request: (
                route.abort() if request.resource_type in ["image", "font"] else route.continue_()
            ))
            page = await context.new_page()
        
            try:
                await page.goto("https://account.ycombinator.com/?continue=https%3A%2F%2Fwww.workatastartup.com%2F", wait_until="domcontentloaded")
                await page.fill('input[name="username"]', username or "")
                await page.fill('input[name="password"]', password or "")
                await page.click('button[type="submit"]')
            except Exception:
                await browser.close()
                raise Exception("Could not login to the YCombinator Portal")
        
            await page.wait_for_timeout(5000)
            if page.url != config.get("configurable", {}).get("filter_url"):
                try:
                    await page.goto(config.get("configurable", {}).get("filter_url") or "", timeout=60000)
                except Exception:
                    await browser.close()
                    raise Exception("Could not open the Job Posting URL")
        
            await page.wait_for_selector("a:has-text('View Job')", timeout=10000)
        
            # Collect hrefs initially
            anchors = await page.locator("a:has-text('View Job')").all()
            hrefs = [await a.get_attribute("href") for a in anchors if await a.get_attribute("href")]
            
            # Scroll if not enough
            max_scrolls = 30
            scrolls_done = 0
        
            while len(hrefs) < no_jobs and scrolls_done < max_scrolls:
                prev_height = await page.evaluate("document.body.scrollHeight")
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000)
                new_height = await page.evaluate("document.body.scrollHeight")
                scrolls_done += 1
        
                if new_height == prev_height:
                    break  # No more scrollable content
        
                # Collect again after scrolling
                anchors = await page.locator("a:has-text('View Job')").all()
                hrefs = [await a.get_attribute("href") for a in anchors if await a.get_attribute("href")]
        
            await browser.close()
        
            seen_urls = set(state.get("job_urls_seen", []))
            new_urls = [url for url in hrefs if url and url not in seen_urls]
            selected_urls = new_urls[:no_jobs]
        
            return Command(update={
                "job_urls": selected_urls,
                "job_urls_seen": list(seen_urls.union(selected_urls)),
                "messages": [
                    ToolMessage(
                        content=f"Fetched {len(selected_urls)} new job postings:\n" + "\n".join(selected_urls),
                        tool_call_id=tool_call_id
                    )
                ]
            })

    
    except Exception as e:
         return Command(update={
            "messages": [
                ToolMessage(
                    content=f"❌ Unexpected error during scraping job postings: {str(e)}",
                    tool_call_id=tool_call_id
                )
            ]
        })
