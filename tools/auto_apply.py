from langchain_core.tools import tool
from playwright.async_api import async_playwright
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState
from typing import Annotated
from utils.types import State
from helpers.decrypt import decrypt_aes_key, decrypt_password
from helpers.supabase import supabase


@tool(description="Auto-apply to a job by logging in with the user's credentials and submitting the cover letter (stored in state) to the job page on WorkAtAStartup.")
async def auto_apply(
    url: str,
    config: RunnableConfig,
    state: Annotated[State, InjectedState]
) -> str:
    """Auto-apply to a job using Playwright. Fetches cover letter from state and uses config credentials."""
    try:
        creds_res = supabase.table("encrypted_credentials_yc").select("*").eq("agent_id", config.get("configurable", {}).get("thread_id") or "").single().execute()
        if getattr(creds_res, "error", None) is not None or not creds_res.data:
            Exception("Failed to fetch credentials from database.")
    
        creds = creds_res.data
        username = creds.get("username")
        padded_aes_key = decrypt_aes_key(creds["aes_key_enc"])
        pad_len = padded_aes_key[-1]
        aes_key = padded_aes_key[:-pad_len] 
        password = decrypt_password(creds["password_enc"], aes_key)
        job_data = state.get("job_results", {}).get(url, {})
        
        if not username or not password:
           Exception( "❌ Missing credentials. Cannot proceed with job application.")
        if not job_data or not job_data.get("cover_letter"):
           Exception(f"❌ No cover letter found in state for {url}. Please generate one first.")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            # Block fonts/images for performance
            await page.route("**/*", lambda route, request: (
                route.abort() if request.resource_type in ["image", "font"] else route.continue_()
            ))

            # Login
            await page.goto("https://account.ycombinator.com/?continue=https%3A%2F%2Fwww.workatastartup.com%2F", wait_until="domcontentloaded")
            await page.fill('input[name="username"]', username or "")
            await page.fill('input[name="password"]', password)
            await page.click('button[type="submit"]')
            await page.wait_for_timeout(5000)

            # Navigate to job page
            await page.goto(url, timeout=10000)
            await page.wait_for_selector("text=Apply", timeout=10000)
            await page.click("text=Apply")

            # Submit cover letter
            await page.wait_for_selector("textarea", timeout=10000)
            await page.fill("textarea", str(job_data.get("cover_letter", "")))
            await page.click("button:has-text('Send')")
            await page.wait_for_timeout(3000)

            await browser.close()
            return f"✅ Successfully applied to {url}"
    except Exception as e:
        return f"⚠️ Error applying to {url}: {str(e)}"
