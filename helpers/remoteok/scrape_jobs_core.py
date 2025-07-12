from typing import Dict, List
from playwright.async_api import async_playwright

async def scrape_jobs_core_remoteok(
    filter_url: str,
    existing_urls: List[str],
    no_jobs: int = 10
) -> List[str]:
    new_urls = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        await context.route("**/*", lambda route, request: (
            route.abort() if request.resource_type in ["image", "font"] else route.continue_()
        ))
        page = await context.new_page()

        await page.goto(filter_url, timeout=60000)
        await page.wait_for_selector("tr.job[data-url]", timeout=10000)

        scrolls_done = 0
        max_scrolls = 30

        while len(new_urls) < no_jobs and scrolls_done < max_scrolls:
            rows = await page.locator("tr.job[data-url]").all()

            for row in rows:
                url = await row.get_attribute("data-url")
                if url:
                    full_url = f"https://remoteok.com{url}" 
                    if full_url not in existing_urls and full_url not in new_urls:
                        new_urls.add(full_url)
                        if len(new_urls) >= no_jobs:
                            break

            prev_height = await page.evaluate("document.body.scrollHeight")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(3000)
            new_height = await page.evaluate("document.body.scrollHeight")

            scrolls_done += 1
            if new_height == prev_height:
                break

        await browser.close()

    return list(new_urls)

