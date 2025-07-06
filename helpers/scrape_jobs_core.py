from typing import Dict, List
from playwright.async_api import async_playwright

async def scrape_jobs_core(
    username: str,
    password: str,
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

        # Log in
        await page.goto("https://account.ycombinator.com/?continue=https%3A%2F%2Fwww.workatastartup.com%2F", wait_until="domcontentloaded")
        await page.fill('input[name="username"]', username)
        await page.fill('input[name="password"]', password)
        await page.click('button[type="submit"]')
        await page.wait_for_timeout(5000)

        # Navigate to filtered job page
        await page.goto(filter_url, timeout=60000)
        await page.wait_for_selector("a:has-text('View Job')", timeout=10000)

        scrolls_done = 0
        max_scrolls = 30

        while len(new_urls) < no_jobs and scrolls_done < max_scrolls:
            anchors = await page.locator("a:has-text('View Job')").all()
            for a in anchors:
                href = await a.get_attribute("href")
                if href and href not in existing_urls and href not in new_urls:
                    new_urls.add(href)
                    if len(new_urls) >= no_jobs:
                        break

            prev_height = await page.evaluate("document.body.scrollHeight")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)
            new_height = await page.evaluate("document.body.scrollHeight")

            scrolls_done += 1
            if new_height == prev_height:
                break

        await browser.close()
    return list(new_urls)
