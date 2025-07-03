from playwright.async_api import async_playwright


async def fetch_desc(url: str) -> str:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=45000)
            await page.wait_for_timeout(2000)
            title = await page.locator(".company-title").all_inner_texts()
            sections = await page.locator(".company-section").all_inner_texts()
            contents = await page.locator(".prose").all_inner_texts()
            parts = [f"## {s.strip()}\n{c.strip()}" for s, c in zip(sections, contents)]
            description = f"# {' '.join(title)}\n\n" + "\n\n".join(parts)
            await browser.close()
            return description
    except Exception as e:
        raise Exception(e)
