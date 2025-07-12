from playwright.async_api import async_playwright


async def fetch_desc_remoteok(url: str) -> str:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=45000)

            # Wait for at least one .html inside .description to load
            await page.wait_for_selector(".description .html", timeout=10000)

            # Get the first HTML block inside .description
            html_handle = page.locator(".description .html").first

            # Option 1: Get plain text
            description = await html_handle.inner_text()

            # ✅ Option 2: Get rich HTML content
            # description = await html_handle.inner_html()

            await browser.close()
            return description.strip()

    except Exception as e:
        raise Exception(f"Error fetching description: {e}")

