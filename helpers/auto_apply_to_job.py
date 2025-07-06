from playwright.async_api import async_playwright

async def auto_apply_to_job(job_url: str, username: str, password: str, cover_letter: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Block unnecessary resources
        await page.route("**/*", lambda route, request: (
            route.abort() if request.resource_type in ["image", "font"] else route.continue_()
        ))

        # Login
        await page.goto("https://account.ycombinator.com/?continue=https%3A%2F%2Fwww.workatastartup.com%2F", wait_until="domcontentloaded")
        await page.fill('input[name="username"]', username)
        await page.fill('input[name="password"]', password)
        await page.click('button[type="submit"]')
        await page.wait_for_timeout(5000)

        # Apply to job
        await page.goto(job_url, timeout=10000)
        await page.wait_for_selector("text=Apply", timeout=10000)
        await page.click("text=Apply")
        await page.wait_for_selector("textarea", timeout=10000)
        await page.fill("textarea", cover_letter)
        await page.click("button:has-text('Send')")
        await page.wait_for_timeout(3000)

        await browser.close()
