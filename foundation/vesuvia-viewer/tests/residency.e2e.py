HASH = "#"


async def check(browser):
    return await browser.evaluate("import('./tests/residency.e2e.js').then(module => module.default())")
