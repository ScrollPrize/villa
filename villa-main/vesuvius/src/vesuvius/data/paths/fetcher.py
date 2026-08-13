import aiohttp

async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()
    except aiohttp.ClientError as e:
        print(f"Failed to fetch {url}: {e}")
        return ""