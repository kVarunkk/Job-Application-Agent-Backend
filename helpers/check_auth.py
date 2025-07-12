from dotenv import load_dotenv
load_dotenv()

from fastapi import Request, HTTPException, Depends
import httpx
from helpers.supabase import url, key


async def get_current_user(request: Request) -> dict:
    auth = request.headers.get("authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = auth[7:]

    # Allow service role key (used by CRON/internal systems)
    if token == key and request.headers.get("X-Internal-Call") == "true":
        return {
            "id": "internal_cron",
            "email": "cron@internal",
            "role": "service",
            "is_cron": True,
        }

    # Otherwise, treat it as a user JWT
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{url}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": key,
            }
        )
        if res.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid token")

        return res.json()