from dotenv import load_dotenv
load_dotenv()

import asyncio
import datetime
import json
from typing import List

from fastapi import FastAPI
from helpers.supabase import supabase, key
from croniter import croniter
import httpx
import os
from typing import List, AsyncGenerator
from contextlib import asynccontextmanager

AGENT_BACKEND_URL = os.environ.get("SERVER_URL", "") 

# app = FastAPI()

# Helper to check if cron should run now
def cron_matches_now(cron_expr: str, last_run_at: datetime.datetime | None) -> bool:
    now = datetime.datetime.now(datetime.timezone.utc)
    base_time = last_run_at or now - datetime.timedelta(minutes=1)
    iter = croniter(cron_expr, base_time)
    next_run = iter.get_next(datetime.datetime)
    if not isinstance(next_run, datetime.datetime):
        next_run = datetime.datetime.fromtimestamp(next_run, tz=datetime.timezone.utc)
    return now >= next_run and now < next_run + datetime.timedelta(minutes=1)

# Background polling task
async def polling_worker():
    while True:
        try:
            response = supabase.table("workflows").select("*").execute()
            workflows: List[dict] = response.data or []

            for workflow in workflows:
                agent_id = workflow.get("agent_id")
                interval = workflow.get("interval")
                last_run_time = workflow.get("last_run_at")
                
                if not interval or not agent_id:
                    continue

                last_run_dt = None
                if last_run_time:
                    try:
                        last_run_dt = datetime.datetime.fromisoformat(last_run_time)
                    except Exception:
                        pass

                if cron_matches_now(interval, last_run_dt) and not workflow.get("pause"):
                    print(f"Triggering workflow for agent: {agent_id}")

                    # Send request to backend to run workflow
                    try:
                        async def trigger_workflow():
                            async with httpx.AsyncClient() as client:
                                res = client.post(f"{AGENT_BACKEND_URL}/run-workflow/{agent_id}", headers={
                                    "X-Internal-Call": "true",
                                    "Authorization": f"Bearer {key}"
                                })
                                # print("Triggered:", res.status_code, res.text)
                    
                        asyncio.create_task(trigger_workflow())
                    
                    except Exception as e:
                        print(f"Failed to run workflow for {agent_id}: {e}")
                    

        except Exception as e:
            print("[Scheduler Error]", e)

        await asyncio.sleep(60)  # Run every minute

@asynccontextmanager
async def lifespan(app: FastAPI):
    global polling_task
    polling_task = asyncio.create_task(polling_worker())
    yield  # <-- This line allows FastAPI to continue running the app
    polling_task.cancel()  # Clean shutdown on exit
    try:
        await polling_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

@app.get("/")
def health():
    return {"status": "scheduler running"}
