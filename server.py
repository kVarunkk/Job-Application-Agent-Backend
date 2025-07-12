from dotenv import load_dotenv
load_dotenv()
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessageChunk
from helpers.graph import build_graph
import os
from helpers.supabase import supabase
from typing import AsyncGenerator, cast
import json
import asyncio
from helpers.workflow_graph import builder
from datetime import datetime, timezone
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from helpers.check_auth import get_current_user

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
from fastapi.responses import Response

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )

FRONTEND_URL= os.environ.get("FRONTEND_URL", "")
SCHEDULER_URL= os.environ.get("SCHEDULER_URL", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, SCHEDULER_URL],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_URI = os.environ["DB_URI"]
KEK_SECRET = os.environ["KEK_SECRET"]

@app.post("/chat/{agent_id}")
@limiter.limit("10/minute")
async def chat(agent_id: str, request: Request, user: dict = Depends(get_current_user)):
    body = await request.json()
    user_message = body.get("message")
    if not user_message:
        raise HTTPException(status_code=400, detail="Missing message.")

    agent_res = supabase.table("agents").select("*").eq("id", agent_id).single().execute()
    if getattr(agent_res, "error", None) is not None or not agent_res.data:
        raise HTTPException(status_code=404, detail="Agent not found.")

    agent = agent_res.data

    config: RunnableConfig = {
        "configurable": {
            "thread_id": agent_id,
            "filter_url": agent["filter_url"],
            "resume_path": agent["resume_path"],
            "agent_type": agent["type"]
        }
    }

    async def event_stream() -> AsyncGenerator[str, None]:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            graph_builder = build_graph(agent["type"])
            graph = graph_builder.compile(checkpointer=checkpointer)

            inputs = {"messages": [HumanMessage(content=user_message)]}
            events = graph.astream(inputs, config=cast(RunnableConfig, config), stream_mode="messages")
            
            async for token, metadata in events:                
                if (
                    not isinstance(token, str)
                    and hasattr(token, "content")
                    and isinstance(token.content, str)
                    and not getattr(token, "additional_kwargs", {}).get("function_call")
                    and not hasattr(token, "tool_call_id")
                    and isinstance(metadata, dict)
                    and metadata.get("langgraph_node") not in ["summarize"]
                ) and not (
                    isinstance(metadata, dict)
                    and metadata.get("langgraph_node") == "suggest_followups"
                    and not getattr(token, "additional_kwargs", {}).get("suggestion_only")
                ):
                    yield f"data: {json.dumps({'text': token.content})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/run-workflow/{agent_id}")
@limiter.limit("10/minute")
async def run_workflow(agent_id: str, request: Request, user= Depends(get_current_user)):
    # Get agent config from Supabase
    workflow_res = supabase.table("workflows").select("*, agents(filter_url, resume_path, type)").eq("agent_id", agent_id).single().execute()
    if getattr(workflow_res, "error", None) is not None or not workflow_res.data:
        raise HTTPException(status_code=404, detail="Workflow not found.")
        
    workflow = workflow_res.data
    agent = workflow.get("agents")
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found in workflow.")

    # Parse optional config override
    # body = await request.json()
    max_jobs_to_apply = workflow.get("no_jobs") or 5
    similarity_threshold = 0.4
    job_title_contains = workflow.get("job_title_contains") or []
    auto_apply = workflow.get("auto_apply") or False
    interval = workflow.get("interval") or "0 8 * * *"
    required_keywords = workflow.get("required_keywords") or []
    excluded_keywords = workflow.get("excluded_keywords") or []
    job_title_contains = workflow.get("job_title_contains") or []
    start_time = datetime.now(timezone.utc).isoformat()

    # Construct config
    config: RunnableConfig = {
        "configurable": {
            "thread_id": agent_id,
            "workflow_id": workflow.get("id"),
            "filter_url": agent.get("filter_url", ""),
            "resume_path": agent.get("resume_path", ""),
            "max_jobs_to_apply": max_jobs_to_apply,
            "similarity_threshold": similarity_threshold,
            "required_keywords": json.dumps(list(required_keywords)),
            "excluded_keywords": json.dumps(list(excluded_keywords)),
            "interval": interval,
            "job_title_contains": json.dumps(list(job_title_contains)),
            "auto_apply": auto_apply,
            "start_time": start_time,
            "agent_type": agent.get("type"),
            "recursion_limit": 50
        }
    }


    # Run the workflow graph
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        workflow_graph = builder.compile(checkpointer=checkpointer)

        try:
            await workflow_graph.ainvoke({}, config=config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={"status": "success", "message": "Workflow completed."})


@app.get("/test")
@limiter.limit("2/minute")
def testFunc(request: Request, user= Depends(get_current_user)):
   return JSONResponse(content={"status": "success"})