from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessageChunk
from helpers.graph import graph_builder
import os
from helpers.supabase import supabase
from typing import AsyncGenerator, cast
import json
import asyncio
from helpers.workflow_graph import builder

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_URI = os.environ["DB_URI"]
KEK_SECRET = os.environ["KEK_SECRET"]

@app.post("/chat/{agent_id}")
async def chat(agent_id: str, request: Request):
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
            "resume_path": agent["resume_path"]
        }
    }

    async def event_stream() -> AsyncGenerator[str, None]:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
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
async def run_workflow(agent_id: str, request: Request):
    # Get agent config from Supabase
    workflow_res = supabase.table("workflows").select("*, agents(filter_url, resume_path)").eq("agent_id", agent_id).single().execute()
    if getattr(workflow_res, "error", None) is not None or not workflow_res.data:
        raise HTTPException(status_code=404, detail="Workflow not found.")
        
    workflow = workflow_res.data
    agent = workflow.get("agents")

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

    # Construct config
    config: RunnableConfig = {
        "configurable": {
            "thread_id": agent_id,
            "filter_url": agent.get("filter_url", ""),
            "resume_path": agent.get("resume_path", ""),
            "max_jobs_to_apply": max_jobs_to_apply,
            "similarity_threshold": similarity_threshold,
            "required_keywords": json.dumps(list(required_keywords)),
            "excluded_keywords": json.dumps(list(excluded_keywords)),
            "interval": interval,
            "job_title_contains": json.dumps(list(job_title_contains)),
            "auto_apply": auto_apply
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