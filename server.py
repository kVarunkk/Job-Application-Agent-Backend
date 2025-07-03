from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessageChunk
from helpers.graph import graph_builder
import os
from helpers.supabase import supabase
from typing import AsyncGenerator, cast
import json
import asyncio

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
