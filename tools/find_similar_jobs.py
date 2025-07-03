from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from utils.types import State
from helpers.shared import model
from sentence_transformers import util
from helpers.fetch_desc import fetch_desc
from typing import Any

# You may already have this
job_embedding_store: dict[str, Any] = {}

@tool("find_similar_jobs", description="Find jobs similar to a given job using embeddings.")
async def find_similar_jobs(
    job_url: str,
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    try:
        job_results = state.get("job_results", {})

        # Ensure we have the job description
        target_desc = job_results.get(job_url, {}).get("description", "")
        if not target_desc:
            target_desc = await fetch_desc(job_url)
    
        # Embed target job if not cached
        if job_url not in job_embedding_store:
            job_embedding_store[job_url] = model.encode(target_desc, convert_to_tensor=True)
    
        target_embedding = job_embedding_store[job_url]
    
        # Embed all other jobs and calculate similarity
        similarities = []
        for other_url, data in job_results.items():
            if other_url == job_url or "description" not in data:
                continue
    
            if other_url not in job_embedding_store:
                job_embedding_store[other_url] = model.encode(data["description"], convert_to_tensor=True)
    
            sim_score = util.cos_sim(target_embedding, job_embedding_store[other_url]).item()
            similarities.append((other_url, sim_score))
    
        # Sort by score descending
        similarities.sort(key=lambda x: x[1], reverse=True)
    
        top_matches = similarities[:5]
        content = f"Top {len(top_matches)} similar jobs to {job_url}:\n\n"
        for url, score in top_matches:
            content += f"- {url} (score: {round(score, 2)})\n"
    
        return Command(update={
            "messages": [ToolMessage(content=content.strip(), tool_call_id=tool_call_id)]
        })

    except Exception as e:
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"❌ Unexpected error in finding similar jobs: {str(e)}",
                    tool_call_id=tool_call_id
                )
            ]
        })
