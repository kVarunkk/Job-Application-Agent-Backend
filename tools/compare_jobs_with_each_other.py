from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from typing import Annotated
from utils.types import State
from langgraph.types import Command
from helpers.shared import resume_embedding_store, model
from langchain_core.messages import ToolMessage
from sentence_transformers import SentenceTransformer, util
from helpers.fetch_desc import fetch_desc


@tool(description="Compare two job descriptions by URL and return a similarity score between them.")
async def compare_jobs_with_each_other(
    job_url_1: str,
    job_url_2: str,
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:

    def get_cached_description(url: str) -> str:
        return state.get("job_results", {}).get(url, {}).get("description", "")

    try:
        job_desc_1 = get_cached_description(job_url_1)
        if not job_desc_1:
            job_desc_1 = await fetch_desc(job_url_1)

        job_desc_2 = get_cached_description(job_url_2)
        if not job_desc_2:
            job_desc_2 = await fetch_desc(job_url_2)

        # Compute similarity
        emb_1 = model.encode(job_desc_1, convert_to_tensor=True)
        emb_2 = model.encode(job_desc_2, convert_to_tensor=True)
        similarity = util.cos_sim(emb_1, emb_2).item()

        # Update state
        updated_results = state.get("job_results", {}).copy()
        for url, desc in [(job_url_1, job_desc_1), (job_url_2, job_desc_2)]:
            if url not in updated_results:
                updated_results[url] = {}
            updated_results[url]["description"] = desc

        return Command(update={
            "job_results": updated_results,
            "messages": [
                ToolMessage(
                    content=f"✅ Similarity score between the two jobs: {round(similarity, 2)}",
                    tool_call_id=tool_call_id
                )
            ]
        })

    except Exception as e:
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"❌ Unexpected error during job comparison: {str(e)}",
                    tool_call_id=tool_call_id
                )
            ]
        })
