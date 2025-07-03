import os
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from utils.types import State
from helpers.shared import resume_embedding_store, model
from sentence_transformers import SentenceTransformer, util
from helpers.supabase import supabase
from helpers.read_pdf import read_pdf
from helpers.fetch_desc import fetch_desc

@tool(description="Compare a job description (from state using the job URL) with the user's resume and update similarity score.")
async def compare_job(
    job_url: str,
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    thread_id = config.get("configurable", {}).get("thread_id")
    resume_path = config.get("configurable", {}).get("resume_path", "")

    if not thread_id or not resume_path:
        return Command(update={
            "messages": [
                ToolMessage(
                    content="⚠️ Missing thread ID or resume path. Cannot compare job.",
                    tool_call_id=tool_call_id
                )
            ]
        })

    try:
        try:
            # Download resume from Supabase
            response = (
                supabase.storage
                .from_("resumes")
                .download(resume_path)
            )
            # Save temporarily
            temp_path = f"/tmp/resume-{thread_id}.pdf"
            with open(temp_path, "wb") as f:
                f.write(response)
            # Extract text
            resume_text = read_pdf(temp_path)
            # Embed and cache
            resume_embedding = model.encode(resume_text, convert_to_tensor=True)
            # Clean up
            os.remove(temp_path)
        except Exception as e:
            return Command(update={
                "messages": [
                    ToolMessage(
                        content=f"❌ Error while embedding resume: {str(e)}",
                        tool_call_id=tool_call_id
                    )
                ]
            })

        # Get job description
        job_desc = state.get("job_results", {}).get(job_url, {}).get("description", "")
        if not job_desc:
            try:
                job_desc = await fetch_desc(job_url)
            except Exception as e:
                return Command(update={
                    "messages": [
                        ToolMessage(
                            content=f"❌ Failed to fetch job description for {job_url}: {str(e)}",
                            tool_call_id=tool_call_id
                        )
                    ]
                })

        # Compare embeddings
        job_embedding = model.encode(job_desc, convert_to_tensor=True)
        similarity = util.cos_sim(resume_embedding, job_embedding).item()

        # Update state
        updated_results = state.get("job_results", {})
        if job_url not in updated_results:
            updated_results[job_url] = {}
        updated_results[job_url]["description"] = job_desc
        updated_results[job_url]["score"] = similarity

        return Command(update={
            "job_results": updated_results,
            "messages": [
                ToolMessage(
                    content=f"✅ Comparison completed for job: {job_url}. Score: {round(similarity, 2)}",
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
