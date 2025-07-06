from dotenv import load_dotenv
import os
load_dotenv()

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from helpers.summarize import summarize_text
from langchain_google_vertexai import ChatVertexAI
from typing import Annotated
from langgraph.prebuilt import InjectedState
from utils.types import State
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from helpers.supabase import supabase
from helpers.shared import resume_text_cache
from helpers.read_pdf import read_pdf
from helpers.fetch_desc import fetch_desc
from helpers.generate_cover_letter_for_job import generate_cover_letter_for_job

@tool(description="Generate a tailored cover letter using the user's resume and a given job url. Stores the generated cover letter inside `job_results` in state.")
async def generate_cover(
    job_url: str,
    config: RunnableConfig,
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    thread_id = config.get("configurable", {}).get("thread_id")
    resume_path = config.get("configurable", {}).get("resume_path", "")

    if not thread_id or not resume_path:
        return Command(update={
            "messages": [
                ToolMessage(
                    content="⚠️ Missing thread ID or resume path. Cannot generate cover letter.",
                    tool_call_id=tool_call_id
                )
            ]
        })

    try:
        # Load or cache resume text
        if thread_id not in resume_text_cache:
            response = supabase.storage.from_("resumes").download(resume_path)
            temp_path = f"/tmp/resume-{thread_id}.pdf"
            with open(temp_path, "wb") as f:
                f.write(response)
            resume_text_cache[thread_id] = read_pdf(temp_path)
            os.remove(temp_path)

        resume_text = resume_text_cache[thread_id]
        job_description = state.get("job_results", {}).get(job_url, {}).get("description")

        if not job_description:
            job_description = await fetch_desc(job_url)

        cover_letter = await generate_cover_letter_for_job(job_url, resume_text, job_description)

        # Update state
        job_results = state.get("job_results", {})
        job_results[job_url] = job_results.get(job_url, {})
        job_results[job_url]["cover_letter"] = cover_letter
        job_results[job_url]["description"] = job_description

        return Command(update={
            "job_results": job_results,
            "messages": [
                ToolMessage(
                    content=f"📄 Generated a tailored cover letter for job: {job_url}",
                    tool_call_id=tool_call_id
                )
            ]
        })

    except Exception as e:
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"❌ Error during cover generation for {job_url}: {str(e)}",
                    tool_call_id=tool_call_id
                )
            ]
        })
