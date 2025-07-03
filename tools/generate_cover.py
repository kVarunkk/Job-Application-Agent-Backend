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
from langchain_groq import ChatGroq

@tool(description="Generate a tailored cover letter using the user's resume and a given job url. Takes the job URL as input, and stores the generated cover letter inside `job_results` in state.")
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
                    content="⚠️ Missing thread ID or resume path. Cannot compare job.",
                    tool_call_id=tool_call_id
                )
            ]
        })

    try: 
        # Cache check: if resume text isn't available, download and extract it
        if thread_id not in resume_text_cache:
            try:
                # Download resume
                response = (
                    supabase.storage
                    .from_("resumes")
                    .download(resume_path)
                )
    
                # Save to temporary file
                temp_path = f"/tmp/resume-{thread_id}.pdf"
                with open(temp_path, "wb") as f:
                    f.write(response)
    
                # Extract and cache text
                resume_text = read_pdf(temp_path)
                resume_text_cache[thread_id] = resume_text
    
                # Clean up
                os.remove(temp_path)
    
            except Exception as e:
                raise Exception("Failed to read resume")
    
        resume_text = resume_text_cache[thread_id]
        job_description = state.get("job_results", {}).get(job_url, {}).get("description")
    
        if not job_description:
            try:
                job_description = await fetch_desc(job_url)
            except Exception as e:
                return Command(update={
                    "messages": [
                        ToolMessage(
                            content=f"❌ Failed to fetch job description for {job_url}: {str(e)}",
                            tool_call_id=tool_call_id
                        )
                    ]
                })
    
        # Summarize both
        summarized_resume = summarize_text(resume_text, 10)
        summarized_job = summarize_text(job_description, 10)
    
        prompt = f"""
        Based on the following resume:
        {summarized_resume}
    
        Write a tailored cover letter of around 200 words for this job description:
        {summarized_job}
        """
    
        # Generate cover letter
        llm = ChatVertexAI(model="gemini-2.0-flash-lite-001")
        # llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=os.getenv("GROQ_API_KEY"))
        content = llm.invoke(prompt).content
        response = " ".join(str(c) for c in content) if isinstance(content, list) else str(content)
    
        # Update state
        job_results = state.get("job_results", {})
        job_results[job_url] = job_results.get(job_url, {})
        job_results[job_url]["cover_letter"] = response
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
                    content=f"❌ Unexpected error during generating cover: {str(e)}",
                    tool_call_id=tool_call_id
                )
            ]
        })
