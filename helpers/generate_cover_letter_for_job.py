from typing import Optional
from langchain_google_vertexai import ChatVertexAI
from helpers.summarize import summarize_text

async def generate_cover_letter_for_job(
    job_url: str,
    resume_text: str,
    job_description: str,
    llm: Optional[ChatVertexAI] = None
) -> str:
    if not llm:
        llm = ChatVertexAI(model="gemini-2.0-flash-lite-001")

    summarized_resume = summarize_text(resume_text, 10)
    summarized_job = summarize_text(job_description, 10)

    prompt = f"""
    Based on the following resume:
    {summarized_resume}

    Write a tailored cover letter of around 200 words for this job description:
    {summarized_job}
    """

    content = llm.invoke(prompt).content
    response = " ".join(str(c) for c in content) if isinstance(content, list) else str(content)
    return response
