from langchain_core.tools import tool


@tool(description="List all the tasks I can help with.")
def list_available_actions() -> str:
    return (
        "I can:\n"
        "- Fetch job postings\n"
        "- Show fetched job URLs or descriptions\n"
        "- Match jobs with your resume\n"
        "- Generate and show cover letters\n"
        "- Apply to jobs\n"
        "- Show applied jobs\n"
        "- Filter or search jobs\n"
        "- Find jobs similar to a job posting\n"
        "- Compare jobs with each other\n"
        "Just ask!"
    )
