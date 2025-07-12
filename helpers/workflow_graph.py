from langgraph.graph import StateGraph, START
from utils.types import State
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from tools.scrape_jobs import scrape_jobs
from langgraph.prebuilt import ToolNode
from helpers.supabase import supabase
from helpers.read_pdf import read_pdf
import os
from helpers.shared import model
from tools.generate_cover import generate_cover
from tools.auto_apply import auto_apply
from sentence_transformers import util
from helpers.fetch_desc import fetch_desc
from helpers.remoteok.fetch_desc import fetch_desc_remoteok
from langchain_google_vertexai import ChatVertexAI
from helpers.generate_cover_letter_for_job import generate_cover_letter_for_job
from helpers.shared import resume_text_cache
from helpers.decrypt import decrypt_aes_key, decrypt_password
from helpers.auto_apply_to_job import auto_apply_to_job
from helpers.scrape_jobs_core import scrape_jobs_core
from helpers.remoteok.scrape_jobs_core import scrape_jobs_core_remoteok
import json
from typing import Dict
from datetime import datetime
from helpers.send_workflow_completion_email import send_success_email, send_error_email


def entry_node(state: State, config: RunnableConfig):
    return {
        "started_at": datetime.utcnow().isoformat()
    }

def check_for_jobs(state: State, config: RunnableConfig):
    job_results = state.get("job_results", {})
    job_urls_seen = set(state.get("job_urls_seen", []))
    no_jobs = config.get("configurable", {}).get("max_jobs_to_apply", 5)
    auto_apply = config.get("configurable", {}).get("auto_apply", False)

    if auto_apply:
        # Case A: Auto-apply mode → reuse suitable existing jobs
        unapplied = [
            job for job in job_results.values()
            if not job.get("applied", False) and job.get("suitable", False)
        ]
    else:
        # Case B: Fetch mode → exclude seen jobs
        unapplied = [
            job for url, job in job_results.items()
            if not job.get("applied", False) and url not in job_urls_seen
        ]

    print("inside check_for_jobs. unapplied jobs: ", len(unapplied))
    
    if len(unapplied) < no_jobs:
        return "scrape"
    return "fetch_descriptions"


async def scrape_jobs_node(state: State, config: RunnableConfig) -> Dict:
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "")
        filter_url = config.get("configurable", {}).get("filter_url", "")
        no_jobs = config.get("configurable", {}).get("max_jobs_to_apply", 5)
        agent_type = config.get("configurable", {}).get("agent_type", "ycombinator")
        seen_urls = list(state.get("job_results", {}).keys())
        new_urls = []

        if (agent_type == "ycombinator"):
            creds_res = supabase.table("encrypted_credentials_yc").select("*").eq("agent_id", thread_id).single().execute()
            if getattr(creds_res, "error", None) or not creds_res.data:
                raise Exception("Failed to fetch credentials")
    
            creds = creds_res.data
            username = creds.get("username") or ""
            padded_aes_key = decrypt_aes_key(creds["aes_key_enc"])
            pad_len = padded_aes_key[-1]
            aes_key = padded_aes_key[:-pad_len]
            password = decrypt_password(creds["password_enc"], aes_key)
    
            new_urls = await scrape_jobs_core(username, password, filter_url, seen_urls, no_jobs)
        elif (agent_type == "remoteok"):
            new_urls = await scrape_jobs_core_remoteok(filter_url, seen_urls, no_jobs)    


        print("new urls: ", new_urls)

        updated_results = state.get("job_results", {})
        for url in new_urls:
            updated_results[url] = {}

        return {"job_results": updated_results, "not_enough_urls": len(new_urls) < no_jobs}

    except Exception as e:
        print(f"[scrape_jobs_node] ERROR: {e}")
        return {"job_results": state.get("job_results", {})}



# 3. Custom Node: Fetch descriptions in bulk
async def fetch_descriptions(state: State, config: RunnableConfig):
    job_results = state.get("job_results", {})
    updated_results = dict(job_results)
    agent_type = config.get("configurable", {}).get("agent_type", "ycombinator")


    for job_url, job_data in job_results.items():
        if job_data.get("applied", False) or (job_data.get("description") and len(job_data.get("description", "")) > 30):
            continue  # skip applied jobs

        try:
            if (agent_type == "ycombinator"):
               description = await fetch_desc(job_url)
               updated_results[job_url]["description"] = description
            elif (agent_type == "remoteok"):
                description = await fetch_desc_remoteok(job_url)
                updated_results[job_url]["description"] = description   
        except Exception as e:
            updated_results[job_url]["description"] = ""

    print("inside fetch description node")
    return {"job_results": updated_results}



async def compare_jobs_bulk(state: State, config: RunnableConfig):
    job_results = state.get("job_results", {})
    resume_path = config.get("configurable", {}).get("resume_path", "")
    thread_id = config.get("configurable", {}).get("thread_id", "")

    try:
        response = supabase.storage.from_("resumes").download(resume_path)
        temp_path = f"/tmp/resume-{thread_id}.pdf"
        with open(temp_path, "wb") as f:
            f.write(response)
        resume_text = read_pdf(temp_path)
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        os.remove(temp_path)
    except Exception:
        return {}

    for job_url, job_data in job_results.items():
        if not job_data.get("applied", False):
           description = job_data.get("description", "")
           if not description:
               continue
           try:
               job_embedding = model.encode(description, convert_to_tensor=True)
               similarity = util.cos_sim(resume_embedding, job_embedding).item()
               job_data["score"] = similarity
           except Exception:
               job_data["score"] = 0.0

    print("inside compare jobs bulk")        

    return {
        "job_results": job_results,
    }


# 5. Keyword filter logic
def filter_keywords(state: State, config: RunnableConfig):
    job_results = state.get("job_results", {})
    job_urls_seen = set(state.get("job_urls_seen", []))  
    suitable_jobs_scraped_or_applied_in_current_run= state.get("suitable_jobs_scraped_or_applied_in_current_run", [])
    similarity_threshold = config.get("configurable", {}).get("similarity_threshold", 0.0)
    included = set(json.loads(config.get("configurable", {}).get("required_keywords", "[]")))
    excluded = set(json.loads(config.get("configurable", {}).get("excluded_keywords", "[]")))
    title_included = set(json.loads(config.get("configurable", {}).get("job_title_contains", "[]")))

    for job_url, job in job_results.items():
        if job.get("applied", False):
            continue

        score = job.get("score", 0.0)
        if score < similarity_threshold:
            job["suitable"] = False
            job_urls_seen.add(job_url)
            continue

        desc = job.get("description", "").lower()

        # Keyword inclusion/exclusion logic
        if included and not any(k.lower() in desc for k in included):
            job["suitable"] = False
        elif any(k.lower() in desc for k in excluded):
            job["suitable"] = False
        else:
            job["suitable"] = True

        if (
           job.get("suitable", False)
           and job_url not in job_urls_seen
           and job_url not in suitable_jobs_scraped_or_applied_in_current_run
        ):
           suitable_jobs_scraped_or_applied_in_current_run.append(job_url)    

        job_urls_seen.add(job_url)

    print("✅ inside filter_keywords")

    return {
        "job_results": job_results,
        "job_urls_seen": list(job_urls_seen),
        "suitable_jobs_scraped_or_applied_in_current_run": suitable_jobs_scraped_or_applied_in_current_run,
    }


def compare_jobs_condition(state: State, config: RunnableConfig):
    job_results = state.get("job_results", {})
    threshold = config.get("configurable", {}).get("similarity_threshold", 0.5)
    no_jobs = config.get("configurable", {}).get("max_jobs_to_apply", 5)
    jobs_available = [job_url for job_url, job_data in job_results.items() if not job_data.get("applied", False) and job_data.get("score", 0.0) >= threshold ]
    if len(jobs_available) >= no_jobs or state.get("not_enough_urls", False):
        return "filter_keywords"
    else: 
        return "scrape"
    
def filter_keywords_condition(state: State, config: RunnableConfig):
    job_results = state.get("job_results", {})
    suitable_jobs_scraped_or_applied_in_current_run= state.get("suitable_jobs_scraped_or_applied_in_current_run", [])
    no_jobs = config.get("configurable", {}).get("max_jobs_to_apply", 5)
    jobs_available = [job_url for job_url, job_data in job_results.items() if job_data.get("suitable", False) and job_url in suitable_jobs_scraped_or_applied_in_current_run ]
    if len(jobs_available) >= no_jobs or state.get("not_enough_urls", False):
        return "generate_cover"
    else: 
        return "scrape"   


# 7. Check auto apply
def check_auto_apply(state: State, config: RunnableConfig):
    print("suitable jobs: ", [job_url for job_url, job_data in state.get("job_results", {}).items() if job_data.get("suitable", False) is True])
    if config.get("configurable", {}).get("auto_apply", False):
        return "apply"
    return "store_workflow_run"

async def generate_covers_bulk(state: State, config: RunnableConfig):
    thread_id = config.get("configurable", {}).get("thread_id")
    resume_path = config.get("configurable", {}).get("resume_path", "")
    job_results = state.get("job_results", {})
    llm = ChatVertexAI(model="gemini-2.0-flash-lite-001")

    if not thread_id or not resume_path:
        return {}

    # Resume cache or download
    if thread_id not in resume_text_cache:
        response = supabase.storage.from_("resumes").download(resume_path)
        temp_path = f"/tmp/resume-{thread_id}.pdf"
        with open(temp_path, "wb") as f:
            f.write(response)
        resume_text = read_pdf(temp_path)
        resume_text_cache[thread_id] = resume_text
        os.remove(temp_path)
    else:
        resume_text = resume_text_cache[thread_id]

    for job_url, job_data in job_results.items():
        if not job_data.get("suitable", False) or (job_data.get("cover_letter") and len(str(job_data.get("cover_letter", ""))) > 20):
            continue  # Skip if already generated

        try:
            job_description = job_data.get("description")
            if not job_description:
                job_description = await fetch_desc(job_url)

            cover_letter = await generate_cover_letter_for_job(
                job_url, resume_text, job_description, llm=llm
            )

            job_data["cover_letter"] = cover_letter
            job_data["description"] = job_description
        except Exception as e:
            continue
            # job_data["cover_letter"] = f"Error: {str(e)}"

    return {"job_results": job_results}


async def auto_apply_bulk(state: State, config: RunnableConfig):
    thread_id = config.get("configurable", {}).get("thread_id") or ""
    job_results = state.get("job_results", {})
    no_jobs = config.get("configurable", {}).get("max_jobs_to_apply", 5)

    
    if not thread_id:
        return {}

    # Fetch and decrypt credentials
    creds_res = supabase.table("encrypted_credentials_yc").select("*").eq("agent_id", thread_id).single().execute()
    if getattr(creds_res, "error", None) or not creds_res.data:
        return {}

    creds = creds_res.data
    username = creds.get("username")
    padded_aes_key = decrypt_aes_key(creds["aes_key_enc"])
    aes_key = padded_aes_key[:-padded_aes_key[-1]]
    password = decrypt_password(creds["password_enc"], aes_key)
    applied_jobs = 0

    if not username:
        return {}

    for job_url, job_data in job_results.items():
        if not job_data.get("suitable") or job_data.get("applied", False) or not job_data.get("cover_letter"):
            continue

        try:
            await auto_apply_to_job(job_url, username, password, str(job_data.get("cover_letter", "")))
            job_data["applied"] = True
            applied_jobs += 1
            if (applied_jobs == no_jobs):
                break
        except Exception as e:
            continue

    return {"job_results": job_results}


async def store_workflow_run_result(state: State, config: RunnableConfig) -> dict:
    try:
        # Extract config values
        config_data = config.get("configurable", {})
        workflow_id = config_data.get("workflow_id")
        no_jobs = config_data.get("max_jobs_to_apply", 5)
        auto_apply = config_data.get("auto_apply", False)
        not_enough_urls = state.get("not_enough_urls", False)

        if not workflow_id:
            raise ValueError("Missing workflow_id in config")

        # Pull data from state
        job_results = state.get("job_results", {})
        suitable_urls = state.get("suitable_jobs_scraped_or_applied_in_current_run", [])

        # Track job counts
        suitable_count = len([
            url for url in suitable_urls
            if job_results.get(url, {}).get("suitable", False)
        ])

        applied_count = len([
            url for url in suitable_urls
            if job_results.get(url, {}).get("applied", False)
        ])

        # Determine success
        if auto_apply:
            success = applied_count >= no_jobs
        else:
            success = suitable_count >= no_jobs

        # Construct workflow_run row
        run_data = {
            "workflow_id": workflow_id,
            "started_at": state.get("started_at") or datetime.utcnow().isoformat(),
            "ended_at": datetime.utcnow().isoformat(),
            "status": "success" if success else "incomplete",
            "error": None if success else ("Job Postings exhausted for the current Job Posting URL. Update your Agent's Job Posting URL." if not_enough_urls else f"{'applied' if auto_apply else 'suitable'} jobs found: {applied_count if auto_apply else suitable_count}, expected: {no_jobs}"),
            "job_results": job_results,
            "suitable_jobs_scraped_or_applied_in_current_run": suitable_urls,
        }

        # Insert into DB
        supabase.table("workflow_runs").insert(run_data).execute()

        supabase.table("workflows").update({"last_run_at": config_data.get("start_time", "")}).eq("id", workflow_id).execute()

        send_success_email(
            to=[config_data.get("user_email", "")],
            agent_name= config_data.get("agent_name", ""),
            summary= f"Workflow completed successfully. {suitable_count} suitable jobs found, {applied_count} applied." if success else (
                f"Workflow completed with issues. {suitable_count} suitable jobs found, {applied_count} applied. "
                f"{'Not enough URLs to scrape.' if not_enough_urls else 'Job Postings exhausted for the current Job Posting URL. Update your Agents Job Posting URL.'}"
            )
        )

        # Return state update
        return {
            "suitable_jobs_scraped_or_applied_in_current_run": [],
            "started_at": "",
            "ended_at": "",
            "not_enough_urls": False,
        }

    except Exception as e:
        # Log failed run
        try:
            config_data = config.get("configurable", {})
            workflow_id = config.get("configurable", {}).get("workflow_id")
            supabase.table("workflow_runs").insert({
                "workflow_id": workflow_id,
                "started_at": state.get("started_at") or datetime.utcnow().isoformat(),
                "ended_at": datetime.utcnow().isoformat(),
                "status": "error",
                "error": str(e),
                "job_results": state.get("job_results", {}),
                "suitable_jobs_scraped_or_applied_in_current_run": state.get("suitable_jobs_scraped_or_applied_in_current_run", []),
            }).execute()

            supabase.table("workflows").update({"last_run_at": config_data.get("start_time", "")}).eq("id", workflow_id).execute()

            send_error_email(
                to=[config_data.get("user_email", "")],
                agent_name=config_data.get("agent_name", ""),
                error_message= str(e)
            )
        except Exception as inner_e:
            print(f"❌ Could not log error to DB: {inner_e}")

        return {
            "suitable_jobs_scraped_or_applied_in_current_run": [],
            "started_at": "",
            "ended_at": "",
            "not_enough_urls": False
        }

builder = StateGraph(State)
builder.add_node("entry_node", entry_node)
builder.add_node("fetch_descriptions", fetch_descriptions)
builder.add_node("compare_jobs", compare_jobs_bulk) 
builder.add_node("scrape", scrape_jobs_node)
builder.add_node("generate_cover", generate_covers_bulk)
builder.add_node("auto_apply", auto_apply_bulk)
builder.add_node("filter_keywords", filter_keywords)
builder.add_node("store_workflow_run", store_workflow_run_result)

builder.set_entry_point("entry_node")
builder.add_conditional_edges("entry_node", check_for_jobs, {
    "scrape": "scrape",
    "fetch_descriptions": "fetch_descriptions"
})
builder.add_edge("scrape", "fetch_descriptions")
builder.add_edge("fetch_descriptions", "compare_jobs")
builder.add_conditional_edges("compare_jobs", compare_jobs_condition, {
    "scrape": "scrape",
    "filter_keywords": "filter_keywords"
})
builder.add_conditional_edges("filter_keywords", filter_keywords_condition, {
    "generate_cover": "generate_cover",
    "scrape": "scrape"
})
builder.add_conditional_edges("generate_cover", check_auto_apply, {
    "apply": "auto_apply",
    "store_workflow_run": "store_workflow_run"
})
builder.add_edge("auto_apply", "store_workflow_run")
builder.add_edge("store_workflow_run", END)