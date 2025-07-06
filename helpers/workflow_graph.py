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
from langchain_google_vertexai import ChatVertexAI
from helpers.generate_cover_letter_for_job import generate_cover_letter_for_job
from helpers.shared import resume_text_cache
from helpers.decrypt import decrypt_aes_key, decrypt_password
from helpers.auto_apply_to_job import auto_apply_to_job
from helpers.scrape_jobs_core import scrape_jobs_core
import json
from typing import Dict

builder = StateGraph(State)

def entry_node(state: State, config: RunnableConfig):
    return {}

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

        creds_res = supabase.table("encrypted_credentials_yc").select("*").eq("agent_id", thread_id).single().execute()
        if getattr(creds_res, "error", None) or not creds_res.data:
            raise Exception("Failed to fetch credentials")

        creds = creds_res.data
        username = creds.get("username")
        padded_aes_key = decrypt_aes_key(creds["aes_key_enc"])
        pad_len = padded_aes_key[-1]
        aes_key = padded_aes_key[:-pad_len]
        password = decrypt_password(creds["password_enc"], aes_key)

        seen_urls = list(state.get("job_results", {}).keys())
        new_urls = await scrape_jobs_core(username, password, filter_url, seen_urls, no_jobs)

        updated_results = state.get("job_results", {})
        for url in new_urls:
            updated_results[url] = {}

        return {"job_results": updated_results}

    except Exception as e:
        print(f"[scrape_jobs_node] ERROR: {e}")
        return {"job_results": state.get("job_results", {})}



# 3. Custom Node: Fetch descriptions in bulk
async def fetch_descriptions(state: State, config: RunnableConfig):
    job_results = state.get("job_results", {})
    no_jobs = config.get("configurable", {}).get("max_jobs_to_apply", 5)
    updated_results = dict(job_results)

    # valid_descriptions = 0

    for job_url, job_data in job_results.items():
        if job_data.get("applied", False) or not job_data.get("description"):
            continue  # skip applied jobs

        # description = job_data.get("description", "")
        # if len(description) >= 30:
            # valid_descriptions += 1
        # else:
        try:
            description = await fetch_desc(job_url)
            updated_results[job_url]["description"] = description
            # if len(description) >= 30:
                # valid_descriptions += 1
        except Exception as e:
            updated_results[job_url]["description"] = ""

        # if valid_descriptions >= no_jobs:
            # break

    print("inside fetch description node")
    return {"job_results": updated_results}



async def compare_jobs_bulk(state: State, config: RunnableConfig):
    job_results = state.get("job_results", {})
    resume_path = config.get("configurable", {}).get("resume_path", "")
    thread_id = config.get("configurable", {}).get("thread_id", "")
    no_jobs = config.get("configurable", {}).get("max_jobs_to_apply", 5)
    threshold = config.get("configurable", {}).get("similarity_threshold", 0.5)

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

    # filtered_jobs = []
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

    # job_results.update({job_url: job_data for job_url, job_data in job_results.items()})
    return {
        "job_results": job_results,
        # "context": {"high_score_jobs": filtered_jobs}
    }


# 5. Keyword filter logic
def filter_keywords(state: State, config: RunnableConfig):
    job_results = state.get("job_results", {})
    no_jobs = config.get("configurable", {}).get("max_jobs_to_apply", 5)
    similarity_threshold = config.get("configurable", {}).get("similarity_threshold", 0.0)
    included = set(json.loads(config.get("configurable", {}).get("required_keywords", [])))
    excluded = set(json.loads(config.get("configurable", {}).get("excluded_keywords", [])))
    title_included = set(json.loads(config.get("configurable", {}).get("job_title_contains", [])))
    job_urls_seen = state.get("job_urls_seen", [])

    # filtered = []
    for job_url, job in job_results.items():
        if not job.get("applied", False) and job.get("description") and job.get("score", 0.0) > similarity_threshold:
           desc = job.get("description", "").lower()
           # title = job.get("title", "").lower()
           if not any(k.lower() in desc for k in included):
               continue
           if any(k.lower() in desc for k in excluded):
               continue
           # if not any(k.lower() in title for k in title_included):
               # continue
           # filtered.append(job_id)
           job["suitable"] = True
           if job_url not in job_urls_seen:
              job_urls_seen.append(job_url)
        # if not job.get("applied", False) and    

    print("inside filter keywords")
    return {
        "job_results": job_results,
        "job_urls_seen": job_urls_seen
    }    


def compare_jobs_condition(state: State, config: RunnableConfig):
    job_results = state.get("job_results", {})
    threshold = config.get("configurable", {}).get("similarity_threshold", 0.5)
    no_jobs = config.get("configurable", {}).get("max_jobs_to_apply", 5)
    jobs_available = [job_url for job_url, job_data in job_results.items() if not job_data.get("applied", False) and job_data.get("score", 0.0) >= threshold ]
    if len(jobs_available) >= no_jobs:
        return "filter_keywords"
    else: 
        return "scrape"
    
def filter_keywords_condition(state: State, config: RunnableConfig):
    job_results = state.get("job_results", {})
    no_jobs = config.get("configurable", {}).get("max_jobs_to_apply", 5)
    jobs_available = [job_url for job_url, job_data in job_results.items() if job_data.get("suitable", False) ]
    if len(jobs_available) >= no_jobs:
        return "generate_cover"
    else: 
        return "scrape"   


# 7. Check auto apply
def check_auto_apply(state: State, config: RunnableConfig):
    if config.get("configurable", {}).get("auto_apply", False):
        return "apply"
    return END

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

    if not username:
        return {}

    for job_url, job_data in job_results.items():
        if not job_data.get("suitable") or job_data.get("applied", False) or not job_data.get("cover_letter"):
            continue

        try:
            await auto_apply_to_job(job_url, username, password, str(job_data.get("cover_letter", "")))
            job_data["applied"] = True
        except Exception as e:
            continue
            # job_data["application_error"] = str(e)

    return {"job_results": job_results}



builder.add_node("entry_node", entry_node)
builder.add_node("fetch_descriptions", fetch_descriptions)
builder.add_node("compare_jobs", compare_jobs_bulk) 
builder.add_node("scrape", scrape_jobs_node)
builder.add_node("generate_cover", generate_covers_bulk)
builder.add_node("auto_apply", auto_apply_bulk)
builder.add_node("filter_keywords", filter_keywords)

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
    END: END
})
builder.add_edge("auto_apply", END)