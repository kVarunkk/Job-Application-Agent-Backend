from dotenv import load_dotenv
import os
load_dotenv()

from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from utils.types import State
from tools.scrape_jobs import scrape_jobs
from tools.fetch_descriptions import fetch_description
from tools.compare_jobs import compare_job
from tools.generate_cover import generate_cover
from tools.auto_apply import auto_apply
from tools.show_fetched_job_urls import show_fetched_job_urls
from tools.show_job_descriptions import show_job_descriptions
# from tools.show_seen_job_urls import show_seen_job_urls
from tools.show_job_descriptions_by_index_or_url import show_job_descriptions_by_index_or_url
from tools.show_top_matches import show_top_matches
from tools.show_applied_jobs import show_applied_jobs
from tools.show_cover_letters import show_cover_letters
from tools.list_available_actions import list_available_actions
from tools.filter_jobs_by_keyword import filter_jobs_by_keyword
from tools.compare_jobs_with_each_other import compare_jobs_with_each_other
from tools.find_similar_jobs import find_similar_jobs
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages import RemoveMessage, BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain.agents.chat.output_parser import ChatOutputParser
from langgraph.graph.message import add_messages
from typing import (Union, Literal, Any)
from pydantic import BaseModel
import uuid
from langchain.prompts import ChatPromptTemplate
from typing import Union, Literal
from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel
import uuid
from langchain_groq import ChatGroq


def collect_recent_ai_messages(messages: list) -> dict[str, str]:
    collected = []
    human_prompt = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            human_prompt = msg.content
            break
        if isinstance(msg, AIMessage):
            collected.append(str(msg.content).strip())
    return {
        "prompt": str(human_prompt),
        "ai_messages": "\n\n".join(reversed(collected))
    }

tools = [
    scrape_jobs,
    fetch_description,
    compare_job,
    show_top_matches,
    generate_cover,
    auto_apply,
    show_fetched_job_urls,
    show_job_descriptions,
    # show_seen_job_urls,
    show_cover_letters,
    show_applied_jobs,
    show_job_descriptions_by_index_or_url,
    list_available_actions,
    filter_jobs_by_keyword,
    compare_jobs_with_each_other,
    find_similar_jobs
]


def chatbot_output_condition(
    state: Union[list[BaseMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["tools", "suggest_followups", "__end__"]:
    """
    Route to 'tools' if the last AIMessage has tool_calls,
    'suggest_followups' if the last AIMessage has plain content,
    otherwise end.

    Supports Gemini-style function_call patching.

    Args:
        state: Current agent state.

    Returns:
        Route key.
    """
    # Extract messages
    if isinstance(state, list):
        messages = state
    elif isinstance(state, dict) and (msgs := state.get(messages_key)):
        messages = msgs
    elif hasattr(state, messages_key):
        messages = getattr(state, messages_key)
    else:
        raise ValueError(f"No messages found in input state: {state}")

    if not messages:
        return "__end__"

    last = messages[-1]

    # Patch Gemini function_call → tool_calls
    # if isinstance(last, AIMessage) and not last.tool_calls:
    #     func_call = last.additional_kwargs.get("function_call")
    #     if func_call and isinstance(func_call, dict):
    #         name = func_call.get("name")
    #         args = func_call.get("arguments")
    #         if name and args:
    #             tool_id = last.id or str(uuid.uuid4())
    #             last.tool_calls = [{
    #                 "id": tool_id,
    #                 "name": name,
    #                 "args": args,
    #                 "type": "tool_call"
    #             }]

    if isinstance(last, AIMessage):
        if last.tool_calls:
            return "tools"
        elif last.content and str(last.content).strip():
            return "suggest_followups"

    return "__end__"



llm = ChatVertexAI(model="gemini-2.0-flash-001")
# llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
llm_with_tools = llm.bind_tools(tools)

def filter_messages(state: State) -> dict:
    messages = drop_unresolved_tool_calls(state.get("messages", []))
    return {"messages": messages}


summarization_model = llm.bind(max_tokens=512)
summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=2048,
    max_tokens_before_summary=1024,
    max_summary_tokens=256,
    output_messages_key="messages"
)

def drop_unresolved_tool_calls(messages: list[BaseMessage]) -> list[BaseMessage]:
    resolved_ids = {
        msg.tool_call_id for msg in messages
        if isinstance(msg, ToolMessage) and msg.tool_call_id
    }

    filtered = []
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            tool_calls = msg.tool_calls or []

            # Patch from function_call → tool_call
            # if not tool_calls and "function_call" in msg.additional_kwargs:
            #     func_call = msg.additional_kwargs["function_call"]
            #     if isinstance(func_call, dict) and "name" in func_call:
            #         tool_id = msg.id or str(uuid.uuid4())
            #         tool_calls = [{
            #             "id": tool_id,
            #             "name": func_call["name"],
            #             "args": func_call["arguments"],
            #             "type": "tool_call"
            #         }]
            #         msg.tool_calls = tool_calls

            valid_tool_calls = [tc for tc in tool_calls if tc["id"] in resolved_ids]
            msg.tool_calls = valid_tool_calls

            if str(msg.content).strip() or valid_tool_calls:
                filtered.append(msg)
            else:
                filtered.append(RemoveMessage(id=str(msg.id)))  
        else:
            filtered.append(msg)

    return filtered


def chatbot(state: State) -> dict:
    raw_messages = drop_unresolved_tool_calls(state.get("messages", []))
    messages = [msg for msg in raw_messages if not isinstance(msg, RemoveMessage)]
    return {"messages": [llm_with_tools.invoke(messages)]}


async def suggest_followups(state: State) -> dict[str, Any]:
    messages = state.get("messages", [])
    data = collect_recent_ai_messages(messages)

    followup_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant suggesting 3 concise, actionable follow-up prompts for the user."
     " Use the assistant's last reply and the user's previous prompt to infer context."
     " Follow-up prompts must guide the user toward one of the assistant's capabilities below."
     " Do NOT repeat instructions already given or use vague confirmations like 'okay', 'ready?', etc."
     " Each suggestion should be under 7 words, and actionable."
     "\n\n"
     "Assistant capabilities:\n"
     "- Fetch job postings\n"
     "- Show fetched job URLs or descriptions\n"
     "- Match jobs with the resume\n"
     "- Generate and show cover letters\n"
     "- Apply to jobs\n"
     "- Show applied jobs\n"
     "- Filter or search jobs\n"
     "- Find jobs similar to a job posting\n"
     "- Compare jobs with each other\n",
     ),

    ("human",
     """
     Previous user prompt:
     {prompt}

     Assistant's last reply:
     {last_message}

     Based on the above, suggest 3 helpful next prompts the user can give.
     Respond with only the prompts, each on a new line.
     """)
])


    
    if not data.get("ai_messages"):
        return {"messages": []}
    
    prompt = followup_prompt.invoke({
        "prompt": data.get("prompt"),
        "last_message": data.get("ai_messages")
    })

    output = llm.invoke(prompt.to_messages())

    suggestion_msg = AIMessage(
        content="<-- SUGGESTIONS -->\n" + str(output.content).strip(),
        additional_kwargs={"suggestion_only": True}
    )

    return {
        "messages": [suggestion_msg]
    }    


graph_builder = StateGraph(State)
graph_builder.add_node("summarize", summarization_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("filter_messages", filter_messages)
graph_builder.add_node("suggest_followups", suggest_followups)

graph_builder.add_edge(START, "filter_messages")
graph_builder.add_edge("filter_messages", "summarize")
graph_builder.add_edge("summarize", "chatbot")
graph_builder.add_conditional_edges("chatbot", chatbot_output_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_finish_point("suggest_followups")
