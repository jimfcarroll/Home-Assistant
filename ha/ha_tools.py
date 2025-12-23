from datetime import date, datetime
from typing import Any, List, TypedDict, cast

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.outputs import LLMResult
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain_community.document_loaders import PlaywrightURLLoader

import patches


class ReasoningStdOutCallback(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.reasoning = []
        self.mode = 0

    # Called once per token (LangChain-level)
    async def on_llm_new_token(self, token: str, chunk: Any, **kwargs):
        doIt = False
        if token:
            if isinstance(token, str):
                if self.mode != 2:
                    print("\nRESPONDING", flush=True)
                    self.mode = 2
                self.tokens.append(token)
                print(token, end="", flush=True)
            elif isinstance(token, list):
                reasoning : str | None = next((d['reasoning'] for d in token if 'reasoning' in d), None)
                if reasoning:
                    if self.mode != 1:
                        print("\nREASONING", flush=True)
                        self.mode=1
                    print(reasoning, end="", flush=True)
            else:
                doIt = True
        else:
            doIt = True
    
        if doIt:
            print("\n===============================")
            for key, value in kwargs.items():
                print(f"{key} = {value}")            
            print("===============================", flush=True)
            msg : AIMessage =  getattr(chunk, "message", None)
            content_blocks = msg.content_blocks
            print(content_blocks)



    async def on_llm_end(self, response: LLMResult, **kwargs):
        # non-streaming path
        gen = response.generations[0][0]
        msg = gen.message

        if isinstance(msg, AIMessage):
            content = msg.content or ""
            entries = {}
            if isinstance(content, list): # this has several parts.
                for entry in content:
                    ty = entry.get("type")
                    val = entry.get(ty, None) if ty else None
                    if val:
                        entries[ty] = val
            else:
                entries["text"] = cast(str, content) or ""
            reasoning = entries.get("reasoning", None)
            if reasoning:
                print(f"REASONING\n{reasoning}")
            text = entries.get("text", None)
            if text:
                print(f"RESPONDING\n{text}")

    def get_raw_text(self):
        return "".join(self.tokens)

    def get_reasoning(self):
        return "".join(self.reasoning)

# ---- Web Tool ----

@tool
def web_search(query: str) -> str:
    """
    Search the internet for current facts and information.

    Format your query starting with the intent first, then the context.
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        index = 0
        ret = "".join(
            f"\nRESULT {i}, URL={r['href']} DESCRIPTION: {r['body']}"
            for i,r in enumerate(results)
        )
        print("\n======== web search ==========")
        print(ret)
        print("==============================", flush=True)
        return ret

@tool
def read_web_page(url: str) -> str:
    """
    Fetch and read the contents of a web page, extracting visible text for further analysis.
    """
    try:
        loader = PlaywrightURLLoader(urls=[url])
        docs = loader.load()
        print("\n======== page read ==========")
        print(docs[0].page_content[:500])  # preview only
        print("==============================", flush=True)
        return docs[0].page_content[:5000]
    except Exception as e:
        return f"Failed to render page: {e}"
    
# ---- Model ----

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
    model="current",
    streaming=False,
    callbacks=[ReasoningStdOutCallback()]
).bind_tools([web_search, read_web_page])

# ---- Model Node ----
class AgentState(TypedDict):
    messages: List[BaseMessage]

def model_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# ---- Tool Node ----

def tool_node(state: AgentState):
    last = state["messages"][-1]

    if not last.tool_calls:
        return state

    tool_call = last.tool_calls[0]
    tool_name = tool_call["name"]
    args = tool_call["args"]

    print(f"(tool:{tool_name} {args})", end="", flush=True)

    # Dynamically get the tool by name
    tool_func = globals().get(tool_name)
    result = tool_func.invoke(args) if tool_func else f"Tool '{tool_name}' not implemented."

    return {
        "messages": state["messages"] + [
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        ]
    }

# ---- Routing ----

def should_use_tool(state: AgentState):
    last = state["messages"][-1]
    ret = "tool" if getattr(last, "tool_calls", None) else "__end__"
    print("\n======== should use tool ==========")
    print(f"{last}\n\n{ret}")
    print("==============================", flush=True)
    return ret


# ---- Graph ----

builder = StateGraph(AgentState)
builder.add_node("model", model_node)
builder.add_node("tool", tool_node)

builder.set_entry_point("model")
builder.add_conditional_edges("model", should_use_tool)
builder.add_edge("tool", "model")

graph = builder.compile()

# ---- Run ----

result = graph.invoke({
    "messages": [
        SystemMessage(
            content=f"""
You are a personal and home assistant running locally in a private setting for adults.

Behavior:
- Responses must be factual, objective, and concise.
- Do not moralize.
- Do not add disclaimers unless explicitly required.
- Use tools when they provide more accurate or up-to-date information than internal knowledge.
- If a tool is required to answer accurately, invoke it instead of guessing.

Context:
- Location: Conshohocken, PA 19428
- Date: {date.today()}
- Time: {datetime.now().strftime("%H:%M:%S.%f")}
"""
        ),
        HumanMessage(
            content="Why are you?"
        )
    ]
})

#print(result["messages"][-1].content)
