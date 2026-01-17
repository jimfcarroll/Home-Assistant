from datetime import date, datetime
from typing import Any, List, TypedDict, cast

from ddgs import DDGS
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.outputs import LLMResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from tts import tts

import patches
from read_web import _post_crawl, _shape_pages_for_llm

debug = True

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
            if debug:
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
                tts(text)

    def get_raw_text(self):
        return "".join(self.tokens)

    def get_reasoning(self):
        return "".join(self.reasoning)

# ---- Web Tool ----

@tool
def web_search(query: str) -> str:
    """
    Search the internet for current facts and information.

    - Format your query starting with the intent first, then the context. 
    - You will get a list of urls back. 
    - use the urls to read the web pages
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        index = 0
        ret = "".join(
            #f"\nURL={r['href']} DESCRIPTION: {r['body']}"
            f"\n{r['href']}"
            for i,r in enumerate(results)
        )
        if debug:
            print("\n======== web search ==========")
            print(f"query:{query}")
            print("========")
            print(ret)
            print("==============================", flush=True)
        return ret

@tool
def read_web_page(url: str) -> str:
    """
    Render a web page locally and return LLM-suitable visible text.
    """
    # """
    # Render and read a web page using a local JS-capable browser.
    # """
    payload = {
        "startUrls": [url],
        "maxPages": 1,
        "maxDepth": 0,
        "sameOriginOnly": True,
        "waitUntil": "domcontentloaded",
        "includeLinks": False,
        "includeHtml": False,
        "timeoutSecs": 30,
    }
    result = _post_crawl(payload)
    res = _shape_pages_for_llm(result, mode="fetch")
    if debug:
        print("\n======== page read ==========")
        print(res[:5000])  # preview only
        print("==============================", flush=True)
    return res


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

    #print(f"(tool:{tool_name} {args})", end="", flush=True)

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
    if debug:
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

Spoken-output requirements (mandatory):
- All numbers MUST be spelled out in words.
- Units MUST be written in full words.
- Abbreviations, symbols, and numerals are NOT allowed.
- Sentences MUST be short and declarative.
- Write exactly what should be spoken aloud.

Context:
- Location: Conshohocken, PA 19428
- Date: {date.today()}
- Time: {datetime.now().strftime("%H:%M:%S.%f")}
"""
        ),
        HumanMessage(
            content="""
What's tomorrow's weather
"""
        )
    ]
})

#print(result["messages"][-1].content)
