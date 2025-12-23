from typing import List, TypedDict

from ddgs import DDGS
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

import patches

class ReasoningStdOutCallback(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.reasoning = []
        self.mode = 0

    # Called once per token (LangChain-level)
    def on_llm_new_token(self, token : str| None, chunk : dict, **kwargs):
        if token:
            if self.mode != 2:
                print(flush=True)
                print("RESPONDING", flush=True)
                self.mode = 2
            self.tokens.append(token)
            print(token, end="", flush=True)
        elif chunk and getattr(chunk, "message", None):
            msg : AIMessage = getattr(chunk, "message")
            reasoning = getattr(msg, "reasoning_content", getattr(msg, "reasoning", None))
            if reasoning:
                if self.mode != 1:
                    print(flush=True)
                    print("REASONING", flush=True)
                    self.mode=1
                print(reasoning, end="", flush=True)

    def get_raw_text(self):
        return "".join(self.tokens)

    def get_reasoning(self):
        return "".join(self.reasoning)

# ---- Web Tool ----

def web_search(query: str):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        ret = "\n\n".join(
            f"{r['title']} - {r['href']}\n{r['body']}"
            for r in results
        )
        print(ret, flush=True)
        return ret

search_tool = Tool(
    name="web_search",
    func=web_search,
    description="Search the internet for current facts and information."
)

# ---- Model ----

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
    model="current",
    streaming=True,
    callbacks=[ReasoningStdOutCallback()]
)
# ---- Model Node ----
class AgentState(TypedDict):
    messages: List[BaseMessage]

def model_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# ---- Tool Node ----

def tool_node(state: AgentState):
    last = state["messages"][-1].content

    print(f"(tool:{last})", end='', flush=True)

    if "SEARCH:" not in last:
        return state

    query = last.split("SEARCH:", 1)[1].strip()
    result = web_search(query)

    return {
        "messages": state["messages"] + [
            SystemMessage(content=f"Search result:\n{result}")
        ]
    }

# ---- Routing ----

def should_use_tool(state: AgentState):
    last = state["messages"][-1].content
    return "tool" if "SEARCH:" in last else "__end__"

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
            content="""
SYSTEM:
You are a personal and home assistent running locally in a private setting for adults. Responses should be factual, objective, and without moralizing.

You are in Conshohocken, PA, 19428.
Today is 12/13/2025. It's 16:26.

TOOLS:
You have access to current web searches for real-time information. When you need the web, output 'SEARCH: <query>'.
"""
        ),
        HumanMessage(
            content="What's tomorrow's weather"
        )
    ]
})

#print(result["messages"][-1].content)
