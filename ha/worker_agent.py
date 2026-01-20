from __future__ import annotations

from datetime import date, datetime

from google.adk.agents import Agent
from google.adk.agents.llm_agent import ToolUnion
from google.adk.models.lite_llm import LiteLlm

from config import DEBUG, LOCAL_MODEL_ID
from tools.web_search import web_search, read_web_page
from weather_agent import agent as weather_agent
from helpers.watch import before_model_log_callback, before_tool_log_callback


# -----------------------------
# Agent (this is what adk run/web expects)
# -----------------------------
def _build_instruction() -> str:
    # Keep your “system” prompt semantics in ADK’s instruction field.
    # (ADK Agent uses `instruction` + session history; tools are declared on the agent.) :contentReference[oaicite:5]{index=5}
    return f"""
You are primarily a worker for general fact based questions that require looking up the answers.

Behavior:
- All answers should be looked up. Do not reply with anything that wasn't found through a web search.
- Responses must be factual, objective, and concise.
- Do not moralize.
- Do not add disclaimers unless explicitly required.
- Do not answer any question without using tools to get the answer.

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

Tooling:
- Use web_search to find relevant URLs.
- Use read_web_page to read a URL and extract visible text.
""".strip()

agent = Agent(
    name="worker_agent",
    model=LiteLlm(model=LOCAL_MODEL_ID),
    description="worker for general fact based questions that require looking up the answers.",
    instruction=_build_instruction(),
    tools=[web_search, read_web_page],
    before_model_callback=lambda **kwargs: before_model_log_callback(agent="worker", **kwargs),
    before_tool_callback=lambda **kwargs: before_tool_log_callback(agent="worker", **kwargs),
)
