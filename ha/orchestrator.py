from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types  # Content/Part containers

import config
from config import DEBUG, LOCAL_MODEL_ID
from helpers.watch import before_model_log_callback, before_tool_log_callback
from weather_agent import agent as weather_agent
from worker_agent import agent as worker_agent


# -----------------------------
# Agent (this is what adk run/web expects)
# -----------------------------
def _build_instruction() -> str:
    # Keep your “system” prompt semantics in ADK’s instruction field.
    # (ADK Agent uses `instruction` + session history; tools are declared on the agent.) :contentReference[oaicite:5]{index=5}
    return f"""
You are a personal and home assistant running locally in a private setting for adults.
You decide which specialized agent should handle the user request.
Transfer control when appropriate.

Behavior:
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
""".strip()


agent = Agent(
    name="root_agent",
    model=LiteLlm(model=LOCAL_MODEL_ID),
    description="Personal/home assistant with web search and page reading tools.",
    instruction=_build_instruction(),
    before_model_callback=lambda **kwargs: before_model_log_callback(agent="orch", **kwargs),
    before_tool_callback=lambda **kwargs: before_tool_log_callback(agent="orch", **kwargs),
    sub_agents=[worker_agent, weather_agent]
)

# -----------------------------
# Standalone runner (optional) — mirrors your script-style invocation
# -----------------------------

@dataclass(frozen=True)
class RunIds:
    app_name: str = "local_home_assistant"
    user_id: str = "user_1"
    session_id: str = "session_001"

async def run_once(user_text: str, ids: RunIds = RunIds()) -> str:
    """
    Executes one turn and returns the final spoken text.
    """
    # Session + runner are the ADK equivalent of your LangGraph graph loop. :contentReference[oaicite:7]{index=7}
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=ids.app_name,
        user_id=ids.user_id,
        session_id=ids.session_id,
    )
    runner = Runner(agent=agent, app_name=ids.app_name, session_service=session_service)

    # User message content container :contentReference[oaicite:8]{index=8}
    new_message = types.Content(role="user", parts=[types.Part(text=user_text)])

    final_text = ""
    async for event in runner.run_async(
        user_id=ids.user_id,
        session_id=ids.session_id,
        new_message=new_message,
    ):
        # If you want to see everything ADK emits, uncomment:
        # print(f"[Event] author={event.author} final={event.is_final_response()} type={type(event).__name__}")

        if event.is_final_response():
            if event.content and event.content.parts and event.content.parts[0].text:
                final_text = event.content.parts[0].text
            break

    if final_text:
        print(final_text)
        #tts(final_text)

    return final_text


async def run_interactive(ids: RunIds = RunIds()) -> str:
    """
    Executes one turn and returns the final spoken text.
    """
    # Session + runner are the ADK equivalent of your LangGraph graph loop. :contentReference[oaicite:7]{index=7}
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=ids.app_name,
        user_id=ids.user_id,
        session_id=ids.session_id,
    )
    runner = Runner(agent=agent, app_name=ids.app_name, session_service=session_service)

    while True:
        user_text = input(">: ")

        # User message content container :contentReference[oaicite:8]{index=8}
        new_message = types.Content(role="user", parts=[types.Part(text=user_text)])

        final_text = ""
        async for event in runner.run_async(
            user_id=ids.user_id,
            session_id=ids.session_id,
            new_message=new_message,
        ):
            # If you want to see everything ADK emits, uncomment:
            # print(f"[Event] author={event.author} final={event.is_final_response()} type={type(event).__name__}")

            if event.is_final_response():
                if event.content and event.content.parts and event.content.parts[0].text:
                    final_text = event.content.parts[0].text
                break

        if final_text:
            print(final_text)


