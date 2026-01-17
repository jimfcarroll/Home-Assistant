# Insert a tiny sleep to allow the warnings module to settle before the heavy ADK import
print ("Hello")

# agent.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Optional

from ddgs import DDGS

from google.adk.agents import Agent
from google.adk.agents.llm_agent import ToolUnion
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types  # Content/Part containers

from tts import tts

import patches  # keep your side effects if required
from read_web import _post_crawl, _shape_pages_for_llm


# -----------------------------
# Config
# -----------------------------

DEBUG = False

# Local OpenAI-compatible server (LiteLLM "openai/..." provider route).
# LiteLLM typically reads these env vars:
# - OPENAI_API_KEY
# - OPENAI_API_BASE (or OPENAI_BASE_URL in some setups)
#
# Keep "openai/current" aligned with your local model name.
LOCAL_MODEL_ID = "openai/current"
LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"
LOCAL_API_KEY = "not-needed"

from google.adk.features import FeatureName, override_feature_enabled

# Explicitly disable Progressive SSE Streaming
override_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING, False)


# -----------------------------
# Tools (ADK uses normal Python callables; docstrings matter)
# -----------------------------

def web_search(query: str) -> str:
    """
    Search the internet for current facts and information.

    Guidance:
    - Format your query starting with the intent first, then the context.
    - You will get a list of URLs back.
    - Use the URLs to read web pages with `read_web_page`.
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)

    ret = "".join(f"\n{r['href']}" for r in results)

    if DEBUG:
        print("\n======== web search ==========")
        print(f"query:{query}")
        print("========")
        print(ret)
        print("==============================", flush=True)

    return ret


def read_web_page(url: str) -> str:
    """
    Render a web page locally and return LLM-suitable visible text.
    """
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

    if DEBUG:
        print("\n======== page read ==========")
        print(res[:5000])  # preview only
        print("==============================", flush=True)

    return res


# -----------------------------
# Optional guardrails / logging callbacks (ADK-native)
# -----------------------------
# ADK agents accept before_model_callback and before_tool_callback :contentReference[oaicite:4]{index=4}

def before_model_log_callback(**kwargs) -> None:
    """
    Runs right before the model is called.

    You can:
    - inspect / modify the model_input
    - block by returning an escalation/action (see ADK docs/tutorial patterns)
    """
    if DEBUG:
        try:
            print("\n======== before_model_callback ==========")
            print(kwargs)
            print("========================================", flush=True)
        except Exception:
            pass

def before_tool_log_callback(tool : ToolUnion, args : dict, tool_context, **kwargs) -> None:
    """
    Runs right before a tool is invoked.

    You can:
    - inspect / modify tool args
    - block tool calls by returning an error-shaped tool result (tutorial pattern)
    """
    if DEBUG:
        try:
            print("\n======== before_tool_callback ==========")
            print(kwargs)
            print("=======================================", flush=True)
        except Exception:
            pass


# -----------------------------
# Agent (this is what adk run/web expects)
# -----------------------------
def _build_instruction() -> str:
    # Keep your “system” prompt semantics in ADK’s instruction field.
    # (ADK Agent uses `instruction` + session history; tools are declared on the agent.) :contentReference[oaicite:5]{index=5}
    return f"""
You are a personal and home assistant running locally in a private setting for adults.

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

Tooling:
- Use web_search to find relevant URLs.
- Use read_web_page to read a URL and extract visible text.
""".strip()


# Model selection:
# - For Gemini you can pass a string (e.g., "gemini-2.0-flash")
# - For OpenAI/Anthropic/etc you wrap with LiteLlm(model="provider/name") :contentReference[oaicite:6]{index=6}
#
# For your local OpenAI-compatible server, LiteLLM generally obeys OPENAI_API_BASE and OPENAI_API_KEY.
local_model = LiteLlm(model=LOCAL_MODEL_ID)

root_agent = Agent(
    name="root_agent",
    model=local_model,
    description="Personal/home assistant with web search and page reading tools.",
    instruction=_build_instruction(),
    tools=[web_search, read_web_page],
    before_model_callback=before_model_log_callback,
    before_tool_callback=before_tool_log_callback,
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
    runner = Runner(agent=root_agent, app_name=ids.app_name, session_service=session_service)

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

def stt_vosk() -> str:
    import queue
    import sys
    import json
    import sounddevice as sd
    from vosk import Model, KaldiRecognizer

    """
    Blocking microphone capture.
    Speak one sentence and pause.
    Returns recognized text.
    """

    model = Model("stt/vosk-model-small-en-us-0.15")
    recognizer = KaldiRecognizer(model, 16000)

    audio_queue: queue.Queue[bytes] = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(bytes(indata))

    print("Listening...")

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    print(f"Heard: {text}")
                    return text


if __name__ == "__main__":
    import os
    import asyncio

    #text = stt_vosk()
    text = "Who is the current president and what year is it"

    # Configure LiteLLM/OpenAI-compatible routing for your local server.
    os.environ.setdefault("OPENAI_API_KEY", LOCAL_API_KEY)
    os.environ.setdefault("OPENAI_API_BASE", LOCAL_BASE_URL)

    # Mirrors your example query
    asyncio.run(run_once(text))
