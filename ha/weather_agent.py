# weather_service.py
from datetime import date, datetime

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from config import LOCAL_MODEL_ID
from helpers.watch import before_model_log_callback, before_tool_log_callback
from tools.weather import get_weather


def _build_instruction() -> str:
    # Keep your “system” prompt semantics in ADK’s instruction field.
    # (ADK Agent uses `instruction` + session history; tools are declared on the agent.) :contentReference[oaicite:5]{index=5}
    return f"""
You provide weather forecasts.

Behavior:
- All answers should be looked up using the tools. Do not reply with any information that didn't result from a tool call.

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
    name="weather_agent",
    model=LiteLlm(model=LOCAL_MODEL_ID),
    instruction=_build_instruction(),
    tools=[get_weather],
    before_model_callback=lambda **kwargs: before_model_log_callback(agent="weather_agent", **kwargs),
    before_tool_callback=lambda **kwargs: before_tool_log_callback(agent="weather_agent", **kwargs),
)

#session_service = InMemorySessionService()
#runner = Runner(agent=agent, app_name="weather_service", session_service=session_service)

# expose runner via A2A HTTP endpoint
