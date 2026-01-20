import asyncio

import config
from orchestrator import run_once, run_interactive

if __name__ == "__main__":
    #text = stt_vosk()
    #text = "What's the current weather in Philadelphia"

    # Mirrors your example query
    asyncio.run(run_interactive())
