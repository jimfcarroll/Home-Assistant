# Local OpenAI-compatible server (LiteLLM "openai/..." provider route).
# LiteLLM typically reads these env vars:
# - OPENAI_API_KEY
# - OPENAI_API_BASE (or OPENAI_BASE_URL in some setups)
#
# Keep "openai/current" aligned with your local model name.
LOCAL_MODEL_ID = "openai/current"
LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"
LOCAL_API_KEY = "not-needed"
CRAWLER_URL = "http://127.0.0.1:3000/crawl"

DEBUG = False

from google.adk.features import FeatureName, override_feature_enabled
import os

# Explicitly disable Progressive SSE Streaming
override_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING, False)

# Configure LiteLLM/OpenAI-compatible routing for your local server.
os.environ.setdefault("OPENAI_API_KEY", LOCAL_API_KEY)
os.environ.setdefault("OPENAI_API_BASE", LOCAL_BASE_URL)


