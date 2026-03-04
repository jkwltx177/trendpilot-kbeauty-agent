import os
from dotenv import load_dotenv
import sys
import types

load_dotenv(override=True)

# Chroma DB Paths
CHROMA_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "vector_db", "local_kb")
CHROMA_WEB_DIR = os.path.join(os.path.dirname(__file__), "vector_db", "web_cache")

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL_VISION = "gpt-4o"
LLM_MODEL_ROUTER = "gpt-4o-mini"
LLM_MODEL_STRATEGY = "gpt-4o"
LLM_MODEL_CREATIVE = "gpt-4o"

# Turn off Chroma telemetry
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CHROMA_LOG_LEVEL"] = "ERROR"

def _hard_disable_chroma_telemetry():
    dummy = types.SimpleNamespace(
        capture=lambda *a, **k: None,
        flush=lambda *a, **k: None,
        shutdown=lambda *a, **k: None,
        __version__="0.0",
    )
    sys.modules["posthog"] = dummy

_hard_disable_chroma_telemetry()
