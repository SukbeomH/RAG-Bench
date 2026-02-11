import os
import sys

# Ensure rag_bench can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rag_bench

    print("rag_bench imported.")
except ImportError as e:
    print(f"Failed to import rag_bench: {e}")

api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print(f"OPENAI_API_KEY found: {api_key[:5]}...{api_key[-5:]}")
else:
    print("OPENAI_API_KEY not found in environment.")

# Check config.py side effects directly
from rag_bench import config

print(f"Project config loaded from: {config.PROJECT_ROOT}")
