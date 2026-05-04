"""HF auth status check."""
import os
from huggingface_hub import whoami

print("HF_TOKEN env set:", bool(os.getenv("HF_TOKEN")))
try:
    info = whoami()
    print("Logged in as:", info.get("name", "?"))
except Exception as e:
    print(f"Not authenticated ({type(e).__name__}): {e}")
