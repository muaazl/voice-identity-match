"""
VoiceMimic Hugging Face Spaces Entrypoint.
Directly serves the FastAPI application and static frontend on port 7860.
Zero-Gradio dependencies to avoid huggingface_hub version conflicts.
"""

import os
import uvicorn
from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
