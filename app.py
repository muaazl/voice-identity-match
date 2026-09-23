"""
VoiceMimic Hugging Face Spaces Gradio-compatible Entrypoint.
Mounts custom FastAPI application and serves on port 7860.
"""

import os
import uvicorn
import gradio as gr
from app.main import app as fastapi_app

# Create a minimal Gradio block to satisfy HF Space Gradio health checks
with gr.Blocks(title="VoiceMimic") as demo:
    gr.HTML("<meta http-equiv='refresh' content='0; url=/'>")

# Mount Gradio at /gradio so root / is served by our custom FastAPI Single-Page Application
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
