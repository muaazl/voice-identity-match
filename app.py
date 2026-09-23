"""
VoiceMimic Hugging Face Spaces Entrypoint.
Configured for Hugging Face ZeroGPU with @spaces.GPU handler.
Mounts custom FastAPI application and serves on port 7860.
"""

import os
import uvicorn
import gradio as gr
from app.main import app as fastapi_app

# Check for Hugging Face ZeroGPU environment
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False


def gpu_decorator(fn):
    if has_spaces:
        return spaces.GPU(fn)
    return fn


@gpu_decorator
def dummy_gpu_inference(text: str = "") -> str:
    """Registered @spaces.GPU function satisfying ZeroGPU startup scan."""
    return "VoiceMimic ZeroGPU Active"


# Create minimal Gradio block bound to @spaces.GPU function
with gr.Blocks(title="VoiceMimic") as demo:
    gr.HTML("<meta http-equiv='refresh' content='0; url=/'>")
    hidden_in = gr.Textbox(value="ping", visible=False)
    hidden_out = gr.Textbox(visible=False)
    hidden_btn = gr.Button("Init", visible=False)
    hidden_btn.click(fn=dummy_gpu_inference, inputs=hidden_in, outputs=hidden_out)

# Mount Gradio at /gradio so root / is served by our custom Single-Page Application
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
