import os
from src.interface import create_demo

# Ensure dummy file exists for Gradio File component initial state
dummy_path = os.path.join("assets", "_")
if not os.path.exists(dummy_path):
    with open(dummy_path, "w") as f:
        f.write("")

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=True,
        debug=False
    )