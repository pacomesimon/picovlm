---
title: VibeLabel
emoji: 📚
colorFrom: pink
colorTo: blue
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: false
---

# VibeLabel

VibeLabel is an application designed for object detection.

## Getting Started

### Run with Docker

To run the application locally using Docker, execute the following command in your terminal:

```bash
docker run -it -p 7860:7860 --platform=linux/amd64 \
	registry.hf.space/pacomesimon-vibelabel:latest python app.py
```

Once the container is started, the application will be accessible at `http://localhost:7860`.

### Run with Python

Running locally is potentially dangerous. Make sure to review this Space code before proceeding.

```bash
# Clone repository
git clone https://huggingface.co/spaces/pacomesimon/vibeLabel
cd vibeLabel

# Create and activate Python environment
python -m venv env
source env/bin/activate

# Install dependencies and run
pip install -r requirements.txt
python app.py
```
