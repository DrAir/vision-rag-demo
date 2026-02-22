# Vision RAG Demo (macOS Apple Silicon)

A local Vision RAG demo using Gradio, Byaldi (ColPali), and MLX-VLM (Qwen3-VL-30B-A3B).

## Features
- **PDF Upload**: Upload any PDF contract/document
- **Visual Retrieval**: Uses ColPali to find the most relevant page
- **Local VLM Generation**: Uses Qwen3-VL-30B-A3B on Apple Silicon via MLX

## Requirements
- macOS Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.11
- 24GB+ RAM recommended

## Setup
```bash
# Install poppler (for PDF conversion)
brew install poppler

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install gradio byaldi mlx-vlm pdf2image transformers
```

## Usage
```bash
source venv/bin/activate
python app.py
```
Then open http://localhost:7860 in your browser.

## Models Used
- **Retrieval**: `vidore/colpali-v1.2` (via Byaldi)
- **Generation**: `mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit` (via MLX-VLM)
