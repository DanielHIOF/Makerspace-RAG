# Makerspace RAG - Høgskolen i Østfold

A local Retrieval-Augmented Generation (RAG) system providing AI-powered assistance for the Makerspace at Høgskolen i Østfold. Get instant answers about 3D printing, laser cutting, CNC, electronics, and all maker-related topics.

## Features

- **Web Interface** - Clean chat interface at localhost:5000
- **Admin Panel** - Protected area for managing knowledge base
- **Local LLM** - Runs entirely on your machine using Ollama
- **Semantic Search** - Finds relevant information based on meaning

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

Download from: https://ollama.com/download

### 3. Pull Required Models

```bash
ollama pull llama3
ollama pull mxbai-embed-large
```

### 4. Run the Application

```bash
python app.py
```

