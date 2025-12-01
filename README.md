# Makerspace RAG - Høgskolen i Østfold

A local Retrieval-Augmented Generation (RAG) system providing AI-powered assistance for the Makerspace at Høgskolen i Østfold. Get instant answers about 3D printing, laser cutting, CNC, electronics, and all maker-related topics.

## Features

- **Web Interface** - Clean, professional chat interface at localhost:5000
- **Local LLM** - Runs entirely on your machine using Ollama
- **645+ Knowledge Chunks** - Comprehensive makerspace documentation
- **Semantic Search** - Finds relevant information based on meaning, not just keywords

## Topics Covered

| Category | Topics |
|----------|--------|
| **3D Printing (FDM)** | Prusa MK4S, Prusa Mini, Input Shaper, PLA/PETG/ABS/TPU settings, infill patterns, troubleshooting |
| **3D Printing (Resin)** | SLA/MSLA operation, resin types, supports, post-processing, safety |
| **Laser Cutting** | Glowforge, materials, settings, safety, design preparation |
| **CNC Milling** | Basics, G-code, feeds/speeds, tooling, CAM software |
| **Electronics** | Arduino, Raspberry Pi (1-5), Pico, GPIO, sensors, circuits |
| **Vinyl Cutting** | Operation, materials, weeding, transfer tape |
| **PCB Design** | KiCad, schematics, layout, fabrication |
| **3D Scanning** | Photogrammetry, mesh repair, scan-to-print |
| **3D Modeling** | Fusion 360, Tinkercad, Blender, file formats |
| **Casting & Molding** | Silicone molds, resin casting, two-part molds |
| **Textiles** | Sewing, embroidery, e-textiles |
| **Finishing** | Painting, wood finishing, post-processing |
| **Safety** | Workshop safety, tool usage, material handling |
| **Sustainability** | Recycling, waste reduction, eco-friendly materials |

## Requirements

- Python 3.10+
- Ollama
- 8GB+ RAM recommended

## Quick Start

### 1. Install Python Dependencies

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

### 4. Run the Web Interface

```bash
python web_app.py
```

Then open **http://localhost:5000** in your browser.

### Alternative: Terminal Interface

```bash
python simple_rag.py
```

## File Structure

```
makerspace-rag/
├── web_app.py         # Web interface (Flask)
├── simple_rag.py      # Terminal interface
├── upload.py          # Document upload GUI
├── vault.txt          # Knowledge base
├── templates/
│   └── index.html     # Web interface template
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Adding Knowledge

### Option 1: Upload Interface
```bash
python upload.py
```
Opens a GUI to upload PDF, TXT, or JSON files to the knowledge base.

### Option 2: Direct Edit
Add content directly to `vault.txt` - one knowledge chunk per line.

## Example Questions

- "What temperature should I use for PETG on the Prusa MK4S?"
- "How do I safely cut acrylic with the laser cutter?"
- "What's the difference between gyroid and honeycomb infill?"
- "How do I connect an LED to Arduino?"
- "What G-code commands do I need for CNC homing?"
- "How do I set up a two-part silicone mold?"

## Technology

- **LLM**: Llama 3 via Ollama
- **Embeddings**: mxbai-embed-large (1024 dimensions)
- **Vector Search**: PyTorch cosine similarity
- **Web Framework**: Flask
- **Method**: RAG (Retrieval-Augmented Generation)

## How It Works

1. Your question is converted to a vector embedding
2. The system finds the most relevant knowledge chunks using semantic similarity
3. Retrieved context is sent to the LLM along with your question
4. The LLM generates a helpful, contextual response

## About

Developed for the Makerspace at Høgskolen i Østfold to provide students and staff with quick access to maker-related knowledge and guidance.
