# ViRAG-Agent

Visual-RAG agent system for industrial vehicle ADAS installation and maintenance manuals.

ViRAG-Agent turns one or many PDF manuals into page-level visual evidence, retrieves the most relevant pages with CLIP/ChromaDB plus hybrid reranking, and asks a vision-language model to answer questions from the original page images.

## Overview

This project was built for visually dense technical documents such as ADAS installation manuals, wiring diagrams, harness color tables, interface labels, parameter tables, and installation procedures. Instead of relying only on extracted PDF text, it renders every PDF page into high-resolution images and keeps the page image as the primary evidence source.

Typical questions include:

- Wiring, connector labels, terminal definitions, and harness color meanings
- Parameter ranges, model specifications, quantities, and numeric facts
- Installation steps, spatial relationships, diagrams, and tables
- Small text, knobs, terminals, labels, and other OCR/visual details
- Source-grounded answers with page references for manual verification

## Key Features

- **Multi-PDF ingestion**: place multiple `.pdf` files under `data/manual/`; the build pipeline indexes all of them together.
- **Visual-first retrieval**: PDF pages are rendered with PyMuPDF and embedded with CLIP.
- **Hybrid retrieval**: combines dense visual retrieval, BM25 text signals, and reciprocal rank fusion.
- **Reranking**: uses a BGE reranker to refine candidate pages before generation.
- **Evidence-aware VQA agent**: sends retrieved page images to Qwen2-VL for grounded answers.
- **Local or API VLM backend**: supports local Qwen2-VL inference or DashScope Qwen API.
- **Smart zoom tool**: crops and enlarges local page regions for detailed visual inspection.
- **Evaluation pipeline**: includes benchmark running, rule-first judging, and optional LLM fallback.

## Architecture

```text
PDF manual(s)
  -> PyMuPDF page rendering
  -> CLIP image/text encoding
  -> ChromaDB visual index
  -> HybridRetriever (dense + BM25 + RRF)
  -> BGE Reranker
  -> EvidenceAwareVQAAgent
  -> Qwen2-VL answer generation
  -> Answer + source pages
```

## Repository Structure

```text
agent/
  react_agent.py              # Simple and evidence-aware VQA agents
  memory.py                   # Conversation memory
apps/
  web_demo.py                 # Gradio web interface
dataset/
  pdf_processor.py            # PDF validation, rendering, and page metadata
evaluation/
  benchmark.json              # Benchmark samples
  evaluator.py                # Rule-first + optional LLM fallback evaluator
  run_benchmark.py            # Benchmark runner
llm/
  client_factory.py           # Selects local or API VLM backend
  qwen2_vl_client.py          # Local Qwen2-VL wrapper
  qwen_api_client.py          # DashScope/OpenAI-compatible API wrapper
  prompts.py                  # Prompt templates
model/
  clip_encoder.py             # CLIP encoder
retrieval/
  retriever.py                # ChromaDB visual retrieval
  hybrid_retriever.py         # ChromaDB + BM25 + RRF
  reranker.py                 # BGE reranker
scripts/
  build_vector_db.py          # Builds the visual vector database
  debug_single_image_vlm.py   # Single-image VLM debugging
tools/
  smart_zoom_tool.py          # Crop/zoom helper
config.py                     # Paths, backend settings, model names
main.py                       # Main CLI entry point
```

## Installation

Python 3.9+ is recommended. A CUDA-capable GPU is recommended for local Qwen2-VL inference.

```bash
pip install -r requirements.txt
```

If you use the API backend, set your DashScope key:

```bash
export LLM_BACKEND=qwen_api
export DASHSCOPE_API_KEY=your_api_key
```

On Windows PowerShell:

```powershell
$env:LLM_BACKEND = "qwen_api"
$env:DASHSCOPE_API_KEY = "your_api_key"
```

For local inference, keep `LLM_BACKEND=local` and update the model paths in `config.py` if your local model cache differs from the defaults.

## Prepare Data

Create the manual directory and put one or more PDFs inside it:

```text
data/manual/
  manual_part_1.pdf
  manual_part_2.pdf
  wiring_appendix.pdf
```

The system treats all PDFs in this directory as one searchable manual corpus. During indexing, each page keeps metadata including source PDF name, original PDF path, page number, rendered image path, page size, rotation, and extracted page text.

Generated files are stored under:

```text
data/page_images/   # rendered page images
data/chroma_db/     # persistent ChromaDB index
```

## Build the Visual Index

```bash
python main.py build
```

Or run the build script directly:

```bash
python scripts/build_vector_db.py
```

To rebuild from scratch:

```bash
python scripts/build_vector_db.py --reset
```

The build pipeline scans every `.pdf` file in `data/manual/`, renders pages at 300 DPI, encodes page images with CLIP, and stores vectors plus metadata in ChromaDB.

## Run

Start the Gradio web interface:

```bash
python main.py web
```

Start CLI mode:

```bash
python main.py cli
```

Start CLI mode with hybrid retrieval and reranking:

```bash
python main.py cli --hybrid
```

Show environment and component information:

```bash
python main.py info
```

## Example Workflow

```text
User question
  -> HybridRetriever finds candidate manual pages
  -> Reranker selects the strongest evidence page(s)
  -> EvidenceAwareVQAAgent sends page images to Qwen2-VL
  -> The answer is returned with source page metadata
```

Example question:

```text
How many wires are in the switch signal box, and what are their colors?
```

The answer includes the model response and source pages so the result can be checked against the original manual evidence.

## Single-Image VLM Debugging

Use this script to test whether a rendered page image can be processed by the VLM without running the retriever, reranker, or agent:

```bash
python scripts/debug_single_image_vlm.py --image "data/page_images/your_page.png"
```

This is useful for diagnosing image size, visual token count, GPU memory usage, and model input issues.

## Evaluation

Run the default benchmark:

```bash
python evaluation/run_benchmark.py --benchmark evaluation/benchmark.json
```

Write results to a specific directory:

```bash
python evaluation/run_benchmark.py --benchmark evaluation/benchmark.json --output_dir evaluation/results
```

Choose a judging mode:

```bash
python evaluation/run_benchmark.py --judge_mode local_llm
python evaluation/run_benchmark.py --judge_mode rule
```

The evaluator uses rule-first keypoint matching when available and can fall back to an LLM judge for more semantic cases. Result files include retrieved pages, reranked pages, final image paths, answer scores, hallucination flags, judge mode, and timing metrics.

## Current Benchmark Snapshot

The following numbers come from the current project benchmark and may change as the dataset, models, and prompts evolve.

| Area | Metric | Result |
| --- | --- | ---: |
| Retrieval | Top-1 Retrieval Accuracy | 23.38% |
| Retrieval | Retrieval Hit@K | 66.23% |
| Reranking | Rerank Top-1 Hit | 62.34% |
| Reranking | Rerank Hit@K | 62.34% |
| Context | Final Context Hit | 62.34% |
| Answer | Exact Accuracy | 80.52% |
| Answer | Weighted QA Accuracy | 81.17% |
| Answer | Average QA Score | 82.10% |
| Safety | Hallucination Rate | 1.30% |
| Stability | Judge Conflict Rate | 0.00% |

## Configuration Notes

Important settings live in `config.py`:

- `LLM_BACKEND`: `local` or `qwen_api`
- `MANUAL_DIR`: source PDF directory, default `data/manual`
- `PAGE_IMAGES_DIR`: rendered page image directory
- `CHROMA_DB_DIR`: persistent vector database directory
- `PDF_RENDER_DPI`: page rendering DPI
- `QWEN_API_MODEL`: API model name for the `qwen_api` backend
- `QWEN_VL_MODEL_NAME`: local Qwen2-VL path or model name

## Use Cases

- ADAS installation guidance
- Industrial vehicle maintenance Q&A
- Wiring diagram and connector lookup
- Harness color and interface label interpretation
- Parameter table, specification table, and installation procedure understanding
- Visual document evidence tracing and automated evaluation

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
