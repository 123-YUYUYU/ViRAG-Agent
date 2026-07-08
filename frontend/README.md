# ViRAG-Agent Frontend Prototype

This is a static frontend prototype for the ViRAG-Agent industrial Visual-RAG QA interface. It is designed as an actual web app surface rather than a README-style explanation page.

## Open

Open directly in a browser:

```text
frontend/index.html
```

## Interface Scope

- Workspace: conversation-first manual QA with retained context
- Session Memory: current topic, recent evidence, and compressed conversation summary
- Query Console: text query, visual query, and retrieval mode switches
- Evidence: candidate manual pages and evidence tags
- Workflow: Function Calling, CLIP, BM25, RRF, BGE Reranker, and Qwen3-VL answer generation
- Backend Panel: connected runtime status with Qwen3-VL / Qwen3-Max and ChromaDB visual index

This prototype is still pure frontend. The current interactions are mock UI behavior and do not call the real model, vector database, or backend services.
