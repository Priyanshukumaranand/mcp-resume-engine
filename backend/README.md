# Resume Processing Engine

This package provides a modular resume ingestion and QA system.

## Structure

```
backend/
├── core/                   # Core processing modules
│   ├── __init__.py
│   ├── text_processor.py   # Text cleaning and preprocessing
│   ├── semantic_chunker.py # NLP-based semantic chunking
│   └── section_detector.py # Heuristic + semantic section detection
├── embeddings/             # Embedding and vector operations
│   ├── __init__.py
│   ├── embedder.py         # Document/query embeddings
│   └── vectorstore.py      # ChromaDB integration
├── retrieval/              # Search and retrieval
│   ├── __init__.py
│   ├── reranker.py         # Cross-encoder reranking
│   └── verifier.py         # LLM verification
├── llm/                    # LLM integrations
│   ├── __init__.py
│   └── gemini.py           # Gemini API wrapper
├── models.py               # Pydantic models
├── graph.py                # LangGraph pipelines
└── main.py                 # FastAPI application
```
