# Resume RAG API

FastAPI backend that ingests resumes from PDFs, chunks them, stores embeddings in ChromaDB, and answers free-form questions over the stored corpus. Swagger UI is available at `/docs` and `/redoc` once the server is running.

## Features
- FastAPI endpoints for health check, PDF ingest, listing, and retrieval QA.
- ChromaDB persistent vector store (`chroma_storage/`) that keeps embeddings + metadata locally.
- Remote Hugging Face feature-extraction keeps the image lean; Gemini answers questions using retrieved chunks.
- LangGraph orchestrates resume ingestion (PDF text → chunk → embed → Chroma) for clearer, testable flow.
- Gemini helper extracts name/role/skills/summary/experience from the PDF text automatically—no manual fields needed.

## Setup
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

### Configure environment variables
Create a `.env` file (the backend loads it automatically) with your keys and CORS settings:

```
GEMINI_API_KEY=your_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token
# Optional: override Gemini model
# GEMINI_MODEL_NAME=gemini-1.5-pro
# Optional: comma-separated list of allowed origins for the BranchConnect frontend (use * to allow all)
ALLOWED_ORIGINS=https://ce-bootcamp.vercel.app,https://branchbase-backend.azurewebsites.net,http://localhost:3000,http://127.0.0.1:3000
```

By default the backend uses `gemini-1.5-flash`; update `GEMINI_MODEL_NAME` if you have access to another Gemini 3 series model.

## Running the FastAPI server
```bash
uvicorn backend.main:app --reload
```
- The API listens on `http://127.0.0.1:8000` by default.
- `POST /ingest_pdf` accepts a PDF, extracts fields automatically, chunks text, embeds, and stores it.
- `GET /resumes` lists everything currently in the `resumes` collection.
- `DELETE /resumes/{resume_id}` removes a stored resume (Streamlit exposes a delete button per card).
- `POST /qa` embeds the question, retrieves the most relevant chunks, and answers using only that context.

## Run with Docker
```bash
docker build -t mcp-resume-engine .
docker run -p 8000:8000 --env-file .env -e ALLOWED_ORIGINS=https://branchconnect.example.com mcp-resume-engine
```

## Deploy to Azure Container Apps (CLI, Docker Hub image)
1) Build and push your image (replace `<dockerhub-username>`):
```bash
docker build -t docker.io/<dockerhub-username>/mcp-resume-engine:latest .
docker push docker.io/<dockerhub-username>/mcp-resume-engine:latest
```
2) Provision and deploy (replace names/locations as needed):
```bash
az group create --name mcp-resume-rg --location eastus
az containerapp env create --name mcp-resume-env --resource-group mcp-resume-rg --location eastus
az containerapp create \
	--name mcp-resume-api \
	--resource-group mcp-resume-rg \
	--environment mcp-resume-env \
	--image docker.io/<dockerhub-username>/mcp-resume-engine:latest \
	--target-port 8000 --ingress external \
	--env-vars GEMINI_API_KEY=your_key_here ALLOWED_ORIGINS=https://branchconnect.example.com
```
3) Get the public URL:
```bash
az containerapp show --name mcp-resume-api --resource-group mcp-resume-rg --query "properties.configuration.ingress.fqdn" -o tsv
```
Use that FQDN as `API_BASE_URL` inside the BranchConnect placement UI.

## API reference
- Interactive docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Key endpoints:
	- `GET /health`
	- `POST /ingest_pdf`
	- `GET /resumes`
	- `DELETE /resumes/{resume_id}`
	- `POST /qa` (retrieves chunks and answers from stored resumes)

## How ChromaDB fits in
1. The API initializes a persistent ChromaDB client pointing at `./chroma_storage` and a collection named `resumes`.
2. When `/ingest_pdf` is called, the PDF text is extracted, chunked, embedded remotely, and upserted with resume metadata inferred by Gemini.
3. `/qa` embeds the incoming question, retrieves the most similar chunks, builds a context string, and asks Gemini to respond using only that context.

By persisting embeddings + metadata locally, restarting the server or Streamlit app preserves the stored resumes without any external databases.
# mcp-resume-engine
