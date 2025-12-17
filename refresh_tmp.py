from pathlib import Path
from backend.embedder import ResumeEmbedder
from backend.llm import ResumeLLM
from backend.vectorstore import ResumeVectorStore
from backend.graph import build_refresh_resume_graph

store = ResumeVectorStore(persist_directory=str(Path('chroma_storage')))
embedder = ResumeEmbedder()
llm = ResumeLLM(model_name="gemini-2.5-flash")
graph = build_refresh_resume_graph(llm, embedder, store)

resumes = store.get_all_resumes()
print(f"Resumes to refresh: {len(resumes)}")
results = []
for r in resumes:
    try:
        graph.invoke({'resume': r})
        results.append((r.id, 'updated'))
    except Exception as exc:
        results.append((r.id, f'error: {exc}'))

for rid, status in results:
    print(rid, status)
