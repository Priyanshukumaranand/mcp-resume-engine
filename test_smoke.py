import json
import os
import requests

API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

sample = {
    "id": "test-1",
    "name": "Alice Example",
    "email": "alice@example.com",
    "role": "",
    "skills": [],
    "experience": "Built backend microservices and REST APIs using Python, FastAPI, and PostgreSQL.",
    "summary": "Backend engineer experienced with APIs and data pipelines."
}

try:
    print("POST /add_resume ->", requests.post(f"{API}/add_resume", json=sample, timeout=60).json())
except Exception as e:
    print("add_resume error:", e)

try:
    print("GET /resumes ->")
    r = requests.get(f"{API}/resumes", timeout=30)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print("resumes error:", e)

recommend_payload = {
    "role": "Backend Engineer",
    "skills": ["python", "fastapi"],
    "summary": "Looking for backend expertise to build APIs",
    "top_k": 5
}
try:
    print("POST /recommend ->")
    r = requests.post(f"{API}/recommend", json=recommend_payload, timeout=60)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print("recommend error:", e)

qa_payload = {"question": "Who has Python backend experience?", "top_k": 3}
try:
    print("POST /qa ->")
    r = requests.post(f"{API}/qa", json=qa_payload, timeout=60)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print("qa error:", e)
