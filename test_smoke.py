"""Smoke tests for resume processing API."""
import os
import pytest
from unittest.mock import MagicMock, patch

# Set dummy env vars before importing main
os.environ["GEMINI_API_KEY"] = "dummy_key"
os.environ["HUGGINGFACE_API_TOKEN"] = "dummy_token"

from backend.main import app
from backend.anonymizer import generate_anon_id, strip_pii, extract_email, UNKNOWN_VALUE
from backend.retrieval.verifier import ResponseVerifier
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock out the heavy/external dependencies."""
    with patch("backend.main.resume_llm") as mock_llm, \
         patch("backend.main.embedder") as mock_embedder, \
         patch("backend.main.vector_store") as mock_vs:
        
        # Mock LLM extraction with strict format
        mock_llm.extract_resume_fields.return_value = {
            "name": "Test User",
            "email": "test@example.com",
            "role": "Software Engineer",
            "skills": ["python", "testing"],
            "projects": ["API Development"],
            "education": "B.S. Computer Science",
            "summary": "Experienced developer",
            "experience": "5 years of experience"
        }
        mock_llm.answer_question.return_value = "Test User has Python experience."

        # Mock Embedder
        mock_embedder.embed_text.return_value = [0.1] * 768
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_embedder.embed_document.return_value = [0.1] * 768
        mock_embedder.embed_documents.return_value = [[0.1] * 768]
        mock_embedder.embed_many.return_value = [[0.1] * 768]
        
        # Mock Vector Store
        mock_vs.has_resumes.return_value = True
        mock_vs.get_all_resumes.return_value = []
        mock_vs.query.return_value = []
        mock_vs.query_by_skills.return_value = []
        
        yield


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_ingest_pdf_mock():
    """Test PDF ingestion with privacy handling."""
    with patch("backend.main._extract_pdf_text", return_value="John Doe john@example.com Phone: 555-1234"):
        files = {"file": ("resume.pdf", b"dummy content", "application/pdf")}
        response = client.post("/ingest_pdf", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processed"
        assert "id" in data
        assert "anon_id" in data
        assert "name" in data
        # Should include section info
        assert "sections_detected" in data
        assert "chunk_count" in data


def test_list_resumes():
    response = client.get("/resumes")
    assert response.status_code == 200
    assert "resumes" in response.json()


def test_qa_flow_with_confidence():
    """Test QA with confidence scoring."""
    from backend.main import vector_store
    from backend.models import Resume
    
    # Create mock resume
    mock_resume = Resume(
        id="1",
        anon_id="abc123",
        name="Alice",
        email="alice@test.com",
        role="Developer",
        skills=["python", "react"],
        projects=["Web App"],
        education="BS CS",
        experience="3 years",
        summary="Full stack developer",
    )
    
    # Mock return with section type
    mock_match = (mock_resume, "Python developer with React experience", 0.85, "skills")
    vector_store.query.return_value = [mock_match]
    
    payload = {"question": "Who knows Python?", "top_k": 3}
    response = client.post("/qa", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Check response fields
    assert "answer" in data
    assert "confidence_score" in data
    assert "source_sections" in data
    assert "is_fallback" in data
    assert len(data["sources"]) > 0
    
    # Verify name is visible but email is not
    source = data["sources"][0]
    assert "name" in source
    assert source["name"] == "Alice"
    assert "anon_id" in source


# ==================== Privacy Tests ====================

def test_anon_id_generation():
    """Same email should produce same anon_id."""
    email1 = "test@example.com"
    email2 = "test@example.com"
    email3 = "other@example.com"
    
    assert generate_anon_id(email1) == generate_anon_id(email2)
    assert generate_anon_id(email1) != generate_anon_id(email3)


def test_anon_id_format():
    """anon_id should be 16 hex characters."""
    anon_id = generate_anon_id("test@example.com")
    assert len(anon_id) == 16
    assert all(c in "0123456789abcdef" for c in anon_id)


def test_strip_pii():
    """PII should be stripped from text."""
    text = """
    John Doe
    Email: john.doe@example.com
    Phone: (555) 123-4567
    LinkedIn: linkedin.com/in/johndoe
    123 Main Street, City, CA 12345
    """
    
    cleaned = strip_pii(text)
    
    # Email should be redacted
    assert "john.doe@example.com" not in cleaned
    assert "[EMAIL REDACTED]" in cleaned
    
    # Phone should be redacted
    assert "(555) 123-4567" not in cleaned
    assert "[PHONE REDACTED]" in cleaned
    
    # LinkedIn should be redacted
    assert "linkedin.com/in/johndoe" not in cleaned
    
    # Name should be preserved (not stripped)
    assert "John Doe" in cleaned


def test_extract_email():
    """Email extraction from text."""
    text = "Contact me at john@example.com for details"
    email = extract_email(text)
    assert email == "john@example.com"
    
    # No email case
    assert extract_email("No email here") is None


# ==================== Validation Tests ====================

def test_confidence_calculation():
    """Test confidence score calculation."""
    verifier = ResponseVerifier()
    
    # High similarity = high confidence
    high_conf = verifier.calculate_confidence([0.9, 0.85, 0.8], 3)
    assert high_conf > 0.7
    
    # Low similarity = low confidence
    low_conf = verifier.calculate_confidence([0.3, 0.2, 0.1], 3)
    assert low_conf < 0.5
    
    # Empty = zero confidence
    zero_conf = verifier.calculate_confidence([], 0)
    assert zero_conf == 0.0


def test_validation_fallback():
    """Test fallback responses."""
    verifier = ResponseVerifier()
    
    # Should return fallback for low confidence
    assert verifier.should_use_fallback(0.3)
    assert not verifier.should_use_fallback(0.9)


def test_no_email_in_qa_response():
    """Ensure email is never exposed in QA responses."""
    from backend.main import vector_store
    from backend.models import Resume
    
    mock_resume = Resume(
        id="1",
        anon_id="abc123",
        name="Test User",
        email="secret@example.com",
        role="Developer",
        skills=["python"],
        projects=[],
        experience="Experience",
        summary="Summary",
    )
    
    mock_match = (mock_resume, "Python developer", 0.8, "skills")
    vector_store.query.return_value = [mock_match]
    
    response = client.post("/qa", json={"question": "Who knows Python?"})
    response_text = response.text
    
    # Email should never appear in response
    assert "secret@example.com" not in response_text
    assert "secret" not in response_text.lower()


# ==================== Semantic Chunking Tests ====================

def test_semantic_chunker():
    """Test semantic chunking."""
    from backend.core import SemanticChunker
    
    chunker = SemanticChunker()
    
    sample_text = """
    John Smith
    Software Engineer
    
    SKILLS
    Python, JavaScript, React, Node.js, PostgreSQL
    
    EXPERIENCE
    Senior Developer at TechCorp (2020-Present)
    Built scalable microservices architecture.
    
    EDUCATION
    B.S. Computer Science, MIT
    """
    
    chunks, sections = chunker.chunk_resume(sample_text)
    
    assert len(chunks) > 0
    assert len(sections) > 0
    
    # Check section types detected
    section_types = {s.section_type for s in sections}
    assert "skills" in section_types or "header" in section_types


def test_section_detector():
    """Test section detection."""
    from backend.core import SectionDetector
    
    detector = SectionDetector()
    
    text = """
    TECHNICAL SKILLS
    Python, Java, React
    
    WORK EXPERIENCE
    Developer at Company
    """
    
    sections = detector.detect_sections(text)
    assert len(sections) > 0


# ==================== Embedding Tests ====================

def test_embedder_dimensions():
    """Test embedder returns correct dimensions."""
    from backend.embeddings import ResumeEmbedder
    
    embedder = ResumeEmbedder.__new__(ResumeEmbedder)
    embedder.model_name = "BAAI/bge-base-en-v1.5"
    embedder._embedding_dim = 768
    
    # The dimension property should return 768
    assert embedder.embedding_dimension == 768


def test_unknown_value_constant():
    """Verify UNKNOWN_VALUE is used consistently."""
    assert UNKNOWN_VALUE == "not provided"
