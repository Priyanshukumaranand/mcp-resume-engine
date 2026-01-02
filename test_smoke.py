import os
import pytest
from unittest.mock import MagicMock, patch

# Set dummy env vars before importing main to bypass dependency init checks
os.environ["GEMINI_API_KEY"] = "dummy_key"
os.environ["HUGGINGFACE_API_TOKEN"] = "dummy_token"

# Now import app
from backend.main import app
from backend.anonymizer import generate_anon_id, strip_pii, extract_email, UNKNOWN_VALUE
from backend.validator import ResponseValidator
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock out the heavy/external dependencies in backend.main"""
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
        mock_embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        mock_embedder.embed_many.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock Vector Store
        mock_vs.has_resumes.return_value = True
        mock_vs.get_all_resumes.return_value = []
        mock_vs.query.return_value = []
        mock_vs.query_by_skills.return_value = []
        
        yield


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_pdf_mock():
    """Test PDF ingestion with privacy handling."""
    with patch("backend.main._extract_pdf_text", return_value="John Doe john@example.com Phone: 555-1234"):
        files = {"file": ("resume.pdf", b"dummy content", "application/pdf")}
        response = client.post("/ingest_pdf", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert "id" in data
        assert "anon_id" in data
        assert "name" in data
        # Email should NOT be in response
        assert "email" not in data or "@" not in str(data.get("email", ""))


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
    
    # Check new response fields
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


def test_team_recommend():
    """Test team recommendation endpoint."""
    from backend.main import vector_store
    from backend.models import Resume
    
    # Create mock resumes with explicit skills
    mock_resume1 = Resume(
        id="1",
        anon_id="user1abc",
        name="Bob",
        email="bob@test.com",
        role="Backend Dev",
        skills=["python", "fastapi", "postgresql"],
        projects=["API Gateway"],
        experience="Backend development",
        summary="Python expert",
    )
    mock_resume2 = Resume(
        id="2",
        anon_id="user2def",
        name="Carol",
        email="carol@test.com",
        role="Frontend Dev",
        skills=["react", "typescript", "css"],
        projects=["Dashboard UI"],
        experience="Frontend development",
        summary="React specialist",
    )
    
    # Mock skill query results
    vector_store.query_by_skills.return_value = [
        (mock_resume1, ["python", "fastapi"], 0.9),
        (mock_resume2, ["react"], 0.8),
    ]
    
    payload = {
        "required_skills": ["python", "react", "fastapi"],
        "team_size": 3
    }
    response = client.post("/teams/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "candidates" in data
    assert "required_skills" in data
    assert "coverage_summary" in data
    
    # Check candidates have correct structure
    if data["candidates"]:
        candidate = data["candidates"][0]
        assert "anon_id" in candidate
        assert "name" in candidate  # Name should be visible
        assert "matched_skills" in candidate
        assert "reasoning" in candidate
        # Reasoning should reference explicit skills
        assert "explicitly" in candidate["reasoning"].lower() or "matched" in candidate["reasoning"].lower()


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
    validator = ResponseValidator()
    
    # High similarity = high confidence
    high_conf = validator.calculate_confidence([0.9, 0.85, 0.8], 3)
    assert high_conf > 0.7
    
    # Low similarity = low confidence
    low_conf = validator.calculate_confidence([0.3, 0.2, 0.1], 3)
    assert low_conf < 0.5
    
    # Empty = zero confidence
    zero_conf = validator.calculate_confidence([], 0)
    assert zero_conf == 0.0


def test_validation_fallback():
    """Test fallback responses."""
    validator = ResponseValidator()
    
    # Should return fallback for low confidence
    assert validator.should_use_fallback(0.3)
    assert not validator.should_use_fallback(0.9)


def test_no_email_in_qa_response():
    """Ensure email is never exposed in QA responses."""
    from backend.main import vector_store
    from backend.models import Resume
    
    mock_resume = Resume(
        id="1",
        anon_id="abc123",
        name="Test User",
        email="secret@example.com",  # This should NOT appear
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


# ==================== Strict Extraction Tests ====================

def test_strict_extraction_format():
    """Test that LLM extraction returns expected format."""
    from backend.llm import ResumeLLM, UNKNOWN_VALUE
    
    # This would require actual Gemini API, so we test the structure
    llm = ResumeLLM.__new__(ResumeLLM)
    
    # Test empty extraction
    empty_result = llm._empty_extraction()
    assert empty_result["name"] == UNKNOWN_VALUE
    assert empty_result["email"] == UNKNOWN_VALUE
    assert empty_result["skills"] == []


def test_unknown_value_constant():
    """Verify UNKNOWN_VALUE is used consistently."""
    assert UNKNOWN_VALUE == "not provided"
