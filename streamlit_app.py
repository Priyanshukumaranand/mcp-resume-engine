from __future__ import annotations

import os
from typing import List

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Hackathon Teammate Finder", layout="wide")
st.title("Hackathon Teammate Finder Demo")


def _post(path: str, payload: dict):
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


with st.expander("Add resume", expanded=True):
    with st.form("resume_form"):
        resume_id = st.text_input("ID", placeholder="student-001")
        name = st.text_input("Name")
        email = st.text_input("Email")
        role = st.text_input("Preferred Role")
        skills = st.text_input("Skills (comma separated)")
        experience = st.text_area("Experience")
        summary = st.text_area("Summary")
        submitted = st.form_submit_button("Store resume")

        if submitted:
            payload = {
                "id": resume_id,
                "name": name,
                "email": email,
                "role": role,
                "skills": [skill.strip() for skill in skills.split(",") if skill.strip()],
                "experience": experience,
                "summary": summary,
            }
            try:
                data = _post("/add_resume", payload)
                st.success(f"Stored resume {data['id']}")
            except requests.HTTPError as exc:
                st.error(f"Failed: {exc.response.text}")

with st.expander("Find a teammate", expanded=True):
    with st.form("recommend_form"):
        role = st.text_input("Desired Role", placeholder="Backend Engineer")
        skills = st.text_input("Important Skills (comma separated)")
        summary = st.text_area("Problem / Project Context")
        top_k = st.slider("Candidates to consider", 1, 20, 5)
        submitted = st.form_submit_button("Recommend teammate")

        if submitted:
            payload = {
                "role": role,
                "skills": [skill.strip() for skill in skills.split(",") if skill.strip()],
                "summary": summary,
                "top_k": top_k,
            }
            try:
                result = _post("/recommend", payload)
                best = result["best_match"]
                st.subheader("Best candidate")
                st.write(best["candidate"])
                st.write({
                    "match_score": best["match_score"],
                    "matching_skills": best["matching_skills"],
                    "explanation": best["explanation"],
                })
            except requests.HTTPError as exc:
                st.error(f"Failed: {exc.response.text}")

with st.expander("Stored resumes"):
    try:
        resumes = _get("/resumes").get("resumes", [])
        st.write(resumes if resumes else "No resumes yet")
    except requests.HTTPError as exc:
        st.error(f"Could not load resumes: {exc.response.text}")
