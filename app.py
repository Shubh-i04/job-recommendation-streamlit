import streamlit as st
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

st.title("AI Job Recommendation System")

uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
jd_input = st.text_area("Paste Job Description(s), separated by '---'", height=300)

if uploaded_file and jd_input:
    resume_text = extract_text_from_pdf(uploaded_file)
    jd_blocks = jd_input.split('---')
    job_descriptions = [{"title": f"Job {i+1}", "description": jd.strip()} for i, jd in enumerate(jd_blocks)]

    all_texts = [resume_text] + [job["description"] for job in job_descriptions]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    for i, job in enumerate(job_descriptions):
        job["score"] = round(similarity_scores[i] * 100, 2)
    sorted_jobs = sorted(job_descriptions, key=lambda x: x['score'], reverse=True)

    st.subheader("Top Matching Jobs:")
    for job in sorted_jobs:
        st.markdown(f"**{job['title']}** - Match: {job['score']}%")
        st.markdown(f"`{job['description']}`")