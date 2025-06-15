import streamlit as st
import fitz
import os
import openai
import re
import tempfile
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI Job Recommender", layout="wide")
st.title("🤖 AI Job Recommendation System")

# --- Helpers ---
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return " ".join([page.get_text() for page in doc])

def keyword_gap(resume_text, jd_text):
    resume_tokens = set(resume_text.lower().split())
    jd_tokens = set(jd_text.lower().split())
    return list(jd_tokens - resume_tokens)

def highlight_keywords(resume_text, jd_text):
    resume_tokens = set(resume_text.lower().split())
    jd_tokens = set(jd_text.lower().split())
    return [(word, word in resume_tokens) for word in jd_tokens]

def format_check(text):
    checks = {
        "Education Section": bool(re.search(r"education", text, re.I)),
        "Contact Info": bool(re.search(r"(\d{10}|@|gmail|phone|email)", text, re.I)),
        "Projects/Experience": bool(re.search(r"(project|experience|work)", text, re.I)),
    }
    return checks

def generate_feedback(resume, jd):
    try:
        prompt = f"""You are a helpful assistant reviewing a resume for a job. Based on the job description below, suggest 3 specific improvements:
\nResume:\n{resume}\n\nJob Description:\n{jd}"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting feedback: {e}"

def create_pdf_report(resume_name, score, missing_keywords, feedback):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Resume Match Report: {resume_name}", ln=True)
    pdf.cell(200, 10, txt=f"Match Score: {score:.2f}", ln=True)
    pdf.cell(200, 10, txt="\nMissing Keywords:", ln=True)
    for kw in missing_keywords:
        pdf.cell(200, 10, txt=f"- {kw}", ln=True)
    pdf.cell(200, 10, txt="\nAI Suggestions:", ln=True)
    for line in feedback.split('\n'):
        pdf.cell(200, 10, txt=line.strip(), ln=True)
    temp_path = tempfile.mktemp(suffix=".pdf")
    pdf.output(temp_path)
    return temp_path

# --- UI Elements ---
api_input = st.text_input("🔑 Enter OpenAI API Key (or set OPENAI_API_KEY in .env)", type="password")
if api_input:
    openai.api_key = api_input

uploaded_files = st.file_uploader("📄 Upload 1–3 Resume PDFs", type=["pdf"], accept_multiple_files=True)
jd_input = st.text_area("📝 Paste Job Description", height=300)

if uploaded_files and jd_input:
    jd_text = jd_input.strip()
    for uploaded_file in uploaded_files:
        st.subheader(f"📌 Analyzing: {uploaded_file.name}")
        resume_text = extract_text_from_pdf(uploaded_file)

        # Similarity Match
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        st.metric("🔍 Similarity Score", f"{score*100:.2f}%")

        # Keyword Gap
        missing = keyword_gap(resume_text, jd_text)
        if missing:
            st.markdown("**❌ Missing Keywords:**")
            st.markdown(", ".join([f"`{kw}`" for kw in missing]))
        else:
            st.success("No major keywords missing!")

        # Visual Keyword Match
        st.markdown("**✅ Keyword Match Summary:**")
        matched_keywords = highlight_keywords(resume_text, jd_text)
        vis = [f"✔️ `{kw}`" if present else f"❌ `{kw}`" for kw, present in matched_keywords[:50]]
        st.markdown(" ".join(vis))

        # Format Check
        st.markdown("**⚠️ Resume Format Check:**")
        for item, ok in format_check(resume_text).items():
            st.write(f"{'✅' if ok else '❌'} {item}")

        # GPT Feedback
        with st.spinner("💡 Generating resume improvement tips..."):
            feedback = generate_feedback(resume_text, jd_text)
            st.markdown("**🧠 GPT Resume Suggestions:**")
            st.text(feedback)

        # Export PDF
        if st.button(f"📄 Export Report for {uploaded_file.name}"):
            pdf_path = create_pdf_report(uploaded_file.name, score, missing, feedback)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download PDF Report", f, file_name="resume_report.pdf")

    # Skill Match Plot (Optional)
    st.subheader("📊 Resume Match Score Comparison")
    scores = []
    names = []
    for uploaded_file in uploaded_files:
        resume_text = extract_text_from_pdf(uploaded_file)
        tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform([resume_text, jd_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        names.append(uploaded_file.name)
        scores.append(sim * 100)
    fig, ax = plt.subplots()
    ax.barh(names, scores, color='skyblue')
    ax.set_xlabel("Match Score %")
    st.pyplot(fig)
