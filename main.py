import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.title("🚀 AI Resume Matcher")

# Job Description Input
job_desc = st.text_area("Job Description-a inga paste pannunga:")

# File Uploader
uploaded_file = st.file_uploader("Unga Resume (PDF)-a upload pannunga", type="pdf")

if uploaded_file and job_desc:
    # PDF extract panna
    with pdfplumber.open(uploaded_file) as pdf:
        resume_text = "".join([page.extract_text() for page in pdf.pages])
    
    # Analysis Logic
    documents = [resume_text, job_desc]
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    match_score = round(similarity[0][0] * 100, 2)
    
    # Results Display
    st.subheader(f"Match Score: {match_score}%")
    st.progress(match_score / 100)
    
    # Keyword analysis
    job_words = set(re.findall(r'\w+', job_desc.lower()))
    resume_words = set(re.findall(r'\w+', resume_text.lower()))
    missing = [w for w in job_words if w not in resume_words and len(w) > 3]
    
    if missing:
        st.write("Missing Keywords:", missing)
    else:
        st.success("All important keywords are present!")