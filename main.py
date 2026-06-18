import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Step 1: Resume PDF read pannuvom
resume_path = "resume.pdf"
with pdfplumber.open(resume_path) as pdf:
    resume_text = ""
    for page in pdf.pages:
        resume_text += page.extract_text()

# Step 2: Job description
job_description = """
We are looking for a Python Developer with experience in SQL, Data Analysis,
Pandas, Power BI, Machine Learning, and API integration. Knowledge of Flask
and Django is a plus. Strong problem solving and communication skills required.
"""

# Step 3: Match percentage calculate pannuvom
documents = [resume_text, job_description]
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(documents)
similarity = cosine_similarity(vectors[0:1], vectors[1:2])
match_percentage = round(similarity[0][0] * 100, 2)

# Step 4: Missing keywords find pannuvom
job_words = set(job_description.lower().split())
resume_words = set(resume_text.lower().split())
missing = job_words - resume_words
cleaned_missing = [re.sub(r'[^a-zA-Z]', '', w) for w in missing]
cleaned_missing = sorted(set([w for w in cleaned_missing if len(w) > 3]))

# Step 5: Nice ah output kaattuvom
print("=" * 50)
print("        AI RESUME ANALYZER - REPORT")
print("=" * 50)
print(f"\nMatch Score: {match_percentage}%")

bar_length = 30
filled = int(bar_length * match_percentage / 100)
bar = "█" * filled + "-" * (bar_length - filled)
print(f"[{bar}]")

if match_percentage >= 70:
    print("Status: Strong Match ✅")
elif match_percentage >= 40:
    print("Status: Moderate Match ⚠️")
else:
    print("Status: Needs Improvement ❌")

print("\nMissing Keywords (consider adding these):")
for word in cleaned_missing:
    print(f"  - {word}")

print("\n" + "=" * 50)