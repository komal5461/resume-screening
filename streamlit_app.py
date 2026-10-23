
import streamlit as st
import pandas as pd
from io import BytesIO
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Screening", page_icon="📄")

st.title("Resume Screening Using Machine Learning")
st.write("Upload resumes and compare them with job requirements.")

job = st.text_area(
    "Enter the job description",
    placeholder="Python, SQL, Machine Learning, Data Analysis..."
)

uploads = st.file_uploader(
    "Upload resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Screen Resumes"):
    if not job.strip():
        st.error("Please enter a job description.")
    elif not uploads:
        st.error("Please upload at least one resume.")
    else:
        names, texts = [], []

        for file in uploads:
            try:
                reader = PdfReader(BytesIO(file.getvalue()))
                text = " ".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )
                if text.strip():
                    names.append(file.name)
                    texts.append(text)
            except Exception:
                st.warning(f"Could not read {file.name}")

        if not texts:
            st.error("No readable resume text found.")
        else:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2)
            )
            matrix = vectorizer.fit_transform([job] + texts)
            scores = cosine_similarity(
                matrix[0:1], matrix[1:]
            ).flatten() * 100

            results = pd.DataFrame({
                "Candidate": names,
                "Match Score (%)": scores.round(2)
            }).sort_values(
                "Match Score (%)", ascending=False
            ).reset_index(drop=True)

            results.insert(0, "Rank", range(1, len(results) + 1))
            st.subheader("Candidate Rankings")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download Results CSV",
                results.to_csv(index=False).encode("utf-8"),
                "resume_screening_results.csv",
                "text/csv"
            )

            st.caption(
                "Scores measure text similarity, not hiring probability. "
                "Review candidates manually."
            )
