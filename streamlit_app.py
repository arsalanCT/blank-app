import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import re

st.set_page_config(page_title="404 Redirect Suggestion Tool", layout="wide")
st.title("🔁 Smart 404 Redirect Suggestion Tool")

# Helper function to clean and extract path keywords
def extract_keywords(url):
    parsed = urlparse(url)
    path = parsed.path
    path = re.sub(r'[^a-zA-Z0-9/]', ' ', path)
    keywords = path.strip("/").replace("/", " ")
    return keywords.lower()

# Upload files
uploaded_404 = st.file_uploader("📄 Upload CSV of 404 URLs", type="csv")
uploaded_valid = st.file_uploader("📄 Upload CSV of Valid URLs (Optional but recommended)", type="csv")

if uploaded_404:
    df_404 = pd.read_csv(uploaded_404)
    urls_404 = df_404.iloc[:, 0].dropna().tolist()

    if uploaded_valid:
        df_valid = pd.read_csv(uploaded_valid)
        valid_urls = df_valid.iloc[:, 0].dropna().tolist()
    else:
        valid_urls = []

    # Process
    redirect_suggestions = []

    if valid_urls:
        # Build TF-IDF matrix
        docs = [extract_keywords(url) for url in valid_urls]
        tfidf = TfidfVectorizer().fit(docs)
        valid_vectors = tfidf.transform(docs)

        for url in urls_404:
            query = extract_keywords(url)
            query_vec = tfidf.transform([query])
            similarity = cosine_similarity(query_vec, valid_vectors).flatten()

            if similarity.max() > 0.3:
                best_match_index = similarity.argmax()
                suggestion = valid_urls[best_match_index]
            else:
                suggestion = "No good match found"
            
            redirect_suggestions.append({
                "404 URL": url,
                "Suggested Redirect": suggestion
            })

    else:
        # Fallback: basic trimming of URL path
        for url in urls_404:
            parsed = urlparse(url)
            path_parts = parsed.path.strip("/").split("/")
            suggestion = "/" + "/".join(path_parts[:-1]) if len(path_parts) > 1 else "/"
            redirect_suggestions.append({
                "404 URL": url,
                "Suggested Redirect": suggestion
            })

    st.success("✅ Suggestions generated!")
    st.dataframe(pd.DataFrame(redirect_suggestions))

    # Download button
    df_out = pd.DataFrame(redirect_suggestions)
    csv = df_out.to_csv(index=False)
    st.download_button("⬇️ Download Suggestions as CSV", data=csv, file_name="redirect_suggestions.csv", mime="text/csv")
