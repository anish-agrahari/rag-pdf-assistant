import streamlit as st
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ---------------- CONFIG ----------------
genai.configure(api_key="AIzaSyDCLkMwGsECcHHXfagdE7matDryoey5zRQ")
model = genai.GenerativeModel("gemini-2.5-flash")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stTextInput > div > div > input {
    background-color: #1c1f26;
    color: white;
}
.stFileUploader {
    background-color: #1c1f26;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
# 📄 RAG PDF Assistant
### Ask intelligent questions from your documents
""")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    top_k = st.slider("Number of Chunks to Retrieve", 1, 5, 3)

with col2:
    question = st.text_input("Ask a question about your document")

# ---------------- PROCESS ----------------
if uploaded_file and question:

    with st.spinner("Processing document..."):
        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

        chunks = [text[i:i+500] for i in range(0, len(text), 500)]

        embeddings = embedding_model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        q_embedding = embedding_model.encode([question])
        q_embedding = np.array(q_embedding).astype("float32")

        distances, indices = index.search(q_embedding, top_k)
        context = "\n".join([chunks[i] for i in indices[0]])

        prompt = f"""
        Answer the question using the context below.
        If the answer is not in the context, say you don't know.

        Context:
        {context}

        Question:
        {question}
        """

        response = model.generate_content(prompt)

    st.success("Answer generated successfully!")

    st.markdown("## 💡 Answer")
    st.write(response.text)

    with st.expander("🔎 Retrieved Context"):
        st.write(context)