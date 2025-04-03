# ✅ UPDATED backend.py with Semantic Chunking + Fusion Retrieval + Corrective RAG

import os
import faiss
import pickle
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# OpenAI API setup
from openai import OpenAI
client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE)

# Local embedding model fallback
local_model = SentenceTransformer("bert-base-nli-mean-tokens")

# ---- Helper: Strip non-ASCII characters ----
def strip_non_ascii(text):
    return ''.join(char for char in text if ord(char) < 128)

# ---- 1. Extract & Chunk Text Semantically ----
def semantic_chunk(text, max_tokens=100):
    text = strip_non_ascii(text)
    sentences = sent_tokenize(text)
    chunks, chunk = [], []
    token_count = 0
    for sent in sentences:
        token_count += len(sent.split())
        if token_count <= max_tokens:
            chunk.append(sent)
        else:
            chunks.append(" ".join(chunk))
            chunk = [sent]
            token_count = len(sent.split())
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def extract_text(file_path):
    content = ""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        content = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        content = "\n".join([p.text for p in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    return semantic_chunk(content)  # return cleaned chunks

def load_documents():
    base_dir = "data/raw"
    all_chunks = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            chunks = extract_text(full_path)
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": file,
                    "type": folder
                })
    return all_chunks

# ---- 2. Embedding via API or fallback ----
def get_embedding(text):
    text = strip_non_ascii(text)
    try:
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print("Groq API failed, falling back to local model:", e)
        return local_model.encode([text])[0].tolist()

# ---- 3. Build FAISS index ----
def build_index():
    docs = load_documents()
    vectors = []
    metadata = []

    for doc in docs:
        vec = get_embedding(doc["text"][:2000])
        vectors.append(vec)
        metadata.append(doc)

    vector_array = np.array(vectors).astype("float32")
    index = faiss.IndexFlatL2(vector_array.shape[1])
    index.add(vector_array)

    faiss.write_index(index, "data/embeddings/faiss_index.index")
    with open("data/embeddings/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

# ---- 4. Fusion Retrieval (Vector + Keyword) ----
def search(query, top_k=5):
    index = faiss.read_index("data/embeddings/faiss_index.index")
    with open("data/embeddings/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    query = strip_non_ascii(query)
    query_vec = np.array([get_embedding(query)], dtype="float32")
    scores, idxs = index.search(query_vec, top_k)
    vector_hits = [metadata[i] for i in idxs[0]]

    # Keyword-based TF-IDF search
    docs_text = [strip_non_ascii(m['text']) for m in metadata]
    tfidf = TfidfVectorizer().fit_transform(docs_text)
    tfidf_query = TfidfVectorizer().fit(docs_text).transform([query])
    tfidf_scores = cosine_similarity(tfidf_query, tfidf).flatten()
    keyword_hits = [metadata[i] for i in tfidf_scores.argsort()[-top_k:][::-1]]

    # Merge & deduplicate
    combined = {entry['text']: entry for entry in vector_hits + keyword_hits}
    return list(combined.values())[:top_k]

# ---- 5. Corrective RAG Helper ----
def validate_answer_with_sources(answer, sources):
    answer = strip_non_ascii(answer)
    answer_keywords = set(re.findall(r"\\b\\w+\\b", answer.lower()))
    support_score = 0
    for source in sources:
        source_text = strip_non_ascii(source['text'])
        source_keywords = set(re.findall(r"\\b\\w+\\b", source_text.lower()))
        match = len(answer_keywords & source_keywords) / len(answer_keywords | source_keywords)
        support_score += match

    avg_score = support_score / len(sources)
    return avg_score >= 0.1
