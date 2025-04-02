# backend.py

import os
import faiss
import pickle
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Use Groq API (OpenAI-compatible)
openai.api_key = GROQ_API_KEY
openai.api_base = GROQ_API_BASE

# Fallback sentence transformer
local_model = SentenceTransformer("bert-base-nli-mean-tokens")

### 1. Extract text from documents
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            return f.read()
    return ""

### 2. Load all documents
def load_documents():
    base_dir = "data/raw"
    all_docs = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            text = extract_text(full_path)
            if text:
                all_docs.append({
                    "text": text,
                    "type": folder,
                    "filename": file
                })
    return all_docs

### 3. Generate embedding (Groq/OpenAI-compatible)
# Import OpenAI client (updated)
from openai import OpenAI
client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE)

# Updated get_embedding function
def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding  # Works with OpenAI >=1.0.0
    except Exception as e:
        print("Groq API failed, falling back to local model:", e)
        return local_model.encode([text])[0].tolist()


### 4. Build FAISS index
def build_index():
    docs = load_documents()
    vectors = []
    metadata = []

    for doc in docs:
        vec = get_embedding(doc["text"][:2000])
        vectors.append(vec)
        metadata.append({
            "title": doc["filename"],
            "snippet": doc["text"][:300],
            "source": doc["type"]
        })

    # Convert and index
    vector_array = np.array(vectors).astype("float32")
    index = faiss.IndexFlatL2(vector_array.shape[1])
    index.add(vector_array)

    # Save
    faiss.write_index(index, "data/embeddings/faiss_index.index")
    with open("data/embeddings/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

### 5. Load index + search
def search(query, top_k=5):
    index = faiss.read_index("data/embeddings/faiss_index.index")
    with open("data/embeddings/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    query_vec = np.array([get_embedding(query)], dtype="float32")
    scores, idxs = index.search(query_vec, top_k)
    return [metadata[i] for i in idxs[0]]
