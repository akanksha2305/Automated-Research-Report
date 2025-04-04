import os
import pickle
import faiss
import numpy as np
import re
from PyPDF2 import PdfReader
from docx import Document as DocxReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# === Folder Structure ===
BASE_DIR = os.path.abspath("data/raw")
EMBED_DIR = "data/embeddings"
os.makedirs(EMBED_DIR, exist_ok=True)

TOPICS = {
    "classic_ml": "Classic ML",
    "ai": "AI",
    "deep_learning": "Deep Learning"
}

model = SentenceTransformer("bert-base-nli-mean-tokens")
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

# === Chunk Text ===
def chunk_text(text, chunk_size=200, overlap=50):
    sentences = re.split(r'[.!?]\s+', text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# === Extract Text from Various Formats ===
def extract_text(file_path):
    content = ""
    if file_path.endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            content = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        except:
            print(f"⚠️ Failed to extract PDF: {file_path}")
    elif file_path.endswith(".docx"):
        try:
            doc = DocxReader(file_path)
            content = "\n".join([p.text for p in doc.paragraphs])
        except:
            print(f"⚠️ Failed to extract DOCX: {file_path}")
    elif file_path.endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except:
            print(f"⚠️ Failed to read TXT: {file_path}")
    return content

# === Load Documents from Folder ===
def load_documents(topic_folder):
    folder_path = os.path.join(BASE_DIR, topic_folder)
    docs = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if file.lower().endswith((".pdf", ".docx", ".txt")):
            text = extract_text(full_path)
            if text.strip():
                docs.append({"text": text, "filename": file})
    return docs

# === Build Index per Topic ===
def build_index_for_topic(topic_key, topic_label):
    print(f"🔧 Building index for: {topic_label}")
    docs = load_documents(topic_key)
    all_chunks, metadata, raw_texts = [], [], []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            vec = model.encode(chunk)
            all_chunks.append(vec)
            metadata.append({
                "title": doc["filename"],
                "snippet": chunk[:300]
            })
            raw_texts.append(chunk)

    vecs = np.array(all_chunks).astype("float32")
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    tfidf_matrix = tfidf_vectorizer.fit_transform(raw_texts)

    # Save
    faiss.write_index(index, f"{EMBED_DIR}/{topic_key}.index")
    with open(f"{EMBED_DIR}/{topic_key}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    with open(f"{EMBED_DIR}/{topic_key}_tfidf.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(f"{EMBED_DIR}/{topic_key}_raw.pkl", "wb") as f:
        pickle.dump(raw_texts, f)

# === Run All Topics ===
for key, label in TOPICS.items():
    build_index_for_topic(key, label)

print("✅ All topic indexes built successfully.")
