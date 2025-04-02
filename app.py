import os
import sqlite3
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
from hashlib import sha256
from docx import Document as DocxWriter
from fpdf import FPDF
import logging

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Load env
load_dotenv()
admin_password = os.getenv("ADMIN_PASSWORD")

if not admin_password:
    st.error("ADMIN_PASSWORD not set in .env.")

# Session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "refresh_index" not in st.session_state:
    st.session_state["refresh_index"] = True

# Auth
def hash_password(password):
    return sha256(password.encode()).hexdigest()

users = {"admin": {"name": "Admin", "password": hash_password(admin_password)}}

def authenticate(username, password):
    return (username in users and users[username]["password"] == hash_password(password)), users.get(username, {}).get("name")

# Embedding model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Topic-based FAISS indexes
topic_indexes = {
    "AI": {"index": None, "ids": [], "table": "ai_articles"},
    "Healthcare": {"index": None, "ids": [], "table": "healthcare_articles"},
    "Environment": {"index": None, "ids": [], "table": "environment_articles"},
}

# DB connection
def init_database():
    conn = sqlite3.connect('repository.db')
    cursor = conn.cursor()
    return conn, cursor

conn, cursor = init_database()

# Build FAISS
def build_faiss_for_topic(topic):
    table = topic_indexes[topic]["table"]
    cursor.execute(f'SELECT id, embedding FROM {table}')
    data = cursor.fetchall()
    if not data:
        return None, []
    ids, embeddings = zip(*[(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in data])
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(embeddings))
    return index, list(ids)

# Refresh all topic indexes
def refresh_all_indexes():
    for topic in topic_indexes:
        index, ids = build_faiss_for_topic(topic)
        topic_indexes[topic]["index"] = index
        topic_indexes[topic]["ids"] = ids

if st.session_state["refresh_index"]:
    refresh_all_indexes()
    st.session_state["refresh_index"] = False

# Search
def search_topic(topic, query, k=3):
    index = topic_indexes[topic]["index"]
    ids = topic_indexes[topic]["ids"]
    if index is None:
        st.warning(f"No data available in {topic}.")
        return []
    query_vec = model.encode(query)
    distances, indices = index.search(np.array([query_vec]), k)
    return [ids[i] for i in indices[0]]

# Report Generation
def generate_report(topic, query, style):
    results = search_topic(topic, query)
    if not results:
        return None
    table = topic_indexes[topic]["table"]
    report = f"# Report on {topic} — Style: {style}\n\n"
    for res_id in results:
        cursor.execute(f"SELECT title, content FROM {table} WHERE id=?", (res_id,))
        title, content = cursor.fetchone()
        report += f"## {title}\n\n{content[:1500]}...\n\n"
    return report

# Download as Word
def download_as_word(text, filename="report.docx"):
    doc = DocxWriter()
    doc.add_heading('Generated Research Report', 0)
    for line in text.split('\n'):
        doc.add_paragraph(line)
    doc.save(filename)
    with open(filename, "rb") as f:
        st.download_button("Download Word", f, file_name=filename)

# Download as PDF
def download_as_pdf(text, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)
    with open(filename, "rb") as f:
        st.download_button("Download PDF", f, file_name=filename)

# === UI with Better Style and Persistent Login ===

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar: Login
st.sidebar.title("🔐 Login Panel")

if not st.session_state.get("authenticated", False):
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        valid, name = authenticate(username, password)
        if valid:
            st.session_state["authenticated"] = True
            st.session_state["user_name"] = name
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials.")
else:
    st.sidebar.success(f"✅ Logged in as {st.session_state['user_name']}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

# ✅ Main content only after login
if st.session_state.get("authenticated", False):
    st.title("📑 Intelligent Research Report Generator")
    st.markdown("Use your query to generate a topic-based report from preloaded research documents.")

    st.subheader("🧠 Query Input")

    topic = st.selectbox("Select Research Topic", list(topic_indexes.keys()))
    query = st.text_input("What would you like to know?")
    style = st.selectbox("Select Report Style", ["Formal", "Summary", "Bullet Points"])

    if st.button("Generate Report"):
        if query:
            report_text = generate_report(topic, query, style)
            if report_text:
                st.success("✅ Report generated successfully!")
                st.markdown("### 📄 Preview of Report")
                st.code(report_text, language="markdown")
                download_as_word(report_text)
                download_as_pdf(report_text)
            else:
                st.warning("⚠️ No results found for this query.")
        else:
            st.error("❗ Please enter a query before generating a report.")


