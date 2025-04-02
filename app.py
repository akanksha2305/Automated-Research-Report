# Import Required Libraries
import sqlite3
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from hashlib import sha256

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
admin_password = os.getenv("ADMIN_PASSWORD")

# Ensure admin password is correctly loaded
if not admin_password:
    st.error("ADMIN_PASSWORD environment variable not set. Please check your .env file.")

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []
if "refresh_index" not in st.session_state:
    st.session_state["refresh_index"] = True

# Hash function for password validation
def hash_password(password):
    return sha256(password.encode()).hexdigest()

# Hardcoded user authentication
users = {"admin": {"name": "Admin", "password": hash_password(admin_password)}}

def authenticate(username, password):
    if username in users and users[username]["password"] == hash_password(password):
        return True, users[username]["name"]
    return False, None

# Initialize SentenceTransformer Model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Database Setup
def init_database():
    try:
        conn = sqlite3.connect('research_repository.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS research_data (
            id INTEGER PRIMARY KEY,
            title TEXT,
            content TEXT,
            tags TEXT,
            embedding BLOB
        )''')
        conn.commit()
        return conn, cursor
    except sqlite3.Error as e:
        st.error(f"Database initialization error: {e}")
        raise

conn, cursor = init_database()

def add_to_database(title, content, tags):
    try:
        embedding = model.encode(content)
        cursor.execute('INSERT INTO research_data (title, content, tags, embedding) VALUES (?, ?, ?, ?)',
                       (title, content, tags, embedding.tobytes()))
        conn.commit()
        st.session_state['refresh_index'] = True  # Mark FAISS index for rebuild
    except sqlite3.Error as e:
        st.error(f"Failed to add data to database: {e}")

# Extract Text from PDF
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
        return None

# Extract Text from Word Documents
def extract_text_from_word(file):
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Failed to extract text from Word file: {e}")
        return None

# Build FAISS Index
def build_faiss_index():
    try:
        cursor.execute('SELECT id, embedding FROM research_data')
        data = cursor.fetchall()

        if not data:  # Handle empty database
            st.warning("The database table `research_data` is empty. Please add some entries.")
            return None, None

        ids, embeddings = zip(*[(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in data])
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.vstack(embeddings))
        return index, ids
    except Exception as e:
        st.error(f"Failed to build FAISS index: {e}")
        raise

# Refresh FAISS Index When Needed
def refresh_faiss_index():
    if st.session_state['refresh_index']:
        global index, ids
        index, ids = build_faiss_index()
        st.session_state['refresh_index'] = False  # Reset the flag

refresh_faiss_index()

# Search for Content
def search(query, k=1):
    if index is None:
        st.warning("FAISS index is not built. Please add some data to the database first.")
        return []
    query_embedding = model.encode(query)
    distances, indices = index.search(np.array([query_embedding]), k)
    results = [ids[i] for i in indices[0]]
    return results

# Authentication and Logout
st.sidebar.header("Login")
if not st.session_state["authenticated"]:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        authenticated, name = authenticate(username, password)
        if authenticated:
            st.sidebar.success(f"Welcome, {name}!")
            st.session_state["authenticated"] = True
        else:
            st.sidebar.error("Authentication failed. Please check your credentials.")
else:
    st.sidebar.header("Actions")
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.sidebar.success("Logged out successfully!")
        st.stop()

# File Upload and Persistent Storage
if st.session_state["authenticated"]:
    st.title("Upload Research Data")
    uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            text_content = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            text_content = extract_text_from_word(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a PDF or Word document.")

        if text_content:
            title = st.text_input("Enter a title for this document:")
            tags = st.text_input("Enter tags (comma-separated):")

            if st.button("Add to Database"):
                if title and tags:
                    add_to_database(title, text_content, tags)
                    st.session_state["uploaded_files"].append({"title": title, "tags": tags})
                    st.success("Document added successfully!")
                else:
                    st.error("Please provide both a title and tags.")

    # Display Uploaded Files in Sidebar
    if st.session_state["uploaded_files"]:
        st.sidebar.header("Uploaded Documents")
        for file in st.session_state["uploaded_files"]:
            st.sidebar.write(f"- {file['title']}")

# Search Functionality
if st.session_state["authenticated"]:
    st.title("Search Research Data")
    query = st.text_input("Enter your search query:")
    if st.button("Submit"):
        if query:
            refresh_faiss_index()  # Ensure FAISS index is up-to-date
            results = search(query)
            if results:
                st.subheader("Search Results")
                unique_results = set()  # Track unique result IDs
                for result_id in results:
                    if result_id not in unique_results:  # Check for duplicates
                        unique_results.add(result_id)
                        cursor.execute('SELECT title, content FROM research_data WHERE id=?', (result_id,))
                        title, content = cursor.fetchone()
                    st.write(f"**Title:** {title}")
                    st.write(f"**Content:** {content[:3000]}...")  # Show snippet of content
            else:
                st.warning("No results found for your query.")
