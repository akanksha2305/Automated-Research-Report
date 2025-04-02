import openai
import groq
from sentence_transformers import SentenceTransformer

# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Use Groq API (OpenAI-compatible)
openai.api_key = GROQ_API_KEY
openai.api_base = GROQ_API_BASE

# Fallback sentence transformer
local_model = SentenceTransformer("all-MiniLM-L6-v2")

# Language Model Integration
def synthesize_content(query):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Generate a research report based on the following query: {query}",
            max_tokens=1500
        )
        return response.choices[0].text
    except Exception as e:
        print("Groq API failed, falling back to local model:", e)
        return local_model.encode([query])[0].tolist()

# Synthesis Pipeline Design
def structure_report(content):
    sections = {
        "Abstract": content[:300],
        "Literature Review": content[300:600],
        "Methodology": content[600:900],
        "Findings": content[900:1200],
        "Conclusion": content[1200:]
    }
    return sections

# Testing and Refinement
def test_synthesis(query):
    content = synthesize_content(query)
    report = structure_report(content)
    return report

# Example usage
if __name__ == "__main__":
    query = "AI research"
    report = test_synthesis(query)
    print(report)