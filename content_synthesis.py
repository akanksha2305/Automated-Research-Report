import openai
import os
import re
from dotenv import load_dotenv
from backend import search  # ✅ Uses your FAISS+TFIDF fusion retrieval

# Load API keys
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = os.getenv("GROQ_API_BASE")

LLM_MODEL = "llama3-8b-8192"

# 🔍 RAG-enabled prompt
def synthesize_section(query, section, context):
    prompt = f"""
You are an expert research assistant.

Given the following research context extracted from papers and articles:

---CONTEXT START---
{context}
---CONTEXT END---

Write the {section} section of a research report for the topic: "{query}"
Ensure it is factual, concise, and derived from the context above.
"""

    try:
        response = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You generate research content based on retrieved documents."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Groq API failed for {section}:", e)
        return "[Error generating section]"

#RAG pipeline — uses FAISS retrieval
def test_synthesis(query, top_k=5):
    retrieved_docs = search(query, top_k=top_k)
    combined_context = "\n\n".join([doc['text'] for doc in retrieved_docs])
    
    sections = ["Abstract", "Literature Review", "Methodology", "Findings", "Conclusion"]
    return {
        section: synthesize_section(query, section, combined_context)
        for section in sections
    }

#Test the full report
if __name__ == "__main__":
    query = "The role of LLMs in automated research report generation"
    report = test_synthesis(query)

    print("\n📝 RAG-Based Structured Research Report:\n")
    for section, text in report.items():
        print(f"## {section}\n{text}\n")
