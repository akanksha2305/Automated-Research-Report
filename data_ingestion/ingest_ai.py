# ingest_general_ai.py

import os
import requests
import xml.etree.ElementTree as ET

SAVE_DIR = "data/raw/general_ai/"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_arxiv_papers(query="artificial intelligence OR AGI OR AI systems", max_results=7):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    root = ET.fromstring(response.content)

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip().replace(" ", "_")[:50]
        pdf_url = entry.find("{http://www.w3.org/2005/Atom}id").text.replace("abs", "pdf") + ".pdf"

        print(f"Downloading: {title}")
        pdf = requests.get(pdf_url)
        if pdf.status_code == 200:
            with open(os.path.join(SAVE_DIR, f"{title}.pdf"), "wb") as f:
                f.write(pdf.content)

download_arxiv_papers()
