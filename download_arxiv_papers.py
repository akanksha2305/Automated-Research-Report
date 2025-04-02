# download_arxiv_papers.py

import os
import requests
import xml.etree.ElementTree as ET

SAVE_DIR = "data/raw/academic_papers/"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_arxiv_pdfs(query="machine learning", max_results=10):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    response = requests.get(url)
    root = ET.fromstring(response.content)

    for i, entry in enumerate(root.findall("{http://www.w3.org/2005/Atom}entry")):
        pdf_url = entry.find("{http://www.w3.org/2005/Atom}id").text.replace("abs", "pdf")
        pdf_url += ".pdf"

        paper_title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip().replace(" ", "_").replace("/", "_")
        filename = f"{paper_title[:50]}.pdf"

        print(f"Downloading: {filename}")
        pdf_response = requests.get(pdf_url)

        if pdf_response.status_code == 200:
            with open(os.path.join(SAVE_DIR, filename), "wb") as f:
                f.write(pdf_response.content)

download_arxiv_pdfs(query="retrieval augmented generation", max_results=10)
