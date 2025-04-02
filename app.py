import streamlit as st
from content_synthesis import test_synthesis
from docx import Document
from fpdf import FPDF

st.set_page_config(page_title="Research Report Generator", layout="centered")
st.title("📘 Research Report Generator")

st.markdown("Enter your query below to generate a structured research report. You can download it as a Word or PDF document.")

query = st.text_input("Enter research query")

if st.button("Generate Research Report") and query:
    sections = test_synthesis(query)
    st.subheader("📝 Report Preview")
    for section, content in sections.items():
        st.markdown(f"### {section}")
        st.write(content)

    # Save Word and PDF
    def save_word():
        doc = Document()
        doc.add_heading('Research Report', 0)
        for section, content in sections.items():
            doc.add_heading(section, level=1)
            doc.add_paragraph(str(content))
        word_path = "research_report.docx"
        doc.save(word_path)
        with open(word_path, "rb") as f:
            st.download_button("⬇️ Download Word Report", f, file_name=word_path)

    def save_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Research Report", ln=True, align="C")
        for section, content in sections.items():
            pdf.ln(10)
            pdf.set_font("Arial", size=12, style='B')
            pdf.multi_cell(0, 10, section)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, str(content))
        pdf_path = "research_report.pdf"
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF Report", f, file_name=pdf_path)

    save_word()
    save_pdf()
