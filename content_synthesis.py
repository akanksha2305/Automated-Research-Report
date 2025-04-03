import openai
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = os.getenv("GROQ_API_BASE")

# Use Groq-supported model
LLM_MODEL = "llama3-8b-8192"

# ✅ 1. Generate structured content using Groq (OpenAI-compatible)
def synthesize_content(query):
    try:
        response = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert research assistant."},
                {"role": "user", "content": f"Generate a detailed research report for the topic: {query}. Include the following sections: Abstract, Literature Review, Methodology, Findings, Conclusion."}
            ],
            max_tokens=2048,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print("Groq API failed:", e)
        return ""

# ✅ 2. Parse content into structured sections (flexible parsing with fallback)
def structure_report(content):
    if not content:
        return {section: "[No content generated]" for section in ["Abstract", "Literature Review", "Methodology", "Findings", "Conclusion"]}

    sections = ["Abstract", "Literature Review", "Methodology", "Findings", "Conclusion"]
    report = {}
    pattern = "|".join([fr"\b{section}\b" for section in sections])
    matches = list(re.finditer(pattern, content, re.IGNORECASE))

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_title = matches[i].group(0).strip().title()
        section_content = content[start + len(section_title):end].strip()
        report[section_title] = section_content

    # Ensure all expected sections are in report
    for sec in sections:
        if sec not in report:
            report[sec] = "[Section not found in response]"

    return report

# ✅ 3. Combined tester
def test_synthesis(query):
    content = synthesize_content(query)
    return structure_report(content)

# ✅ 4. Run it and print output
if __name__ == "__main__":
    query = "The role of LLMs in automated research report generation"
    report = test_synthesis(query)

    print("\n📝 Structured Research Report:\n")
    for section, text in report.items():
        print(f"## {section}\n{text}\n")
