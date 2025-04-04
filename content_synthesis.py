import openai
import os
import re
from dotenv import load_dotenv

from langchain_core.tracers import LangChainTracer
from langchain_core.callbacks import CallbackManager
import os

# Set up LangSmith tracer
tracer = LangChainTracer()
callback_manager = CallbackManager([tracer])

# ---- Helper: Strip non-ASCII characters ----
def strip_non_ascii(text):
    return ''.join(char for char in text if ord(char) < 128)
 
# Load environment variables
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = os.getenv("GROQ_API_BASE")
 
# Use Groq-supported model
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

LLM_MODEL = "llama3-8b-8192"

chat = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base=os.getenv("GROQ_API_BASE"),
    temperature=0.7,
    callback_manager=callback_manager
)

def synthesize_section(query, section):
    prompt = f"Write the {section} section of a research report on: {query}"
    try:
        messages = [
            SystemMessage(content="You are an expert research assistant."),
            HumanMessage(content=prompt)
        ]
        response = chat(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Groq API failed for {section}:", e)
        return "[Error generating section]"
 
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
    sections = ["Abstract", "Literature Review", "Methodology", "Findings", "Conclusion"]
    return {section: synthesize_section(query, section) for section in sections}
   
# ✅ 4. Run it and print output
if __name__ == "__main__":
    query = "The role of LLMs in automated research report generation"
    report = test_synthesis(query)
 
    print("\n📝 Structured Research Report:\n")
    for section, text in report.items():
        print(f"## {section}\n{text}\n")
