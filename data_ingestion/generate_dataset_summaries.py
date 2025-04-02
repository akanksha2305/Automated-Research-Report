import os

SAVE_DIR = "data/raw/datasets/"
os.makedirs(SAVE_DIR, exist_ok=True)

summaries = [
    ("AI Research Papers Metadata", "Contains titles, authors, abstracts of 100k+ papers used in building citation graphs and recommender systems."),
    ("RAG System Evaluation Dataset", "This dataset provides sample queries, relevant documents, and reference summaries for evaluating retrieval-augmented generation pipelines."),
    ("OpenAI Embedding Test Set", "Includes text chunks and corresponding embeddings generated via OpenAI API for reproducibility tests."),
    ("Scientific Abstracts for QA", "A set of abstracts used to train and test question-answering over scientific text."),
    ("Vector Search Benchmark", "Comparison dataset used to evaluate vector DBs (FAISS, Pinecone, Chroma) on precision and speed."),
    ("LLM Response Rating Dataset", "Human-rated outputs from various prompts and models, useful for reinforcement learning."),
    ("Groq Inference Timing Logs", "Timing benchmarks for LLaMA-2 and Mixtral models running on Groq for research inference loads."),
]

def generate_dataset_summaries():
    for i, (title, desc) in enumerate(summaries):
        filename = os.path.join(SAVE_DIR, f"{i+1:02d}_{title.replace(' ', '_')}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n{desc}")
        print(f"✅ Saved: {filename}")

generate_dataset_summaries()
