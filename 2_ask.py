"""
2_ask.py
========
CLI entry point for the ONLINE query pipeline (terminal version).
Good for quick testing. For the actual demo you show your TL, use:
    streamlit run app.py

Run:
    python 2_ask.py
"""

import rag_core


def main():
    vector_store = rag_core.load_vector_store()
    print(f"Loaded {len(vector_store)} chunks from {rag_core.FAISS_INDEX_FILE}.")
    print("Type a question, or 'quit' to exit.\n")

    while True:
        question = input("Ask> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer, sources = rag_core.answer_question(question, vector_store)

        print("\n--- ANSWER ---")
        print(answer)

        if sources:
            print("\n--- SOURCES (folder / file / chunk / relevance) ---")
            for i, (score, entry) in enumerate(sources, start=1):
                print(f"[{i}] {entry['path']}  (chunk {entry['chunk_index']}, relevance {score:.3f})")
        print()


if __name__ == "__main__":
    main()