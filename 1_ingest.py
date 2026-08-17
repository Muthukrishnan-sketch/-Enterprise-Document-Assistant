"""
1_ingest.py
===========
CLI entry point for the OFFLINE indexing pipeline. All the real
logic lives in rag_core.py - this file just calls it and prints
progress, so it doubles as the simplest possible example of how to
use rag_core.

Run this once (or again whenever documents/ changes):
    python 1_ingest.py
"""

import rag_core


def main():
    print(f"Indexing every .txt file under {rag_core.DOCS_FOLDER}/ ...\n")

    vector_store = rag_core.build_vector_store(
        chunk_size=rag_core.DEFAULT_CHUNK_SIZE,
        chunk_overlap=rag_core.DEFAULT_CHUNK_OVERLAP,
        progress_callback=lambda msg: print(" ", msg)
    )

    rag_core.save_vector_store(vector_store)

    print(f"\nDone. {len(vector_store)} chunks saved to {rag_core.FAISS_INDEX_FILE} + {rag_core.CHUNK_METADATA_FILE}")
    print("Next, run ONE of:")
    print("  python 2_ask.py        (ask questions in the terminal)")
    print("  streamlit run app.py   (the web UI demo)")


if __name__ == "__main__":
    main()