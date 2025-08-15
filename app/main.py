import sys
from dotenv import load_dotenv
load_dotenv()

def try_haystack():
    try:
        from wba.document_store import setup_retriever
        r = setup_retriever()
        print("[boot] Haystack retriever ready.")
        return r
    except Exception as e:
        print(f"[boot] Haystack path unavailable: {e}")
        return None

def try_fallback():
    try:
        from wba.fallback_retriever import FallbackRetriever
        r = FallbackRetriever(json_path="extracted_text.json")
        print("[boot] Fallback retriever ready (no external deps).")
        return r
    except Exception as e:
        print(f"[boot] Fallback retriever failed: {e}")
        return None

def main():
    from wba.rag_helpers import answer_query
    retriever = try_haystack() or try_fallback()
    if retriever is None:
        print("[boot] No retriever available. Exiting.")
        sys.exit(1)

    print("Windeward Bound Assistant — type 'quit' to exit.")
    while True:
        q = input("Q: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break
        try:
            answer, docs = answer_query(retriever, q, top_k=5)
            print("\nANSWER:\n" + answer + "\n")
            print("--- Top snippets used (debug) ---")
            for i, d in enumerate(docs[:3], 1):
                meta = getattr(d, "meta", {}) or {}
                pg = meta.get("page_num", "?")
                snippet = (getattr(d, "content", "") or "")[:200].replace("\n", " ")
                print(f"{i:02d}. [page {pg}] {snippet}...")
            print("--------------------------------\n")
        except Exception as e:
            print(f"[error] answering failed: {e}")

if __name__ == "__main__":
    main()
