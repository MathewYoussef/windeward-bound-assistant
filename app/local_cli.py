#!/usr/bin/env python3
import os
from collections import deque
from wba.local_rag import LocalRAG

# --- flags ------------------------------------------------------------------
DEBUG_MODE   = bool(os.getenv("WBA_DEBUG"))
MEM_HISTORY  = deque(maxlen=6)         # recent (q,a) pairs for prompt
# ----------------------------------------------------------------------------

def main() -> None:
    rag     = LocalRAG(json_path="extracted_text.json")
    history = []                       # Conversation history for the model
    topic   = None                     # sticky entity (“windeward” / “mistral”)

    print("Windeward Bound Assistant (local) — type 'quit' to exit.")
    while True:
        q = input("Q: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break

        # -------- prepend brief chat history --------
        hist_block = ""
        if MEM_HISTORY:
            hist_block = "\n\nPREVIOUS TURNS:\n" + "\n".join(
                f"USER: {u}\nASSISTANT: {a}" for u, a in MEM_HISTORY
            )
        new_q = q + hist_block if hist_block else q

        # -------- RAG + answer generation wrapped in try/except --------
        try:
                new_q, history=history[-4:], sticky_topic=topic, top_k=8
            )
            topic = info.get("topic", topic)
        except Exception as e:
            print(f"[error] {e}")
            continue

        # -------- record & memory --------
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        MEM_HISTORY.append((q, answer))

        # -------- output --------
        print("\nANSWER:\n" + answer + "\n")

        if DEBUG_MODE:
            ctx_preview = info.get("context", [])[:3]
            if ctx_preview:
                print("--- Context sentences used ---")
                for line in ctx_preview:
                    print(line if len(line) < 240 else line[:237] + "…")
                pages = info.get("pages", [])
                if pages:
                    print(f"(pages: {', '.join(map(str, pages))})")
                print("------------------------------\n")

if __name__ == "__main__":
    main()
