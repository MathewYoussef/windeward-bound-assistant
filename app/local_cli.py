import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dotenv import load_dotenv
load_dotenv()

from collections import deque
from wba.local_rag import LocalRAG

DEBUG_MODE  = bool(os.getenv("WBA_DEBUG"))
MEM_HISTORY = deque(maxlen=6)     # (question, answer) pairs


def main() -> None:
    rag   = LocalRAG(json_path="extracted_text.json")
    topic = None  # sticky entity

    print("Windeward Bound Assistant (local) — type 'quit' to exit.")
    while True:
        q = input("Q: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break

        try:
            # ---- prepend brief chat history (as messages, not inline) ----
            hist_msgs = []
            for u, a in MEM_HISTORY:
                hist_msgs.extend(
                    [
                        {"role": "user", "content": u},
                        {"role": "assistant", "content": a},
                    ]
                )

            # ---- RAG ----------------------------------------------------
            answer, info = rag.answer(
                q, history=hist_msgs[-4:], sticky_topic=topic, top_k=8
            )
            topic = info.get("topic", topic)

            # ---- bookkeeping -------------------------------------------
            MEM_HISTORY.append((q, answer))

            # ---- output -------------------------------------------------
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

        except Exception as e:
            print(f"[error] {e}")

if __name__ == "__main__":
    main()
