import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")

from dotenv import load_dotenv
load_dotenv()

from wba.local_rag import LocalRAG
import os
DEBUG_MODE = bool(os.getenv('WBA_DEBUG'))  # set WBA_DEBUG=1 to show context

from collections import deque

# Keep last N question/answer pairs
MEM_HISTORY = deque(maxlen=6)  # type: list[tuple[str,str]]


def main():
    rag = LocalRAG(json_path="extracted_text.json")
    history = []  # list of {"role": "user"/"assistant", "content": "..."}
    topic = None  # "windeward" | "mistral" | None

    print("Windeward Bound Assistant (local, chat) — type 'quit' to exit.")
    while True:
        q = input("Q: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break

        try:

            try:
            # prepend brief chat history (if any)
            hist_block = ""
            if MEM_HISTORY:
            hist_block = (
            "\n\nPREVIOUS TURNS:\n" +
            "\n".join(f"USER: {u}\nASSISTANT: {a}" for u, a in MEM_HISTORY)
            )
            new_q = q + hist_block if hist_block else q

            # RAG
            answer, info = rag.answer(new_q, history=history[-4:], sticky_topic=topic, top_k=8)
            topic = info.get("topic", topic)

            # record turn
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": answer})
            MEM_HISTORY.append((q, answer))

            # output
            print("\nANSWER:\n" + answer + "\n")

            if DEBUG_MODE:
            ctx_preview = info.get("context", [])[:3]
            if ctx_preview:
            print("--- Context sentences used ---")
            for line in ctx_preview:
            print(line if len(line) < 240 else line[:237] + "...")
            pages = info.get("pages", [])
            if pages:
            print(f"(pages: {', '.join(map(str, pages))})")
            print("------------------------------\n")
            print("\nANSWER:\n" + answer + "\n")
            # --- optional debug context ----------------------------------
            if DEBUG_MODE:
            ctx_preview = info.get('context', [])[:3]
            if ctx_preview:
            print('--- Context sentences used ---')
            for line in ctx_preview:
            print(line if len(line) < 240 else line[:237] + '...')
            pages = info.get('pages', [])
            if pages:
            print(f"(pages: {', '.join(map(str, pages))})")
            print('------------------------------\n')
            if not DEBUG_MODE:
            continue  # normal mode: skip retrieval details
                
            ctx_preview = info.get('context', [])[:3]
            if ctx_preview:
            print('--- Context sentences used ---')
            for line in ctx_preview:
            print(line if len(line) < 240 else line[:237] + '...')
            pages = info.get('pages', [])
            if pages:
            print(f"(pages: {', '.join(map(str, pages))})")
            print('------------------------------\n')
            except Exception as e:
            print(f"[error] {e}")
            if __name__ == "__main__":
            main()
