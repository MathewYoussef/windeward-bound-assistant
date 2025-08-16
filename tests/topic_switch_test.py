import pytest
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from wba.local_rag import LocalRAG


def make_rag():
    # Construct a LocalRAG without invoking __init__ to avoid heavy embedding work
    rag = LocalRAG.__new__(LocalRAG)
    rag.pages = [
        {
            "content": "The Windeward Bound is a tall ship from Hobart offering sail training voyages around Tasmania for young crew.",
            "page_num": 1,
        },
        {
            "content": "The Mistral II is a new training vessel being built to expand youth sail training opportunities across Australia.",
            "page_num": 2,
        },
    ]
    return rag


def fake_retrieve(self, query, top_k=8, include=None, exclude=None):
    results = []
    for p in self.pages:
        text = p["content"].lower()
        if include and not all(term in text for term in include):
            continue
        if exclude and any(term in text for term in exclude):
            continue
        results.append(p)
    return results


def fake_gen_chat(messages, max_new_tokens=180):
    # Echo the context to make assertions easy
    user = messages[-1]["content"]
    ctx = user.split("CONTEXT:\n", 1)[1].split("\n\nQUESTION:", 1)[0]
    return ctx.strip()


def test_topic_switch_mistral_overrides_windeward(monkeypatch):
    rag = make_rag()
    monkeypatch.setattr(LocalRAG, "retrieve", fake_retrieve)
    import wba.local_rag as lr
    monkeypatch.setattr(lr, "_gen_chat", fake_gen_chat)

    ans1, info1 = rag.answer("Tell me about the Windeward Bound.")
    assert "Windeward Bound" in ans1
    assert info1["topic"] == "windeward"

    ans2, info2 = rag.answer("And what is Mistral II?", sticky_topic=info1["topic"])
    assert "Mistral II" in ans2
    assert "Windeward" not in ans2
    assert info2["topic"] == "mistral"
