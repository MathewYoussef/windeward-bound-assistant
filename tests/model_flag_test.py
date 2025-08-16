import os
import sys
import numpy as np

sys.path.append("src")


def _prepare_rag(monkeypatch, model_arg):
    """Create a LocalRAG instance with heavy deps stubbed."""
    monkeypatch.setattr(
        "wba.local_rag.load_pages", lambda jp: [{"content": "text", "page_num": 1}]
    )
    monkeypatch.setattr(
        "wba.local_rag._load_embeddings",
        lambda texts: np.zeros((len(texts), 384), dtype=np.float32),
    )
    monkeypatch.setattr(
        "wba.local_rag._embed_texts",
        lambda texts: np.zeros((len(texts), 384), dtype=np.float32),
    )
    monkeypatch.setattr(
        "wba.local_rag.LocalRAG.retrieve",
        lambda self, q, top_k=8, include=None, exclude=None: [
            {"content": "doc", "page_num": 1}
        ],
    )

    used = {}

    def fake_ensure_chat(model_id=None):
        used["model_id"] = model_id or os.getenv("CHAT_MODEL_ID", "default")
        return None, None

    def fake_gen_chat(messages, max_new_tokens=180, model_id=None):
        fake_ensure_chat(model_id)
        return "stub"

    monkeypatch.setattr("wba.local_rag._gen_chat", fake_gen_chat)

    from wba.local_rag import LocalRAG

    rag = LocalRAG(json_path="dummy.json", model_id=model_arg)
    return rag, used


def test_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL_ID", "env_model")
    rag, used = _prepare_rag(monkeypatch, "flag_model")
    rag.answer("question")
    assert used["model_id"] == "flag_model"


def test_env_used_when_no_flag(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL_ID", "env_model")
    rag, used = _prepare_rag(monkeypatch, None)
    rag.answer("question")
    assert used["model_id"] == "env_model"

