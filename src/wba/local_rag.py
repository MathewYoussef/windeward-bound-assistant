import os, re, json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# -------- embedding-cache --------
EMB_CACHE_PATH = Path("data/embeddings.npy")

def _build_embeddings(texts: List[str]) -> np.ndarray:
    """(Re)create the sentence-embedding cache for provided texts."""
    EMB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    arr = _embed_texts(texts)
    import numpy as _np
    _np.save(EMB_CACHE_PATH, arr)
    return arr

def _load_embeddings(texts: List[str]) -> np.ndarray:
    """Load prebuilt embeddings; raise if cache is missing or stale."""
    if not EMB_CACHE_PATH.exists():
        raise FileNotFoundError(
            "Missing embedding cache; run codex_setup.sh to build it"
        )
    import numpy as _np
    arr = _np.load(EMB_CACHE_PATH)
    if arr.shape[0] != len(texts):
        raise ValueError("Embedding cache is out of date; rebuild required")
    return arr

# ---- Globals (lazy-loaded) ----
_SENT_ENCODER = None
_TOK = None
_LLM = None
_DEVICE = "cpu"

NOISY_PATTERNS = [
    r"\bcomment\b",
    r"From Wikipedia, the free encyclopedia",
    r"This article needs additional citations",
    r"\bBuy a copy\b",
]

# ---------------- Core utils ----------------
def _device_hint():
    global _DEVICE
    try:
        import torch
        if torch.backends.mps.is_available():
            _DEVICE = "mps"
        else:
            _DEVICE = "cpu"
    except Exception:
        _DEVICE = "cpu"
    return _DEVICE

def _ensure_sentence_encoder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    global _SENT_ENCODER
    if _SENT_ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _SENT_ENCODER = SentenceTransformer(model_name, device=_device_hint())
    return _SENT_ENCODER

def _embed_texts(texts: List[str]) -> np.ndarray:
    enc = _ensure_sentence_encoder()
    vecs = enc.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)

def _ensure_chat(model_id=None):
    """Load a small instruction-tuned chat LLM."""
    global _TOK, _LLM
    if _TOK is None or _LLM is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_id = model_id or os.getenv("CHAT_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
        _device = _device_hint()
        torch_dtype = None
        try:
            import torch
            torch_dtype = torch.float16 if _device == "mps" else torch.float32
        except Exception:
            pass
        _TOK = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if _TOK.pad_token_id is None:
            _TOK.pad_token = _TOK.eos_token  # allows making attention_mask
        _LLM = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    return _TOK, _LLM

def _tok_max_ctx(tok, llm) -> int:
    cand = getattr(tok, "model_max_length", None)
    if cand is None or cand > 32768:
        cand = getattr(getattr(llm, "config", None), "max_position_embeddings", 4096) or 4096
    return int(cand)

def _is_noisy_sentence(s: str) -> bool:
    s = s.strip()
    if len(s) < 40:
        return True
    for pat in NOISY_PATTERNS:
        if re.search(pat, s, flags=re.I):
            return True
    return False

def _select_sentences(query: str, text: str, k: int = 3) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    q_terms = set(re.findall(r"[A-Za-z0-9']+", query.lower()))
    scored = []
    for s in sents:
        s0 = s.strip()
        if not s0 or _is_noisy_sentence(s0):
            continue
        low = s0.lower()
        score = sum(low.count(t) for t in q_terms) + (2 if "windeward" in low else 0)
        scored.append((score, s0))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [s for _, s in scored[:k]]
    if not out:
        out = [s.strip() for s in sents if s.strip() and not _is_noisy_sentence(s)][:k]
    return out

def _proper_case(text: str) -> str:
    repl = [
        (r"\bwindeward bound trust\b", "Windeward Bound Trust"),
        (r"\bwindeward bound\b", "Windeward Bound"),
        (r"\bmistral ii\b", "Mistral II"),
        (r"\bhobart\b", "Hobart"),
        (r"\btasmania\b", "Tasmania"),
        (r"\baustralia\b", "Australia"),
    ]
    out = text
    for pat, rep in repl:
        out = re.sub(pat, rep, out, flags=re.I)
    return out

def _concat_with_pages(question: str, docs: List[Dict[str, Any]], char_budget=1800) -> Tuple[List[str], List[int]]:
    pieces, pages, used = [], [], 0
    for d in docs:
        pg = d.get("page_num")
        sel = _select_sentences(question, d["content"], k=3)
        if not sel:
            continue
        chunk = " ".join(sel)
        chunk = chunk[: max(0, char_budget - used)]
        if not chunk:
            break
        tag = f"[p.{pg}] " if pg is not None else ""
        pieces.append(tag + chunk)
        used += len(chunk)
        if pg is not None:
            pages.append(pg)
        if used >= char_budget:
            break
    return pieces, sorted(set(pages))

def _truncate_text_tokens(text: str, tok, max_tokens: int) -> str:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens], skip_special_tokens=True)

# ---------------- Voice control ----------------
def _pirate_tone(text: str, level: str = "medium") -> str:
    """Light, consistent sailor tone. No parody."""
    if not text:
        return text
    s = text.strip()

    # Always keep facts; only add light seasoning.
    needs_opener = not re.search(r"\bAye\b", s)
    needs_closer = not re.search(r"Fair winds", s, flags=re.I)

    # Mild replacements (do not mangle numbers or names)
    def soft_replace(a, b):
        nonlocal s
        s = re.sub(rf"\b{a}\b", b, s, flags=re.I)

    if level.lower() in ("medium", "bold"):
        soft_replace("vessel", "ship")
        soft_replace("program", "voyage program")
        soft_replace("participants", "crew")

    if needs_opener:
        s = "Aye, " + s[0].lower() + s[1:] if s and s[0].isalpha() else "Aye, " + s

    if level.lower() == "bold":
        # Add one tasteful sailorism if none present
        if not re.search(r"weather eye|under sail|charted|astern|aboard|ship’s log", s, flags=re.I):
            s += " Keep a weather eye on the horizon."

    if needs_closer:
        s = s.rstrip() + " Fair winds."

    return _proper_case(s)

# ---------------- Prompting ----------------
def _build_messages(question: str, context_sentences: List[str], history: List[Dict[str,str]]) -> List[Dict[str,str]]:
    system = {
        "role": "system",
        "content": (
            "You are the ship’s knowledge-keeper aboard the tall ship Windeward Bound.\n"
            "Speak in a light, steady sailor’s voice throughout (2–3 nautical turns of phrase max). "
            "Do NOT lapse into neutral tone mid-answer. Stay clear and professional.\n"
            "Rules: Use ONLY the CONTEXT. Answer in 3–5 sentences. Prefer specifics. If unsure, say so plainly."
        ),
    }
    user = {
        "role": "user",
        "content": (
            "CONTEXT:\n" + "\n".join(context_sentences) + "\n\n"
            f"QUESTION: {question}\n"
            "Begin your reply with “Aye,” and keep the sailor’s tone consistently to the end."
        ),
    }
    msgs = [system]
    trimmed = history[-4:] if history else []
    msgs.extend(trimmed)
    msgs.append(user)
    return msgs

def _gen_chat(messages: List[Dict[str,str]], max_new_tokens=180, model_id=None) -> str:
    import torch
    tok, llm = _ensure_chat(model_id)
    max_ctx = _tok_max_ctx(tok, llm)

    def encode(msgs):
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(llm.device)
        mask = torch.ones_like(ids)  # explicit attention mask even with pad==eos
        return ids, mask

    input_ids, attention_mask = encode(messages)

    headroom = max(96, max_new_tokens + 32)
    if input_ids.shape[1] > max_ctx - headroom:
        msgs = [messages[0]] + messages[-1:]   # drop history first
        input_ids, attention_mask = encode(msgs)
        if input_ids.shape[1] > max_ctx - headroom:
            # Trim context inside the user message
            u = msgs[-1]["content"]
            pre, ctx_and_rest = u.split("CONTEXT:\n", 1)
            ctx, rest = ctx_and_rest.split("\n\nQUESTION:", 1)
            ctx = _truncate_text_tokens(ctx, tok, max(128, (max_ctx - headroom)//2))
            msgs[-1]["content"] = f"{pre}CONTEXT:\n{ctx}\n\nQUESTION:{rest}"
            input_ids, attention_mask = encode(msgs)
        messages = msgs

    gen_ids = llm.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,           # deterministic for tidy tone/facts
        num_beams=4,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    out = tok.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    return _proper_case(out)

# ---------------- Public API ----------------
def load_pages(json_path: str) -> List[Dict[str, Any]]:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {json_path}. Place it in project root.")
    data = json.loads(p.read_text(encoding="utf-8"))
    pages = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                content = item.get("content") or item.get("text") or ""
                pg = item.get("page_num", None)
            else:
                content, pg = str(item), None
            if content and content.strip():
                pages.append({"content": content.strip(), "page_num": pg})
    elif isinstance(data, dict):
        for t in data.get("texts", []) or []:
            pages.append({"content": str(t).strip(), "page_num": None})
    else:
        raise ValueError("Unsupported JSON structure in extracted_text.json")
    return pages

class LocalRAG:
    def __init__(self, json_path="extracted_text.json", model_id=None):
        self.model_id = model_id
        self.pages = load_pages(json_path)
        self.texts = [p["content"] for p in self.pages]
        self.embeddings = (
            _load_embeddings(self.texts)
            if self.texts
            else np.zeros((0, 384), dtype=np.float32)
        )

    def retrieve(self, query: str, top_k: int = 8,
                 include: Optional[List[str]] = None,
                 exclude: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if not self.texts:
            return []
        qv = _embed_texts([query])
        sims = (qv @ self.embeddings.T)[0]
        idx = np.argsort(-sims)[:top_k*2]  # over-retrieve; filter below

        out = []
        for i in idx:
            doc = self.pages[i]
            txt_low = doc["content"].lower()
            if include and not all(term in txt_low for term in include):
                continue
            if exclude and any(term in txt_low for term in exclude):
                continue
            if any(re.search(p, doc["content"], flags=re.I) for p in NOISY_PATTERNS):
                continue
            out.append(doc | {"score": float(sims[i])})
            if len(out) >= top_k:
                break
        return out or [self.pages[i] | {"score": float(sims[i])} for i in idx[:top_k]]

    def answer(self, question: str,
               history: Optional[List[Dict[str,str]]] = None,
               sticky_topic: Optional[str] = None,
               top_k: int = 8) -> Tuple[str, Dict[str, Any]]:
        q_low = question.lower()
        include, exclude = [], []

        # Prefer entities mentioned in the current question; fall back to the
        # previous sticky topic only if no entity is mentioned.
        if "windeward" in q_low:
            include.append("windeward")
            exclude.extend(["mistral", "blue water warriors"])
        elif "mistral" in q_low:
            include.append("mistral")
        elif sticky_topic == "windeward":
            include.append("windeward")
            exclude.extend(["mistral", "blue water warriors"])
        elif sticky_topic == "mistral":
            include.append("mistral")

        include = include or None
        exclude = exclude or None

        top = self.retrieve(question, top_k=top_k, include=include, exclude=exclude)
        if not top:
            return (
                "Can’t say for certain from the ship’s logs—no relevant texts aboard.",
                {"pages": [], "context": [], "topic": sticky_topic},
            )

        fed_sents, pages = _concat_with_pages(question, top, char_budget=1500)

        history = history or []
        messages = _build_messages(question, fed_sents, history)
        raw = _gen_chat(messages, max_new_tokens=180, model_id=self.model_id)

        # Enforce tone consistently, lightly
        level = os.getenv("PIRATE_LEVEL", "medium")
        answer = _pirate_tone(raw, level=level)

        # Topic stickiness for next turn
        topic = sticky_topic
        text_pool = " ".join([d["content"].lower() for d in top[:3]])
        if "windeward" in q_low or "windeward" in text_pool:
            topic = "windeward"
        elif "mistral" in q_low or "mistral" in text_pool:
            topic = "mistral"

        return (answer, {"pages": pages, "context": fed_sents, "topic": topic})
