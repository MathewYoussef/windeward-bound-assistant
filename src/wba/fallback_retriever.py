from pathlib import Path
import json, re

class Doc:
    def __init__(self, content, meta=None):
        self.content = content
        self.meta = meta or {}

class FallbackRetriever:
    """
    Zero-dependency, in-memory retriever.
    Loads ./extracted_text.json and returns top_k by naive keyword score.
    """

    def __init__(self, json_path="extracted_text.json"):
        p = Path(json_path)
        if not p.exists():
            raise FileNotFoundError(f"Could not find {json_path}. Place it in project root or update path.")
        data = json.loads(p.read_text(encoding="utf-8"))
        self.docs = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    self.docs.append(Doc(item))
                elif isinstance(item, dict):
                    text = item.get("content") or item.get("text") or ""
                    meta = {k:v for k,v in item.items() if k not in ("content","text")}
                    self.docs.append(Doc(text, meta=meta))
        elif isinstance(data, dict):
            texts = data.get("texts") or data.get("items") or []
            for t in texts:
                if isinstance(t, str):
                    self.docs.append(Doc(t))
                else:
                    self.docs.append(Doc(json.dumps(t)))
        else:
            raise ValueError("Unsupported JSON format for extracted_text.json")

    def _score(self, q, t):
        if not q or not t: return 0
        q_terms = re.findall(r"[A-Za-z0-9']+", q.lower())
        t_low  = t.lower()
        return sum(t_low.count(term) for term in q_terms)

    def retrieve(self, query, top_k=5):
        ranked = sorted(((self._score(query, d.content), d) for d in self.docs),
                        key=lambda x: x[0], reverse=True)
        return [d for s, d in ranked[:top_k] if s > 0] or [d for s, d in ranked[:top_k]]
