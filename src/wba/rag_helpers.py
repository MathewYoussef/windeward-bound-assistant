import os, textwrap

def _concat_context(docs, char_limit=3000):
    chunks, used = [], 0
    for d in docs:
        t = (getattr(d, "content", None) or str(d) or "").strip().replace("\n", " ")
        if not t: 
            continue
        t = t[: max(0, char_limit - used)]
        chunks.append(f"- {t}")
        used += len(t)
        if used >= char_limit:
            break
    return "\n".join(chunks)

def build_prompt(question, docs):
    context = _concat_context(docs)
    # Pirate persona, but still precise and grounded in context.
    return textwrap.dedent(f"""
    You are the ship's knowledge-keeper aboard the tall ship STV Windeward Bound.
    Speak with a light old-sailor/pirate lilt—sprinkle a few nautical turns of phrase—
    but keep things clear, helpful, and professional. Do NOT overdo the accent.

    RULES:
    - Use ONLY the CONTEXT below. If the answer is not in the context, say you don't know.
    - Be concise: 3–6 sentences.
    - Prefer specifics (names, dates, places) if present in CONTEXT.
    - If uncertain, say so plainly (e.g., "Can’t say for certain from the ship’s logs").

    CONTEXT:
    {context}

    QUESTION: {question}

    In your answer, sound like a seasoned sailor: a bit of brine, not a parody.
    End with a short, relevant follow-up you could help with next.
    """).strip()

def _new_openai_client():
    try:
        from openai import OpenAI
        return ("new", OpenAI())
    except Exception:
        return ("old", None)

def generate_answer(prompt, model=None, temperature=0.2):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")
    which, client = _new_openai_client()
    if which == "new" and client is not None:
        mdl = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    else:
        # Legacy SDK
        import openai
        openai.api_key = api_key
        mdl = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        resp = openai.ChatCompletion.create(
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message["content"].strip()

def answer_query(retriever, question, top_k=5, model=None):
    docs = retriever.retrieve(question, top_k=top_k)
    prompt = build_prompt(question, docs)
    answer = generate_answer(prompt, model=model)
    return answer, docs
