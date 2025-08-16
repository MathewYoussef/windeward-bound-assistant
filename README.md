# Windeward Bound Assistant (staging)

A lightweight LLM-powered assistant about the STV **Windeward Bound**.
This is a cleaned, staged version of earlier experiments, in a professional structure.

## Structure
    windeward-bound-assistant/
      app/                  # CLI / entrypoints
        main.py
      src/wba/              # your modules
        __init__.py
        document_store.py
        pdf_to_text.py
        RAG.py
        Web_scrape.py
      tests/                # experimental/test scripts preserved
      data/
        sample/
          extracted_text.json
      scripts/
        run_local.sh
      requirements.txt
      .env.sample
      .gitignore

## Quickstart
1) Copy env and add your key (do NOT commit real secrets):
       cp .env.sample .env
       # then open .env and set OPENAI_API_KEY

2) Install dependencies:
       pip install -r requirements.txt

3) Run locally:
       ./scripts/run_local.sh
       # shows retrieved snippets (debug)

4) Web UI:
       streamlit run app/web_ui.py
       # Runs entirely locally; no OpenAI key required once models are cached.
