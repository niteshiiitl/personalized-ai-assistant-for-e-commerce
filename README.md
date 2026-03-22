# ShopSmart AI — E-Commerce Assistant

RAG-powered shopping assistant built with LangChain, ChromaDB, Groq LLM, and Streamlit.

## Run Locally

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
python3 main.py --ingest    # build vector store (once)
python3 main.py --app       # launch app
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (`.env` and `vectorstore/` are gitignored)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set the main file path to: `ecommerce_app/app.py`
4. Go to **Settings → Secrets** and add:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Deploy — the vectorstore builds automatically on first run

## Project Structure

```
├── data/products.csv          # product catalog
├── ecommerce_app/app.py       # Streamlit UI
├── assistant/
│   ├── ingest.py              # ChromaDB ingestion
│   ├── rag_chain.py           # LangChain + Groq RAG
│   └── finetune_prep.py       # fine-tune dataset prep
├── main.py                    # local entry point
└── requirements.txt
```
