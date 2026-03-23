# ShopSmart AI — E-Commerce Assistant

RAG-powered shopping assistant built with LangChain, ChromaDB, Groq LLM, and Streamlit.

## Run Locally

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
python3 main.py --ingest    # build vector store (once)
python3 main.py --app       # launch app
``

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
