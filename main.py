"""
Entry point for the ShopSmart AI e-commerce assistant.

Steps:
  1. Ingest products into ChromaDB (run once)
  2. Launch the Streamlit app

Usage:
  python main.py --ingest       # build/rebuild the vector store
  python main.py --app          # launch the Streamlit UI
  python main.py --finetune     # generate fine-tune JSONL dataset
"""

import sys
import os
import argparse
import subprocess


def run_ingest():
    print("Ingesting products into ChromaDB...")
    from assistant.ingest import build_vectorstore
    build_vectorstore()


def run_app():
    print("Launching ShopSmart AI...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.join("ecommerce_app", "app.py")
    ])


def run_finetune():
    print("Generating fine-tune dataset...")
    from assistant.finetune_prep import build_finetune_dataset
    build_finetune_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShopSmart AI Entry Point")
    parser.add_argument("--ingest",   action="store_true", help="Ingest products into ChromaDB")
    parser.add_argument("--app",      action="store_true", help="Launch the Streamlit app")
    parser.add_argument("--finetune", action="store_true", help="Generate fine-tune JSONL dataset")
    args = parser.parse_args()

    if args.ingest:
        run_ingest()
    elif args.app:
        run_app()
    elif args.finetune:
        run_finetune()
    else:
        parser.print_help()
