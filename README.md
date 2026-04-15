# AI Knowledge Assistant (RAG + Confidence Scoring)

A lightweight AI assistant that answers questions based on uploaded documents using Retrieval-Augmented Generation (RAG).

## Features

-  Document-based question answering (PDF, text)
-  Semantic search with vector database (Chroma)
-  LLM-powered answers (OpenAI)
-  Source transparency with chunk previews (hover)
-  Confidence score based on evidence support (LLM Judge)

## Tech Stack

- Python
- LangChain
- OpenAI API
- ChromaDB
- Gradio

## How it works

1. Documents are split into chunks and embedded
2. Relevant chunks are retrieved for each query
3. LLM generates an answer based on retrieved context
4. A second LLM evaluates how well the answer is supported by the sources

## Use Cases

- Knowledge management
- Document analysis
- Assistive systems in regulated environments (e.g. Pharma, GxP)

## Run locally

```bash
pip install -r requirements.txt
python app.py
