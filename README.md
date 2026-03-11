# Simple RAG Pipeline with LangChain, ChromaDB, Sentence Transformers, and Groq

This project is a beginner-friendly implementation of a simple
Retrieval-Augmented Generation (RAG) pipeline in Python.

The goal of this project is to learn how a RAG system works step by
step, starting from document ingestion and chunking, then creating
embeddings, storing them in a vector database, retrieving relevant
chunks for a user query, and finally generating an answer using an LLM
through Groq.

## What this project does

This project demonstrates the following pipeline:

1.  Load text files and PDF documents
2.  Split documents into smaller chunks
3.  Convert chunks into embeddings using a SentenceTransformer model
4.  Store embeddings and metadata in ChromaDB
5.  Retrieve the most relevant chunks for a query
6.  Send the retrieved context to a Groq-hosted LLM
7.  Generate a final answer based on the retrieved context

## Technologies used

-   Python
-   LangChain
-   LangChain Community
-   LangChain Text Splitters
-   Sentence Transformers
-   ChromaDB
-   PyPDF / PyMuPDF
-   Groq API
-   python-dotenv

## Project structure

``` bash
project-root/
├── data/
│   ├── text_files/
│   ├── pdf/
│   └── vector_store/
├── notebooks/
├── README.md
├── requirements.txt
└── .env
```

## Main Components

### Document Ingestion

The project loads documents from different sources including text files
and PDFs.\
LangChain loaders such as `TextLoader`, `DirectoryLoader`,
`PyPDFLoader`, and `PyMuPDFLoader` are used to read and structure
documents.

### Document Splitting

Documents are split into smaller chunks using:

`RecursiveCharacterTextSplitter`

Chunking improves retrieval performance because smaller pieces of text
are easier to match with user queries.

Typical configuration:

-   chunk size: 1000 characters
-   overlap: 200 characters

### Embedding Generation

Text chunks are converted into numerical vectors using the HuggingFace
model:

`sentence-transformers/all-MiniLM-L6-v2`

Embeddings allow the system to compare text semantically instead of
relying on exact keyword matches.

### Vector Storage

Embeddings and document metadata are stored in **ChromaDB**, a vector
database designed for AI applications.

The vector store persists locally so the embeddings do not need to be
regenerated every time the system runs.

### Retrieval

The `RAGRetriever` component converts the user query into an embedding
and searches the vector database for the most similar document chunks.

Similarity is measured using cosine similarity.

### Response Generation

The retrieved chunks are inserted into a prompt and sent to a
Groq-hosted LLM.

Model used in this project:

`llama-3.1-8b-instant`

The model generates an answer grounded in the retrieved document
context.

## Installation

### Clone the repository

``` bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### Create a virtual environment

Using uv:

``` bash
uv venv
source .venv/bin/activate
```

Using Python:

``` bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory:

    GROQ_API_KEY=your_api_key_here

## Example Usage

Example query using the simple RAG pipeline:

``` python
answer = rag_simple("What is attention mechanism?", rag_retriever, llm)
print(answer)
```

## Learning Goals

This project was created to practice and understand:

-   document ingestion
-   chunking strategies
-   embedding generation
-   vector databases
-   semantic retrieval
-   prompt construction
-   integrating LLM APIs

## Possible Improvements

Future improvements may include:

-   adding duplicate detection before inserting documents
-   adding hybrid search (vector + keyword)
-   supporting additional file formats
-   implementing reranking models
-   building a small web interface with Streamlit or Gradio
-   evaluating retrieval quality

## Author

Ali Jamali

This repository documents my learning journey in RAG systems and modern
AI tooling.

## License

This project is intended for educational purposes.