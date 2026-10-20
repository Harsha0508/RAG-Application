# RAG OCR System

A local document question-answering app that extracts text from PDFs and images, indexes it, and returns relevant passages with source-page references. It includes a FastAPI backend and a Streamlit chat interface.

## Features

- Ingest PDF, PNG, JPG, JPEG, and WEBP files.
- Extract selectable PDF text directly and use RapidOCR for scanned pages and images.
- Split extracted text into sentence-aware overlapping chunks.
- Create local text embeddings with scikit-learn's `HashingVectorizer` and store vectors in an embedded Qdrant database.
- Ask questions through the chat UI and view source documents and page numbers.
- Optionally generate answers using the Groq API. Without a Groq key, the app returns relevant excerpts instead.
- Automatically process supported files placed in the `ingestion/` folder.

## Requirements

- Python 3.10 or newer.
- Windows PowerShell (commands below use PowerShell syntax).
- A Groq API key only if you want LLM-generated answers; retrieval and context-only responses work without one.

## Setup

From the project root, create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root with the chunking settings:

```dotenv
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

These settings are used when splitting extracted text. To enable Groq-powered answers, add your API key and a Groq-supported model name:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_groq_model
```

The Groq settings are optional. Keep the API key private and do not commit `.env`.

## Run

Start the backend in one terminal:

```powershell
python main.py
```

Start the Streamlit frontend in a second terminal from the project root:

```powershell
python -m streamlit run frontend/app.py
```

Open the Streamlit URL shown in the terminal (usually `http://localhost:8501`). The backend runs at `http://localhost:8000`; its interactive API documentation is available at `http://localhost:8000/docs`.

## Use

1. Upload a supported document using the sidebar, or copy it into the top-level `ingestion/` folder.
2. Wait for processing to finish. Files placed in `ingestion/` are detected automatically while the backend is running.
3. Ask a question in the chat. The sidebar shows indexed documents, system status, and the number of context chunks to retrieve.
4. Use the delete button beside an indexed document to remove it from the index.

The upload endpoint accepts files up to 100 MB. Successful files detected in `ingestion/` are moved to `ingestion/processed/`; files that fail processing are moved to `ingestion/failed/`. Uploaded files are processed by the API and their source copies are not stored in those folders.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/chat` | Retrieve relevant chunks and answer a question. Send JSON with `query` and optional `top_k` (1–20). |
| `POST` | `/api/ingest` | Upload and index a PDF or supported image as multipart form field `file`. |
| `GET` | `/api/status` | Get index counts, embedding model, LLM availability, and file-watcher status. |
| `GET` | `/api/health` | Check backend health and version. |
| `GET` | `/api/documents` | List indexed documents. |
| `DELETE` | `/api/documents/{doc_name}` | Remove a document and its chunks from the index. |

## Project data

- `ingestion/` — watched input folder; `processed/` and `failed/` contain files handled by the watcher.
- `data/qdrant/` — local Qdrant vector database.
- `data/chunks.sqlite` — local SQLite storage for chunk text and metadata.

These local data files contain the indexed document content and should be handled accordingly when sharing or backing up the project.
