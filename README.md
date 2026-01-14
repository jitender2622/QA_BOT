# 📄 RAG-based PDF QA Bot with Gemini & LangChain

This project is a **Retrieval-Augmented Generation (RAG)** application built using Python. It allows users to upload a PDF document and ask questions related to its content. The application uses **Google's Gemini** model to generate accurate answers based on the context retrieved from the document.

## 🚀 Features
* **PDF Document Loading:** Uses `PyMuPDFLoader` to extract text from uploaded PDF files.
* **Text Chunking:** Splits large documents into manageable chunks using `RecursiveCharacterTextSplitter`.
* **Vector Embeddings:** Utilizes **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) to convert text into vector representations.
* **Vector Database:** Stores and retrieves document chunks using **ChromaDB** for efficient semantic search.
* **LLM Integration:** Powered by **Google Gemini 1.5 Flash** (via `LangChain`) for fast and context-aware responses.
* **User Interface:** Simple and interactive web interface built with **Gradio**.

## 🛠️ Tech Stack
* **Language:** Python
* **Orchestration:** LangChain
* **LLM:** Google Generative AI (Gemini Flash)
* **Embeddings:** HuggingFace
* **Vector Store:** ChromaDB
* **UI Framework:** Gradio

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/gemini-pdf-qa-bot.git](https://github.com/your-username/gemini-pdf-qa-bot.git)
    cd gemini-pdf-qa-bot
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have `langchain`, `gradio`, `chromadb`, `langchain-google-genai`, `langchain-huggingface`, and `pymupdf` installed)*

3.  **Set up your Google API Key:**
    * Get your API key from [Google AI Studio](https://aistudio.google.com/).
    * Set it as an environment variable or create a `.env` file (recommended).

## 🏃‍♂️ Usage

Run the application locally:

```bash
python app.py
