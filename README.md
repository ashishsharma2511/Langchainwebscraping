# LangChain Web Scraping with Ollama

This project demonstrates how to use **LangChain** with **Ollama** to perform question answering (QA) over scraped website data.

## 🚀 Features
- Scrape content from websites
- Store and index scraped text using embeddings
- Use Ollama as the LLM backend (e.g., Gemma, LLaMA, Phi-3)
- Query the indexed data via LangChain QA chain

## 🛠️ Requirements
- Python 3.9+
- [Ollama](https://ollama.ai) installed and running locally
- The following Python libraries:
  - langchain
  - langchain-community
  - langchain-ollama
  - beautifulsoup4
  - requests

## ⚡ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/langchain-webscraping.git
   cd langchain-webscraping
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure **Ollama** is installed and running. You can verify with:
   ```bash
   ollama --version
   ```

5. Pull a model (example: Gemma 2B):
   ```bash
   ollama pull gemma:2b
   ```

## 📖 Usage

Run the script with:
```bash
python webscraping.py
```

When prompted, enter a question about the scraped website content.

### Example
```
You: hi
Bot: Hello! How can I help you with the scraped content?
```

## 🔧 Notes
- In **LangChain 0.1.0+**, the chain call has been updated.  
  Replace:
  ```python
  result = qa_chain({"query": question})
  ```
  with:
  ```python
  result = qa_chain.invoke({"query": question})
  ```

## 📜 License
This project is licensed under the MIT License.
