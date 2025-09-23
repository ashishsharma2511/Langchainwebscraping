import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

def webscrape(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = "\n".join([para.get_text() for para in paragraphs])
    return text

url = "https://en.wikipedia.org/wiki/LangChain"  # change to any URL you want
data = webscrape(url)
print(f"Scraped {len(data)} characters from {url}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.create_documents([data])

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)

from langchain.llms import Ollama
llm = Ollama(model="mistral")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Chat loop 
print("\n=== Chat with the website! Type 'exit' to quit ===\n")
while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    result = qa_chain({"query": question})
    print("\nBot:", result["result"], "\n")

