import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

def webscrape(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    elements = soup.find_all(["p", "li", "h1", "h2", "h3"])
    text = "\n".join(el.get_text() for el in elements)
    return text.strip()


url = "https://en.wikipedia.org/wiki/Sachin_Tendulkar"  # change to any URL you want
data = webscrape(url)
print(f"Scraped {len(data)} characters from {url}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.create_documents([data])

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)


from langchain.llms import Ollama
llm = Ollama(model="gemma:2b")

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

