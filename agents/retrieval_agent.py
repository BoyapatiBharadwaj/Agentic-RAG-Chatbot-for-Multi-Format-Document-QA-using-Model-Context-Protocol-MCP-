# agents/retrieval_agent.py

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

class RetrievalAgent:
    def __init__(self):
        self.vectorstore = None
        self.embedding_model = OpenAIEmbeddings()

    def build_vectorstore(self, documents):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(docs, self.embedding_model)

    def retrieve_context(self, query, k=5):
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=k)
