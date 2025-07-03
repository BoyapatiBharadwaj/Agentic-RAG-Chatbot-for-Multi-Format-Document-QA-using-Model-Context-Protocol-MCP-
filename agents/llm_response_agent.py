# agents/llm_response_agent.py

from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

class LLMResponseAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    def generate_answer(self, query, retrieved_docs):
        chain = load_qa_chain(self.llm, chain_type="stuff")
        result = chain.run(input_documents=retrieved_docs, question=query)
        return result
