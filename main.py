from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from vector import clearDB


model = OllamaLLM(model = "llama3.2")

template = """
You are an expert of Data Structures and Algorithms in Python

Here is a textbook of all concepts: {content}

Here is the question to ansewr: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


while True:
    question = input("Ask your question (q to quit)")
    if question == "q":
        break
    content = retriever.invoke(question)
    result = chain.invoke({"content": [], "question": question})
    print(result)

clearDB()