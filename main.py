from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader


model = OllamaLLM(model = "llama3.2")

template = """
You are an expert in whatever information the user uploads.

Provide the most reasonable response based on their question.

Here is the relevant content: {content}

Here is the question to ansewer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model



