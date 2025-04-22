from langchain_ollama import ChatOllama
import chainlit as cl
from typing import cast
from vector import extract_pdf_text,split_text,add_to_db
from langchain.memory import ConversationBufferMemory
from main import chain
from vector import retriever


memory = ConversationBufferMemory(memory_key="chat_history")

@cl.on_chat_start
async def start():
    # Initialize conversation history
    memory.clear()
    await cl.Message(content="Welcome to the RAG Chatbot! Please upload your PDF files.").send()
    
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    with open(file.path, "rb") as f:
        file_content = f.read()

    documents = extract_pdf_text(file_content)
    chunks = split_text(documents)
    add_to_db(chunks)

    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()



@cl.on_message
async def main(message: cl.Message):
    # Retrieve the conversation history
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    # Retrieve relevant documents based on the user's message
    context= retriever.invoke(message.content)
    
    # Generate a response using the LLM
    response = chain.invoke({"content": context, "question": message.content})
    
    # Update memory with the new message and response
    memory.save_context({"input": message.content}, {"output": response})
    
    # Send the response back to the user
    await cl.Message(content=response).send()

    




    
    

