import streamlit as st
from vector import extract_pdf_text
from main import invoke_model
from vector import split_text
from vector import add_to_db

st.title("Enter a pdf to chat over")


uploaded_file = st.file_uploader("Enter a pdf file", type="pdf")

if uploaded_file is not None:

    with st.spinner("Extracting text..."):

        text = extract_pdf_text(uploaded_file)

        chunks = split_text(text)

        add_to_db(chunks)

        
    st.success("PDF file uploaded.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if user_input := st.chat_input("Hello! I can help you retrieve information from your PDF. What would you like to know?"):




    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.chat_history.append({"role": "user","content" :user_input})

    response = invoke_model(user_input)
    
    with st.chat_message("AI"):
        st.markdown(response)

    st.session_state.chat_history.append({"role": "agent","content": response})

            
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    






