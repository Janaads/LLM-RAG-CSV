from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
import pandas as pd

def load_csv(file):
    """Loads content from the uploaded CSV file."""
    try:
        df = pd.read_csv(file)
        return df.to_string(index=False)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return ""

def get_response(user_query, chat_history, document_text):
    """Generates a response based on user query, chat history, and CSV content."""
    llm = ChatOllama(model='dolphin-mistral:latest')

    template = '''
        Welcome to the ChatBot powered by Ollama Mistral.
        '''

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
        "document_text": document_text
    })

# Streamlit app setup
st.set_page_config(page_title="LLM RAG CHATBOT")
st.title('Distributed UI CORE Exploitation Pattern Platform using cloud')

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="How can I help you?")]
if "document_text" not in st.session_state:
    st.session_state.document_text = ""

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    # Load CSV content
    st.session_state.document_text = load_csv(uploaded_file)

if st.session_state.document_text:
    st.write("CSV file loaded successfully. You can now ask questions based on its content.")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User query input
user_query = st.chat_input("Enter your message")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Generate and display response
    document_text = st.session_state.document_text if st.session_state.document_text else "No CSV file uploaded."
    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history, document_text))
    st.session_state.chat_history.append(AIMessage(content=response))
