import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()


def get_vectorstore_from_url(url):
    # load url
    loader = WebBaseLoader(url)
    documents = loader.load()
    # split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)
    # vector store
    vector_store = Chroma.from_documents(
        document_chunks,
        GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"],
            task_type="retrieval_document",
        ),  # we can HuggingFaceEmbeddings
    )

    return vector_store


def get_context_retriever_chain(vector_store: Chroma):
    llm = ChatGoogleGenerativeAI(
        google_api_key=os.environ["GOOGLE_API_KEY"],
        model="gemini-1.5-flash",
        temperature=0.1,
    )
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )
    chain = create_history_aware_retriever(llm, retriever, prompt)

    return chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatGoogleGenerativeAI(
        google_api_key=os.environ["GOOGLE_API_KEY"],
        model="gemini-1.5-flash",
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff)


# App config
st.set_page_config(page_title="Ask question from web page")
st.title("Ask question from web page")


# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("website url")

if website_url is None or website_url == "":
    st.info("Please provide a website url!")
else:
    # initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="How can i help you today?")]

    if "vectorDB" not in st.session_state:
        st.session_state.vectorDB = get_vectorstore_from_url(website_url)

    chain = get_context_retriever_chain(st.session_state.vectorDB)
    conversation = get_conversational_rag_chain(chain)
    # user input
    user_input = st.chat_input("Type message here....")
    if user_input is not None and user_input != "":
        response = conversation.invoke(
            {"chat_history": st.session_state.chat_history, "input": user_input}
        )
        # response = get_response(user_input)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response["answer"]))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
