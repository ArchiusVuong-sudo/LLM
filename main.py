import os
import asyncio

import streamlit as st
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import load_tools
from langchain_core.documents import Document

from core.document_loader import DocumentLoader
from core.document_summarizer import DocumentSummarizer
from core.retriever_builder import VectorStore, GraphStore

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver

from core.get_log import setup_logger


# Setup log
logger = setup_logger('main_file', './log/main.log')

async def get_summary(llm: ChatOpenAI, doc: Document):
    summarizer = DocumentSummarizer(
        llm,
        docs
    )
    
    # Run the summarization application
    app = summarizer.compile_app()

    async for step in app.astream(
        {"contents": [doc.page_content]},
        {"recursion_limit": 10},
    ):
        logger.info(step.keys())

    return step


if __name__ == '__main__':
    load_dotenv()
    llm = ChatOpenAI(model='gpt-4o')

    vector_store = VectorStore()
    graph_store = GraphStore(llm)

    @tool
    def hybrid_retriever(question: str):
        """
        Retrieves information using a hybrid approach combining graph-based and vector-based retrieval methods.

        Args:
            question (str): The query or question to retrieve information for.

        Returns:
            str: A formatted string that includes both graph-based data and vector-based document content.

        Description:
            This function first retrieves relevant data from a graph database using `graph_store.graph_retriever()`
            and from a vector database by invoking `vector_retriever.invoke()`. The results from both sources are 
            combined into a single formatted string. Graph data is displayed as-is, while vector-based data 
            is joined with a '#Document ' separator between entries.
        """
        graph_data = graph_store.graph_retriever(question)
        vector_data = [el.page_content for el in vector_retriever.invoke(question)]
        final_data = f"""Graph data:
    {graph_data}
    vector data:
    {"#Document ". join(vector_data)}
        """
        return final_data
    
    tools = load_tools(["google-serper"]) + [hybrid_retriever]

    with st.sidebar:
        st.title("📝 File Explorer")
        st.success('API key already provided!', icon='✅')

        uploaded_file = st.file_uploader("Upload document onto the knowledge base", type=("pdf"), accept_multiple_files=False)

        add_to_graph = st.checkbox("Add this document to Graph Knowledge Base")

        if uploaded_file:
            with st.status("Downloading data..."):
                st.write("Uploading the file...")
                file_path = os.path.join(os.environ.get("DOCUMENT_PATH"), uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.write("Extracting the texts...")
                st.write("Explaining the images...")
                document_loader = DocumentLoader()
                docs = document_loader.load(file_path)
                
                st.write("Adding to Knowledge Base...")
                
                if add_to_graph:
                    graph_store.add_doc(docs)
                    graph_store.create_index()
                
                vector_store.add_doc(docs)
                vector_retriever = vector_store.get_retriever()
                
                st.write("Summarizing the document...")
                summary = asyncio.run(get_summary(llm, docs[0]))
            
            st.toast("Finished Processing the document!", icon='😍')

        "[![Open in GitHub Repo](https://github.com/codespaces/badge.svg)](https://github.com/ArchiusVuong-sudo/LLM)"

    st.title("🔎 Book Recommender Chatbot")

    """
    In this demo, we leverage LangChain 🤝 Streamlit Agent for an automated LLM Recommender System.
    """

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello World"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with ConnectionPool(
                # Example configuration
                conninfo=os.environ['DB_URI'],
                max_size=20,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                },
            ) as pool:
                checkpointer = PostgresSaver(pool)
                checkpointer.setup()

                graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
                config = {"configurable": {"thread_id": "1"}}
                res = graph.invoke({"messages": [("human", prompt)]}, config)
                response = res['messages'][-1].content
                checkpoint = checkpointer.get(config)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
