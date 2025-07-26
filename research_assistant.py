import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import tempfile
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv


load_dotenv()


try:
    groq_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it to your app's secrets.")
    st.stop()  # Stop the app if the key is not found



@st.cache_resource
def get_llm_embedding():
    """Caches the LLM and Embedding models to avoid re-loading on every rerun."""
    llm = ChatGroq(groq_api_key=groq_key, model='llama3-70b-8192', temperature=1.0)
    embeddings = HuggingFaceEndpointEmbeddings(model="Qwen/Qwen3-Embedding-8B", task='feature-extraction')
    return llm, embeddings


llm, embedding = get_llm_embedding()


#Custom Tool for local pdf
@tool
def process_local_pdf(query: str) -> str:
    """Processes an uploaded PDF research paper, extracts text and answers based on the questions asked.
    This tool assumes a PDF has been previously uploaded and its path is available in Streamlit's session state
    under 'temp_pdf_path'. Use this tool when the user asks a question about an uploaded document.
    Args:
        query (str): The question to answer based on the PDF content.
    Returns:
        str: The answer to the query based on the PDF, or an error message.
    """
    if "temp_pdf_path" not in st.session_state or not os.path.exists(st.session_state.temp_pdf_path):
        return "No PDF file has been uploaded or found for analysis. Please upload a PDF first."

    pdf_path = st.session_state.temp_pdf_path

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        splitted_docs = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(splitted_docs, embedding)
        retriever = vectorstore.as_retriever()


        prompt_template = ChatPromptTemplate.from_template("""You are a very helpful research assistant. Answer the user's question based on the provided context. If you cannot find the answer in the context, say "I cannot find the answer in the provided PDF."
            Context: {context}
            Question: {input}""")

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        response_chain = create_retrieval_chain(retriever, document_chain)

        # The retrieval chain expects 'input' as the query
        result = response_chain.invoke({"input": query})
        return result['answer']
    except Exception as e:
        return f"An error occurred while processing the PDF: {e}"


arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [process_local_pdf, arxiv_tool, wikipedia_tool]


# This is a standard ReAct prompt present in Langchain
prompt = hub.pull("hwchase17/react")


agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# UI (Streamlit)
st.set_page_config(page_title="Research Assistant AI", layout="centered")
st.title(" Research Assistant AI")
st.markdown("Ask questions about research papers, general knowledge, or upload your own PDF!")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# upload a PDF
st.header("Upload a PDF for Analysis")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_pdf_path = tmp_file.name

    st.session_state.temp_pdf_path = temp_pdf_path

    st.success(f"PDF '{uploaded_file.name}' uploaded successfully!")
    st.info(f"You can now ask questions related to '{uploaded_file.name}'.")
else:

    if "temp_pdf_path" in st.session_state and os.path.exists(st.session_state.temp_pdf_path):
        os.remove(st.session_state.temp_pdf_path)
    if "temp_pdf_path" in st.session_state:
        del st.session_state.temp_pdf_path


if query := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_content = ""
            try:

                # The 'process_local_pdf' tool itself will internally fetch 'temp_pdf_path' from st.session_state
                agent_response = agent_executor.invoke({"input": query})
                response_content = agent_response["output"]

            except Exception as e:
                response_content = f"An error occurred with the agent: {e}"
                st.error(response_content)

            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})