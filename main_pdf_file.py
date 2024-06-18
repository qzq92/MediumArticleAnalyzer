import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
# Fast local iteration alternative to PineCone
from langchain_community.vectorstores import FAISS

def format_docs(docs) -> str:
    """Formatter function that joins documents into single string delimited by 2 newlines.

    Args:
        docs (List): List of Documents to be processed

    Returns:
        str: Newline joined documents
    """
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    load_dotenv()
    print("FAISS Vectorstore")
    pdf_path = os.environ.get("PDF_FILEPATH")
    # By default it will chunk it by page. May still be too large in context window
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    # Split further by character
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    # Embeddings and llm model
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
    llm = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Use FAISS Vector store to loaded chunkified docs into RAM.
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    vectorstore.save_local("faiss_index_react")

    # Load data
    new_vectorstore = FAISS.load_local(
        "faiss_index_react",
        embeddings=embeddings,
        allow_dangerous_deserialization=True #Allow deserialisation for trusted file.Otherwise it is not advised to. This is a feature to prevent any dangerous executions by default from a .pkl file
    )

    # Use open sourced chat prompt as part of ReAct for QA use case. https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(
        llm = llm, 
        prompt=retrieval_qa_chat_prompt)
    
    retrieval_chain = create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain)
    
    query = "Give me the gist of ReAct in 3 sentences"

    result = retrieval_chain.invoke({"input": query})
    print(result["answer"])
    print()
    print("LCEL equivalent")
    print("----------------------------")
    # For retrieval dict, you need to identify the variables in the prompt. Contains variables context/input
    retrieval = {
        "context": vectorstore.as_retriever() | format_docs,
        "input": RunnablePassthrough(),
    }

    # Feed dictionary to retrieval qa_chat_prompt followed by LLM to parse.
    rag_pdf_chain = (
        retrieval | retrieval_qa_chat_prompt | llm
    )
    # Invoke with require template input(chat prompt) and get result
    result = rag_pdf_chain.invoke(input=query)
    print(result.content)
