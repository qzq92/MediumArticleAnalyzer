import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
load_dotenv()


def format_docs(docs) -> str:
    """Formatter function that joins documents into single string delimited by 2 newlines.

    Args:
        docs (List): List of Documents to be processed

    Returns:
        str: Newline joined documents
    """
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retreiving...")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    llm = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print("Query without RAG, pure llm inference with pretrained knowledge.")
    query = "What is Pinecone in machine learning?"
    # Define chain with LCEL
    chain = PromptTemplate.from_template(template=query) | llm
    # Invoke with no variable since query as no query variable encoded.
    result = chain.invoke(input={})
    print(result.content)
    print()
    print("-------------------------")
    print("With RAG")
    # Inititalise vectorstore which RAG architecture is used. Embeddings must be consistent
    vectorstore = PineconeVectorStore(
        index_name = os.environ["PINECONE_INDEX"],
        embedding = embeddings
    )

    # Use open sourced chat prompt as part of ReAct for QA use case. https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create Runnable chain by stuffing docs into llm prompt aided by promt and model
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    # Invoke and get result
    result = retrieval_chain.invoke(input={"input": query})
    print(result["answer"])

    print()
    print("-------------------------")
    print("With Custom Template")
    # Customised template instead of retrieval QA chat prompt template from langchain hub. Context will be formated according to our needs

    custom_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say I do not know and do not attempt to make up an answer. The answer should be as concise as possible, limited to three sentences at maximum. Always say thanks "thanks for asking!" at the end of the answer.
    
    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template=custom_template)
    # LCEL, output type will be determine on the final chain involved, which is AImessage
    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()} #Propagate question thru to LLM call
        | custom_rag_prompt
        | llm
    )

    # Invoke and get result
    custom_result = rag_chain.invoke(input=query)
    print(custom_result.content)
    print(type(custom_result))