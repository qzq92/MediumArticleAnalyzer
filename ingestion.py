import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    # Step 1: Load data
    loader = TextLoader(file_path=os.environ.get("TEXT_FILEPATH"), autodetect_encoding=True, encoding="utf-8")
    document = loader.load() # Langchain docs

    # Step 2: Data split
    print("Splitting...")
    # Heuristics for chunk size decision (Relevant context/chunk -> Better result). Chunking is still preferred as sending big chunks to LLM may result in worst answer due to irrelevant content. Chunk overlap set to 0 to avoid overlap data and simplicity.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # Expect uneven chunks due to delimiters involved. Some may exceed specified chunks.
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    # Step 3: Embed
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Ingesting...")
    # Do not run multiple times to avoid duplications
    PineconeVectorStore.from_documents(
        index_name=os.environ.get("PINECONE_INDEX"),
        documents=texts,
        embedding=embeddings,
    )
    print("finish")