# Medium Article and PDF File analyzer with LangChain and Retrieval Augmented Generation (RAG) Framework
This is a simple exploration of the application of RAG applied on example *Medium* and PDF articles downloaded online. 

In particular, a LCEL code equivalent implementation for RAG is included under *main_pdf_file.py* as a alternative to the use of libraries involving *create_stuff_documents_chain* and *create_retrieval_chain* which is not covered by the course content.

## API used

Please note that the following API requires the use of API key to work and are not free.
- *Langchain's ChatOpenAi* to chat with OpenAI's GPT-3.5-Turbo model (https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html).

## Environment file to edit
Please create an *.env* file with the following parameters. 

The *TEXT_FILEPATH* and *PDF_FILEPATH* are variables referencing to sample *Medium* and *pdf* articles whcih are to be downloaded from online as part of RAG (which sample articles are provided under *files* subfolder), while *PINECONE_INDEX* refers to the name of Pinecone vector database created to store embedded document chunks. The use of OPENAI_API_KEY is to facilitate the llm models offered by OpenAI (Example: gpt-3.5-turbo)
```
OPENAI_API_KEY = <YOUR API KEY>
PINECONE_INDEX = <CREATED PINECONE INDEX>
TEXT_FILEPATH = "files/mediumblog1.txt"
PDF_FILEPATH = "files/ReAct.pdf"

# Optional if you are not using LangSmith for tracking llm utilisation related metrics
LANGCHAIN_API_KEY = <YOUR API KEY>
LANGCHAIN_TRACING_V2 = true
LANGCHAIN_PROJECT = <NAME FOR YOUR PROJECT>
```

For more information on Langsmith, refer to https://www.langchain.com/langsmith

## Installation and execution
Please use Anaconda distribution to install the necessary libraries with the following command

```
conda env create -f environment.yml
```

Upon installation and environment exectuion, run either of the command to execute relevant main python script on analysing sample PDF and medium article respectively.
```
python main_pdf_file.py

python main_medium_article.py
```

## Programming languages/tools involved
- Python
- Langchain
    - ChatOpenAI
    - Hubs for Question-Answering RAG templates: "langchain-ai/retrieval-qa-chat"
    - RAG related chains: create_stuff_documents_chain, create_retrieval_chain
    - TextSplitters
    - VectorStores: Pinecone, FAISS

## Acknowledgement and Credits

The codebase developed are in reference to *Section 5: The GIST of RAG-Embeddings, Vector Databases and Retrieval* of Udemy course titled "LangChain- Develop LLM powered applications with LangChain" available via https://www.udemy.com/course/langchain.