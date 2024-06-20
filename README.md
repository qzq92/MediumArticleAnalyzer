# Medium Article and PDF File analyzer with LangChain and Retrieval Augmented Generation (RAG) Framework
This is a simple exploration of the application of RAG applied on example *Medium* and PDF articles downloaded online. 

In particular, a LCEL code equivalent implementation for RAG is included under *main_pdf_file.py* as a alternative to the use of libraries involving *create_stuff_documents_chain* and *create_retrieval_chain* which is not covered by the course content.

## API used

Please note that the following API requires the use of API key to work and is not free.
- *Langchain's ChatOpenAi* to chat with OpenAI's GPT-3.5-Turbo model (https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html).

Other APIs that were used under Free tier
- PineCone VectorStore
- LangSmith

## Environment file to edit
Please create an *.env* file with the following parameters. 

The *TEXT_FILEPATH* and *PDF_FILEPATH* are variables referencing to sample *Medium* and *pdf* articles whcih are to be downloaded from online as part of RAG (which sample articles are provided under *files* subfolder), while *PINECONE_INDEX* refers to the name of Pinecone vector database created to store embedded document chunks. The use of *OPENAI_API_KEY* is to facilitate the llm models offered by OpenAI (Example: gpt-3.5-turbo), while *HUGGINGFACEHUB_API_TOKEN* is used for accessing LLM models offered by HuggingFace Platform.
```
OPENAI_API_KEY = <YOUR API KEY>
PINECONE_INDEX = <CREATED PINECONE INDEX>
TEXT_FILEPATH = "files/mediumblog1.txt"
PDF_FILEPATH = "files/ReAct.pdf"
HUGGINGFACEHUB_API_TOKEN = <YOUR API KEY>

# Following are optional if you are not using LangSmith for tracking llm utilisation related metrics
LANGCHAIN_API_KEY = <YOUR API KEY>
LANGCHAIN_TRACING_V2 = true
LANGCHAIN_PROJECT = <NAME FOR YOUR PROJECT>
```

For more information on Langsmith, refer to https://www.langchain.com/langsmith

## Important note on the use of Pinecone VectorStore for Document embedding
The index dimensions for this repo is 1536 which is based on OpenAI default embedding model "text-embedding-ada-002". In the event that other embedding models are to be used, a new Pinecone index with compatible dimension is to be created.

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

## Difference in codebase from Udemy's course content

Specifically for *main_pdf_file.py*, differences include the following:
1) the use of vector store embedding was changed from OpenAIEmbedding to HuggingFaceEmbeddings;
```
    ## Old code involving OpenAI service
    #embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    

    # Use of huggingface's Google t5-base as example
    embedding_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs={'device':'cpu'}
    )
```

2) Use of LCEL declaration for LangChain declaration.
```
    llm_chatopenai = ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        temperature=0
    )
    print()
    print("LCEL implementation with retrieval > chat prompt > LLM")
    print("----------------------------")
    # For retrieval dict, you need to identify the variables in the prompt. Contains variables context/input
    retrieval = {
        "context": vectorstore.as_retriever() | format_docs,
        "input": RunnablePassthrough(),
    }

    # Feed dictionary to retrieval qa_chat_prompt followed by LLM to parse.
    rag_pdf_chain_openai = (
        retrieval | retrieval_qa_chat_prompt | llm_chatopenai
    )
```

3. Experimenting with RetrievalQA as chain and custom prompts:
```
    qa = RetrievalQA.from_chain_type(
        llm=llm_chatopenai,
        chain_type="refine",
        retriever=retriever,
    )
    result = qa.invoke({"query": query})
    print(result["result"])

    ...


    # Create Prompt
    template = dedent("""\
        Answer any use questions based solely on the context below:
        
        <context>
        
        {context}

        </context>
        
        Question: {question}
        Answer:"""
    )
    prompt = PromptTemplate.from_template(template)

    qa = RetrievalQA.from_chain_type(
        llm=llm_chatopenai,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},  # The prompt is added here
    )
```

4. Experimenting with HuggingFaceEndpoints package
```
    callbacks = [StreamingStdOutCallbackHandler()]
    # initialize Hub LLM with model. Note the model size
    hub_llm = HuggingFaceEndpoint(
        repo_id = os.environ.get("HUGGINGFACEHUB_LLM_QA_MODEL_NAME"),
        temperature = 0.01,
        top_k = 5,
        huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        max_new_tokens = int(os.environ.get("HUGGINGFACEHUB_LLM_QA_MODEL_MAX_TOKEN")),
        callbacks = callbacks
    )

    qa = RetrievalQA.from_chain_type(
        llm=hub_llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},  # The prompt is added here
    )
    result = qa.invoke({"query": query})
    print(result["result"])
    print()
```

## Programming languages/tools involved

- Python
- Langchain
    - ChatOpenAI
    - Hubs for Question-Answering RAG templates: "langchain-ai/retrieval-qa-chat"
    - RAG related chains: create_stuff_documents_chain, create_retrieval_chain
    - TextSplitters
    - VectorStores: Pinecone, FAISS
    - HuggingFaceEmbeddings, HuggingFaceHub, HuggingFaceEndPoints

## References for Huggingface as part of exploration

- https://huggingface.co/docs/transformers/en/tasks/question_answering
- Differences with Langchain Retrieval (.from_llm and without): https://stackoverflow.com/questions/77033163/whats-the-difference-about-using-langchains-retrieval-with-from-llm-or-defini
- Prompt Versioning: https://docs.smith.langchain.com/old/cookbook/hub-examples/retrieval-qa-chain-versioned
- HuggingFace Endpoints: https://python.langchain.com/v0.1/docs/integrations/llms/huggingface_endpoint/

## Acknowledgement and Credits

The codebase developed are in reference to *Section 5: The GIST of RAG-Embeddings, Vector Databases and Retrieval* of Udemy course titled "LangChain- Develop LLM powered applications with LangChain" available via https://www.udemy.com/course/langchain.