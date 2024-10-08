# flake8: noqa: E501
import os
from typing import List
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain.storage import LocalFileStore
from chainlit.types import AskFileResponse
from langchain.embeddings import CacheBackedEmbeddings
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import QdrantClient
import chainlit as cl
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.config import RunnableConfig
from dotenv import load_dotenv
import uuid

load_dotenv()

### Global Section ###
"""
GLOBAL CODE HERE
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

rag_system_prompt_template = """\
You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existance of context.
"""

rag_message_list = [
    {"role": "system", "content": rag_system_prompt_template},
]

rag_user_prompt_template = """\
Question:
{question}
Context:
{context}
"""

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", rag_system_prompt_template), ("human", rag_user_prompt_template)]
)

chat_model = ChatOpenAI(model="gpt-4o-mini")


def process_file(file: AskFileResponse):
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tempfile:
        with open(tempfile.name, "wb") as f:
            f.write(file.content)

    Loader = PyMuPDFLoader

    loader = Loader(tempfile.name)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs


@cl.on_chat_start
async def on_chat_start():
    files = None

    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...",
    )
    await msg.send()

    docs = process_file(file)

    collection_name = f"pdf_to_parse_{uuid.uuid4()}"
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    core_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings, store, namespace=core_embeddings.model
    )
    vectorstore = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=cached_embedder
    )
    vectorstore.add_documents(docs)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    # Create a chain that uses the QDrant vector store
    # Parallelization: LCEL runnables are parallelized by default, allowing for efficient
    # execution of multiple steps in the chain simultaneously, improving overall performance.
    retrieval_augmented_qa_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | chat_prompt
        | chat_model
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", retrieval_augmented_qa_chain)


@cl.author_rename
def rename(orig_author: str):
    rename_dict = {
        "ChatOpenAI": "the Generator...",
        "VectorStoreRetriever": "the Retriever...",
    }
    return rename_dict.get(orig_author, orig_author)


### On Message Section ###
@cl.on_message
async def main(message: cl.Message):
    runnable = cl.user_session.get("chain")

    msg = cl.Message(content="")

    # Async method: Using astream allows for asynchronous streaming of the response,
    # improving responsiveness and user experience by showing partial results as they become available.
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk.content)

    await msg.send()
