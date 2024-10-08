# flake8: noqa ignore

from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
import numpy as np
from numpy.linalg import norm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.config import RunnableConfig
from dotenv import load_dotenv
import chainlit as cl
import os
import uuid

load_dotenv()

HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]

document_loader = TextLoader("data/paul-graham-to-kindle/paul_graham_essays.txt")
documents = document_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
split_documents = text_splitter.split_documents(documents)

hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=HF_EMBED_ENDPOINT,
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN,
)

if os.path.exists("./data/vectorstore/index.faiss"):
    vectorstore = FAISS.load_local(
        "./data/vectorstore",
        hf_embeddings,
    )
    hf_retriever = vectorstore.as_retriever()
    print("Loaded Vectorstore")
else:
    print("Indexing Files")
    for i in range(0, len(split_documents), 32):
        if i == 0:
            vectorstore = FAISS.from_documents(
                split_documents[i : i + 32], hf_embeddings
            )
            continue
        vectorstore.add_documents(split_documents[i : i + 32])

hf_retriever = vectorstore.as_retriever()

RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

hf_llm = HuggingFaceEndpoint(
    endpoint_url=f"{HF_LLM_ENDPOINT}",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HF_TOKEN,
)


@cl.on_chat_start
async def on_chat_start():
    lcel_rag_chain = (
        {"context": itemgetter("query") | hf_retriever, "query": itemgetter("query")}
        | rag_prompt
        | hf_llm
    )

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)
    await cl.Message(
        content="Hi! What questions do you have about Paul Graham's essays?"
    ).send()


@cl.author_rename
def rename(orig_author: str):
    rename_dict = {
        "ChatOpenAI": "the Generator...",
        "VectorStoreRetriever": "the Retriever...",
    }
    return rename_dict.get(orig_author, orig_author)


@cl.on_message
async def main(message: cl.Message):
    runnable = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
