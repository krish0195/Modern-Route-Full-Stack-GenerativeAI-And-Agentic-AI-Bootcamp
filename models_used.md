llm_embeddings_rag_cheatsheet.py

✅ LLMs, Embeddings, Tools, RAG Chain — Organized with Usage Comments

-------------------------------------------------------------------------------------

🔹 1. ENVIRONMENT SETUP (Optional)

-------------------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv()  # ✅ Loads .env variables like API keys into the environment

import os
os.environ["TAVILY_API_KEY"] = "your-api-key"  # Needed for Tavily search tool

-------------------------------------------------------------------------------------

🔹 2. GOOGLE GEMINI MODELS

-------------------------------------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

✅ For using Gemini 1.5 Pro

llm_gemini_pro = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

### text to text : free tier available

Model: gemini-2.5-flash (free)

Model: gemini-2.5-pro(free)

Model: gemini-2.5-flash-lite(free)

Model: gemini-2.5-flash-native-audio-preview(free)

Model: gemini-2.5-flash-preview-tts(free)

Model: gemini-embedding-2(free)

Model: gemini-embedding-001(free)

✅ For Gemini 2.0 Flash or latest Flash

llm_gemini_flash = ChatGoogleGenerativeAI(model="gemini/gemini-2.0-flash")
llm_gemini_latest = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

✅ Google Embeddings

google_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

Length of output may be 384, 768, or 1024

-------------------------------------------------------------------------------------

🔹 3. GROQ MODELS

-------------------------------------------------------------------------------------

from langchain_groq import ChatGroq

✅ For DeepSeek distillation on LLaMA-70B

groq_model = ChatGroq(model="deepseek-r1-distill-llama-70b")

groq_model = ChatGroq(model="qwen/qwen3-32b")

llm = ChatGroq(model="qwen/qwen3-32b")

model="meta-llama/llama-4-scout-17b-16e-instruct"

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

✅ Used in RAG chain (below)

llm_rag = ChatGroq(model="llama3-70b-8192")


-------------------------------------------------------------------------------------

🔹 4. HUGGINGFACE MODELS & EMBEDDINGS

-------------------------------------------------------------------------------------

from langchain_huggingface import (
ChatHuggingFace,
HuggingFacePipeline,
HuggingFaceEndpoint,
HuggingFaceEmbeddings
)

✅ Option 1: Using HuggingFacePipeline

llm_hf_pipeline = HuggingFacePipeline.from_model_id(
model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
task='text-generation',
pipeline_kwargs=dict(temperature=0.5, max_new_tokens=100)
)

✅ Option 1: Using HuggingFacePipeline


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512
)


✅ Option 2: Direct Endpoint

llm_hf_endpoint = HuggingFaceEndpoint(
repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
task="text-generation"
)

✅ Embeddings from HuggingFace

hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

-------------------------------------------------------------------------------------

🔹 5. OPENAI MODELS & EMBEDDINGS

-------------------------------------------------------------------------------------

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

✅ GPT-4 model

llm_openai = ChatOpenAI(model="gpt-4")

llm = ChatOpenAI(model="gpt-5-nano")



✅ OpenAI Embedding Variants

openai_embedding_large = OpenAIEmbeddings(model="text-embedding-3-large")
openai_embedding_small = OpenAIEmbeddings(model="text-embedding-3-small")
openai_embedding_64 = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=64)


-------------------------------------------------------------------------------------

🔹 6. Use Anthropic/Claude
-------------------------------------------------------------------------------------
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0
)


-------------------------------------------------------------------------------------

🔹 6. Openrouter
-------------------------------------------------------------------------------------

from langchain_openrouter import ChatOpenRouter

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4.5",max_tokens=2048)





-------------------------------------------------------------------------------------

🔹 6. EXTERNAL TOOLS (PythonREPL, Tavily)

-------------------------------------------------------------------------------------

from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults

✅ Tavily search engine tool

tavily_tool = TavilySearchResults()
tavily_tool.invoke("What is GDP?")

✅ Python code execution tool

repl = PythonREPL()
repl.run("print(2 * 5)")

-------------------------------------------------------------------------------------

🔹 7. RAG CHAIN (Using ChatGroq + HuggingFace Embeddings)

-------------------------------------------------------------------------------------

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from ragas.llms import LangchainLLMWrapper  # Optional wrapper

✅ Define question-answering prompt

rag_prompt_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use two sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:
"""

✅ Convert template to LangChain prompt

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

✅ Define embedding model for document retrieval

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

✅ Define RAG chain (retriever must be defined earlier)

rag_chain = (
{"context": retriever, "question": RunnablePassthrough()}
| rag_prompt
| llm_rag
| StrOutputParser()
)

✅ Sample questions

questions = [
"What is the current status of Japan's industrial production?",
"How have retail sales in Japan performed recently?",
"What did the Bank of Japan say about the economic outlook?",
"What has been the performance of the Nikkei 225 index?",
"How has the Japanese yen moved in recent times?",
"What are some challenges Japan's economy is still facing?"
]

✅ Run the chain on a sample question

result = rag_chain.invoke(questions[1])
print(result)

