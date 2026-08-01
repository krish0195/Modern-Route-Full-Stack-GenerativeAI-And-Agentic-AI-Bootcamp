langchain_graph_cheatsheet.py

✅ MASTER CHEATSHEET — LANGCHAIN + LANGGRAPH (Beginner Friendly)

------------------------------------------------------------------------------------

🔹 1. FOUNDATION IMPORTS (Types, Utilities)

------------------------------------------------------------------------------------

from typing import TypedDict, Annotated, Sequence
from typing_extensions import Literal
import operator

Use for all message types

from langchain_core.messages import (
AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
)

Use when creating structured outputs or schema

from pydantic import BaseModel, Field

------------------------------------------------------------------------------------

🔹 2. PROMPTS (Single-turn and Multi-turn Prompt Templates)

------------------------------------------------------------------------------------

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

🟢 Use PromptTemplate when asking single-turn questions with variables

🟢 Use ChatPromptTemplate when simulating system/human AI chat style

Example - Single Prompt

template = PromptTemplate(
template="Say hello to {name} in 5 different languages.",
input_variables=["name"]
)
prompt = template.invoke({"name": "Virat Kohli"})

Example - Chat-style Prompt

chat_template = ChatPromptTemplate.from_messages([
("system", "You are a helpful {domain} expert."),
("human", "Explain {topic} in simple terms.")
])
chat_prompt = chat_template.invoke({"domain": "medical", "topic": "malaria"})

------------------------------------------------------------------------------------

🔹 3. PARSERS (To convert LLM output)

------------------------------------------------------------------------------------

from langchain_core.output_parsers import (
StrOutputParser, JsonOutputParser, PydanticOutputParser
)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

🟢 Use StrOutputParser to extract raw text

🟢 Use JsonOutputParser when expecting JSON structure

🟢 Use PydanticOutputParser or StructuredOutputParser for validation + structure

parser = StrOutputParser()

------------------------------------------------------------------------------------

🔹 4. MESSAGE STATES (Conversation Memory / TypedDict for LangGraph)

------------------------------------------------------------------------------------

class AgentState(TypedDict):
messages: Annotated[Sequence[BaseMessage], operator.add]

class MessagesState(TypedDict):
messages: Annotated[list, operator.add]

class Router(TypedDict):
next: Literal['researcher', 'coder', 'FINISH']

🟢 Use AgentState or MessagesState to persist messages between steps in LangGraph

🟢 Use Router when defining conditional routing between agents

------------------------------------------------------------------------------------

🔹 5. CHAINS (Sequential, Parallel, Custom Logic)

------------------------------------------------------------------------------------

from langchain.schema.runnable import RunnableParallel
from langchain_core.runnables import RunnablePassthrough

🟢 Use Sequential Chain when steps depend on each other

🟢 Use Parallel Chain when steps can run independently and merge afterward

Example: Sequential Chain

rag_chain = prompt | openai_model | parser

rag_chain.invoke("question")

Example: Parallel + Merge

parallel_chain = RunnableParallel({
"summary": prompt1 | openai_model | parser,
"qa": prompt2 | openai_model | parser
})
merge_chain = prompt3 | openai_model | parser
final_chain = parallel_chain | merge_chain

------------------------------------------------------------------------------------

🔹 6. LANGGRAPH (Graph-based Agent Workflows)

------------------------------------------------------------------------------------

from langgraph.graph import Graph, StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

🟢 Use Graph or StateGraph to define multi-step LLM workflows

🟢 Define state (like AgentState) and pass between nodes using commands

workflow = StateGraph(MessagesState)
workflow.add_node("start", some_function)
workflow.set_entry_point("start")
workflow.set_finish_point("end")

------------------------------------------------------------------------------------

🔹 7. TOOL USAGE (External Tool Calls by LLM)

------------------------------------------------------------------------------------

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL  # for executing code

🟢 Use @tool decorator to register functions callable by the LLM

@tool
def multiply(a: int, b: int) -> int:
"""Multiplies two integers."""
return a * b

Tool Binding Example

ai_msg = openai_model.bind_tools([multiply]).invoke(messages)

------------------------------------------------------------------------------------

🔹 8. DOCUMENT HANDLING & VECTORS (RAG Setup)

------------------------------------------------------------------------------------

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import (
CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter
)

🟢 Use TextLoader or DirectoryLoader to load files

🟢 Use RecursiveCharacterTextSplitter to chunk documents intelligently

🟢 Use Chroma or FAISS to store and retrieve embeddings

------------------------------------------------------------------------------------

🔹 9. RAG (Retrieval-Augmented Generation using ChatGroq + Embeddings)

------------------------------------------------------------------------------------

from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

🔸 Setup LLM + Embedding model

llm = ChatGroq(model="llama3-70b-8192")
embedding_model = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)

🔸 RAG Prompt Template

template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use two sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

🔸 RAG Chain

rag_chain = (
{"context": retriever, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)

🔸 Test RAG Chain

questions = [
"What is the current status of Japan's industrial production?",
"How have retail sales in Japan performed recently?",
]
response = rag_chain.invoke(questions[1])
print(response)



def format_docs(docs):
    return "nn".join([doc.page_content for doc in docs])

#RunnablePassthrough=Question on run time

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
------------------------------------------------------------------------------------

🔹 10. FULL EXAMPLES

------------------------------------------------------------------------------------

🟢 Chain with brief response

message = AgentState["messages"]
question = message[-1]
response = llm.invoke(f"Brief answer only: {question}")
AgentState["messages"].append(response.content)

🟢 Classifier Example

classification_template = """
Classify into: [Japan, Not Related]
User Query: {question}
{format_instructions}
"""
classifier_prompt = PromptTemplate(
template=classification_template,
input_variables=["question"],
partial_variables={"format_instructions": parser.get_format_instructions()}
)
classification_chain = classifier_prompt | llm | parser
classification_response = classification_chain.invoke({
"question": "Is Tokyo part of Japan?",
"format_instructions": parser.get_format_instructions()
})

🟢 Tool + AI Message Handling

answer = "Rain expected in Delhi."
new_messages = [
ToolMessage(content=answer, tool_call_id="12345"),
AIMessage(content=answer)
]
app2.update_state(config, {"messages": new_messages})

🟢 Invocation Styles

app.invoke({'messages': ['What is the GDP of India?']})
app.invoke({"messages": [HumanMessage(content="What is the GDP of India?")]})

