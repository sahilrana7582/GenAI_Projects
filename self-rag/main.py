import os

# faiss and torch/sklearn each ship their own libomp on macOS -> allow both to load
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import List,TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()

LINE = "=" * 80


def section(title: str):
    print(f"\n{LINE}\n  {title}\n{LINE}")


def doc_label(doc: Document) -> str:
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page", 0) + 1
    return f"{source} (page {page})"


def preview(text: str, width: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width] + "..."

docs = (
    PyPDFLoader("./data/HexaAI_Products_and_Pricing_2026.pdf").load() +
    PyPDFLoader("./data/HexaAI_Employee_Handbook_2026.pdf").load() +
    PyPDFLoader("./data/HexaAI_Customer_Policies.pdf").load() +
    PyPDFLoader("./data/HexaAI_Company_Overview.pdf").load()
)

chunks = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=150
).split_documents(docs)


section("INGESTION")
print(f"  Pages loaded : {len(docs)}")
print(f"  Chunks       : {len(chunks)}")

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
# Embedding model
embedding = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

vector_store = FAISS.from_documents(chunks, embedding)
retriever = vector_store.as_retriever(search_kwargs={"k":4})

class State(TypedDict):
    question: str
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    ans: str

class RetrievalDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True, if the external documents are needed to answer reliably, else False."
    )

decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON that matches this schema:\n"
            "{{'should_retrieve': boolean}}\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts, citations, or info likely not in the model.\n"
            "- should_retrieve=False for general explanations, definitions, or reasoning that doesn't need sources.\n"
            "- If unsure, choose True."
        ),
        ("human", "Question: {question}"),
    ]
)


should_retrieve_llm = llm.with_structured_output(RetrievalDecision)



def decide_retrieval(state: State):
    decision: RetrievalDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    section("STEP 1  |  DECIDE RETRIEVAL")
    print(f"  Question       : {state['question']}")
    print(f"  Need retrieval : {'YES -> retrieve' if decision.should_retrieve else 'NO  -> answer directly'}")
    return {"need_retrieval": decision.should_retrieve}


direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful assistant answering WITHOUT any documents, using only your general knowledge.\n\n"
            "Strict rules:\n"
            "- Answer ONLY if the answer is well-established general knowledge you are confident about "
            "(definitions, concepts, widely known facts, simple reasoning or math).\n"
            "- NEVER guess, assume or invent facts, numbers, names, dates, prices or policies.\n"
            "- If the question is about a specific company, product, person, internal policy, pricing, "
            "or anything private, recent or not widely known, do NOT answer from memory.\n"
            "- If you are not fully sure, or the answer would require information you do not have, "
            "reply with exactly: I do not know.\n"
            "- Do not add partial answers, speculation, or suggestions after saying I do not know.\n"
            "- Keep correct answers short, clear and direct."
        ),
        ("human", "{question}"),
    ]
)

def generate_direct(state: State):
    out = llm.invoke(
        direct_generation_prompt.format_messages(
            question=state["question"]
        )
    )
    section("STEP 2  |  GENERATE DIRECT (no documents)")
    print("  Answer generated from general knowledge.")
    return {"ans": out.content}

def retrieve(state: State):
    docs = retriever.invoke(state["question"])
    section(f"STEP 2  |  RETRIEVE  ({len(docs)} chunks)")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc_label(doc)}")
        print(f"      {preview(doc.page_content)}")
    return {"docs": docs}


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True if the document contains useful facts that directly help answer the question, else False."
    )

relevance_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance grader.\n"
            "You will be given a user question and ONE document.\n"
            "Return JSON that matches this schema:\n"
            "{{'is_relevant': boolean}}\n\n"
            "Rules:\n"
            "- is_relevant=True ONLY if the document alone contains useful facts needed to correctly answer the question.\n"
            "- is_relevant=False if the document is off-topic, or using it would require guessing or outside knowledge.\n"
            "- Judge only by what is written in the document. Do not use your own knowledge.\n"
            "- If unsure, choose False."
        ),
        ("human", "Question: {question}\n\nDocument:\n{document}"),
    ]
)

relevance_llm = llm.with_structured_output(RelevanceDecision)

def is_relevant(state: State):
    relevant_docs = []
    section("STEP 3  |  GRADE RELEVANCE")
    for i, doc in enumerate(state['docs'], 1):
        decision: RelevanceDecision = relevance_llm.invoke(
            relevance_prompt.format_messages(
                question=state["question"],
                document=doc.page_content
            )
        )
        verdict = "RELEVANT  " if decision.is_relevant else "IRRELEVANT"
        print(f"  [{i}] {verdict} | {doc_label(doc)}")
        if decision.is_relevant:
            relevant_docs.append(doc)

    print(f"\n  Kept {len(relevant_docs)} of {len(state['docs'])} chunks")
    return {"relevant_docs": relevant_docs}

def route_after_decide(state: State):
    if state["need_retrieval"]:
        return "retrieve"
    return "generate_direct"

g = StateGraph(State)

g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)
g.add_node("is_relevant", is_relevant)

g.add_edge(START, "decide_retrieval")
g.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {
        "generate_direct": "generate_direct",
        "retrieve": "retrieve"
    }
)

g.add_edge("generate_direct", END)
g.add_edge("retrieve", "is_relevant")
g.add_edge("is_relevant", END)

app = g.compile()

result = app.invoke(
    {
        "question": "who is the CEO?",
        "need_retrieval": False,
        "docs": [],
        "relevant_docs": [],
        "ans": ""
    }
)

section("RESULT")
print(f"  Question : {result['question']}")
print(f"  Answer   : {result['ans'] or '(not generated yet)'}")
print(f"\n  Relevant docs ({len(result['relevant_docs'])}):")
for i, doc in enumerate(result["relevant_docs"], 1):
    print(f"\n  [{i}] {doc_label(doc)}")
    print("  " + "-" * 76)
    for line in doc.page_content.splitlines():
        print(f"    {line}")
print(LINE)