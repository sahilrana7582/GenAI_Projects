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
    context: str
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    unsupported_claims: List[str]
    revisions: int

MAX_REVISIONS = 3

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

rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval-augmented question-answering system.\n"
            "Answer the question using ONLY the information explicitly provided in the context.\n\n"
            "Rules:\n"
            "- Do not use your internal knowledge or external information.\n"
            "- Do not make assumptions, infer unsupported facts, or fabricate information.\n"
            "- If the context does not contain sufficient information to answer the question, "
            "respond exactly with: I don't know based on the provided context.\n"
            "- If only part of the question can be answered from the context, "
            "respond exactly with: I don't know based on the provided context.\n"
            "- Answer only when the context provides sufficient evidence for the complete answer.\n"
            "- Keep the answer clear and directly relevant to the question."
        ),
        (
            "human",
            "Context:\n{context}\n\n"
            "Question:\n{question}"
        ),
    ]
)

def generate_from_context(state: State):
    context = "\n\n----\n\n".join(
        d.page_content for d in state.get("relevant_docs", [])
    ).strip()

    section(f"STEP 4  |  GENERATE FROM CONTEXT ({len(state.get('relevant_docs', []))} chunks)")

    if not context:
        print("  No context -> I don't know.")
        return {
            "ans": "I don't know based on the provided context.",
            "context": "",
        }

    out = llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"],
            context=context,
        )
    )
    print("  Answer generated from the relevant chunks.")

    return {
        "ans": out.content,
        "context": context,
    }

class SupportDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"] = Field(
        ...,
        description=(
            "fully_supported: every claim in the answer is stated in the context. "
            "partially_supported: some claims are in the context, some are not. "
            "no_support: the answer is not backed by the context at all."
        )
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Exact claims from the answer that are NOT stated in the context. Empty if fully_supported."
    )

issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict hallucination checker.\n"
            "You will be given a question, a context and an answer.\n"
            "Check EVERY claim in the answer (facts, names, numbers, dates, prices, conditions) against the context.\n"
            "Return JSON that matches this schema:\n"
            "{{'issup': 'fully_supported' | 'partially_supported' | 'no_support', 'unsupported_claims': [string]}}\n\n"
            "Rules:\n"
            "- fully_supported: every claim is explicitly stated in the context (100% supported).\n"
            "- partially_supported: at least one claim is in the context, but at least one claim is not.\n"
            "- no_support: none of the claims are in the context, or the answer contradicts it.\n"
            "- A claim that adds details, numbers or conditions not written in the context is unsupported.\n"
            "- Judge only by the context. Do not use your own knowledge, even if the claim is true in the real world.\n"
            "- The answer \"I don't know based on the provided context.\" makes no claims: return fully_supported.\n"
            "- List every unsupported claim word for word in unsupported_claims."
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"),
    ]
)

issup_llm = llm.with_structured_output(SupportDecision)

def is_sup(state: State):
    decision: SupportDecision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=state["question"],
            context=state["context"],
            answer=state["ans"],
        )
    )
    section(f"STEP 5  |  IS SUPPORTED?  (revision {state.get('revisions', 0)}/{MAX_REVISIONS})")
    print(f"  Answer  : {preview(state['ans'], 200)}")
    print(f"  Verdict : {decision.issup.upper()}")
    for claim in decision.unsupported_claims:
        print(f"    - unsupported: {claim}")
    return {"issup": decision.issup, "unsupported_claims": decision.unsupported_claims}

revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You revise an answer so it is 100% supported by the context.\n\n"
            "Rules:\n"
            "- REMOVE every unsupported claim listed below. Do not rephrase them, do not replace them with guesses.\n"
            "- Keep every claim that IS stated in the context, using the context's exact facts and numbers.\n"
            "- Do not add any new information that is not written in the context.\n"
            "- If nothing supported is left to answer the question, respond exactly with: "
            "I don't know based on the provided context.\n"
            "- Return only the revised answer."
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\n"
            "Current answer:\n{answer}\n\nUnsupported claims to remove:\n{unsupported_claims}"
        ),
    ]
)

def revise_ans(state: State):
    claims = "\n".join(f"- {c}" for c in state.get("unsupported_claims", [])) or "- (none listed; remove anything not in the context)"
    out = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            context=state["context"],
            answer=state["ans"],
            unsupported_claims=claims,
        )
    )
    revisions = state.get("revisions", 0) + 1
    section(f"STEP 6  |  REVISE ANSWER  (revision {revisions}/{MAX_REVISIONS})")
    print(f"  Old : {preview(state['ans'], 200)}")
    print(f"  New : {preview(out.content, 200)}")
    return {"ans": out.content, "revisions": revisions}

def route_after_issup(state: State):
    if state["issup"] == "fully_supported":
        return "end"
    if state.get("revisions", 0) >= MAX_REVISIONS:
        print(f"\n  Max revisions ({MAX_REVISIONS}) reached -> stopping with a {state['issup']} answer")
        return "end"
    return "revise_ans"

def decide_retrieval(state: State):
    decision: RetrievalDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    section("STEP 1  |  DECIDE RETRIEVAL")
    print(f"  Question       : {state['question']}")
    print(f"  Need retrieval : {'YES -> retrieve' if decision.should_retrieve else 'NO  -> answer directly'}")
    return {"need_retrieval": decision.should_retrieve}

def route_after_decide(state: State):
    if state["need_retrieval"]:
        return "retrieve"
    return "generate_direct"

def route_after_retrieval(state: State):
    if not state.get("relevant_docs"):
        return "generate_direct"

    return "generate_from_context"

g = StateGraph(State)

g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)
g.add_node("is_relevant", is_relevant)
g.add_node("generate_from_context", generate_from_context)
g.add_node("is_sup", is_sup)
g.add_node("revise_ans", revise_ans)

g.add_edge(START, "decide_retrieval")
g.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {
        "generate_direct": "generate_direct",
        "retrieve": "retrieve"
    }
)

g.add_conditional_edges(
    "is_relevant",
    route_after_retrieval,
    {
        "generate_direct": "generate_direct",
        "generate_from_context": "generate_from_context"
    }
)

g.add_conditional_edges(
    "is_sup",
    route_after_issup,
    {
        "revise_ans": "revise_ans",
        "end": END
    }
)

g.add_edge("retrieve", "is_relevant")
g.add_edge("generate_direct", END)
g.add_edge("generate_from_context", "is_sup")
g.add_edge("revise_ans", "is_sup")

app = g.compile()

# Redraw graph.png on every run so it always matches the current nodes/edges.
# draw_mermaid_png() sends the graph structure (node names only) to mermaid.ink to render it.
try:
    with open("graph.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    print("\n  graph.png updated")
except Exception as e:
    print(f"\n  graph.png not updated ({e}) -> paste this into https://mermaid.live:\n")
    print(app.get_graph().draw_mermaid())

result = app.invoke(
    {
        "question": "who is the CEO?",
        "need_retrieval": False,
        "docs": [],
        "relevant_docs": [],
        "ans": "",
        "context": "",
        "issup": "no_support",
        "unsupported_claims": [],
        "revisions": 0
    }
)

section("RESULT")
print(f"  Question : {result['question']}")
print(f"  Answer   : {result['ans'] or '(not generated yet)'}")
print(f"  Support  : {result['issup'] if result['context'] else '(not checked - no documents used)'}")
print(f"  Revisions: {result['revisions']}")
print(f"\n  Relevant docs ({len(result['relevant_docs'])}):")
for i, doc in enumerate(result["relevant_docs"], 1):
    print(f"\n  [{i}] {doc_label(doc)}")
    print("  " + "-" * 76)
    for line in doc.page_content.splitlines():
        print(f"    {line}")
print(LINE)