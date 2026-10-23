"""
Agent node functions for the Meeting/Document Knowledge Assistant's
multi-agent LangGraph pipeline.

Flow:
    START -> planner --(Send, one per sub-question, parallel fan-out)--> process_sub_question
                                                                                |
                                                                                v
                                                                           aggregator -> END
                                                                           

Each sub-question runs its own retrieve -> generate -> critique loop
(with up to MAX_RETRIES retries on ungrounded answers) inside
process_sub_question, so a bad chunk for one sub-question doesn't block
the others.
"""
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langgraph.types import Send

from llm import get_llm
from retriever import BM25Retriever
import config

MAX_RETRIES = 2  # how many times a sub-question's critique can send it back to retrieve


# --- State schemas ---

class SubQuestionResult(TypedDict):
    question: str
    question_type: str
    answer: str
    grounded: bool
    sources: list[str]


class GraphState(TypedDict):
    question: str
    sub_questions: list[str]
    results: Annotated[list[SubQuestionResult], operator.add]
    final_answer: str    


_retriever_instance = None


def _get_retriever() -> BM25Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = BM25Retriever.from_cache()
    return _retriever_instance


# --- Planner agent ---

PLANNER_PROMPT = """You are a strict planning assistant for a document QA system. Your ONLY job is to output a list of sub-questions based on the user's question. 

CRITICAL RULES:
- Do NOT chat, converse, or greet the user.
- Do NOT provide conversational filler (e.g. "Sure, here is...").
- Each sub-question must start with "- ".
- If the question does not need decomposition, output ONLY the original question starting with "- ".

Examples:
Question: What is the launch date?
- What is the launch date?

Question: Which browsers are supported?
- Which browsers are supported?

Question: {question}
"""

def planner_node(state: GraphState) -> dict:
    question = state["question"].strip()
    
    # Simple queries don't need the planner LLM to avoid chatty misbehavior
    # If it's a short question or ends with a question mark, keep it directly as a single sub-question
    if len(question.split()) <= 12 or question.endswith("?"):
        return {"sub_questions": [question]}

    llm = get_llm()
    prompt = PLANNER_PROMPT.format(question=question)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    lines = [
        line.strip("- ").strip()
        for line in response.content.strip().splitlines()
        if line.strip() and not line.lower().startswith("sure") and not line.lower().startswith("please")
    ]
    
    sub_questions = lines if lines and len(lines[0]) < 150 else [question]
    return {"sub_questions": sub_questions}

def fan_out_to_sub_questions(state: GraphState) -> list[Send]:
    """Conditional edge: dispatch one Send per sub-question, to run in parallel."""
    return [
        Send("process_sub_question", {"question": sq}) for sq in state["sub_questions"]
    ]


# --- Prompts reused inside process_sub_question ---

ROUTER_SYSTEM_PROMPT = """You classify questions about a set of meeting notes and reference \
documents into exactly one category. Reply with ONLY the category word, nothing else.

Categories:
- factual: asks for a specific fact, decision, date, number, or detail
- summary: asks to summarize, list, or condense information across one or more documents

Examples:
"What is the launch date?" -> factual
"Summarize the action items from last week" -> summary
"What was the Q3 budget?" -> factual
"Give me an overview of the client call" -> summary
"""

FACTUAL_PROMPT = """You answer questions using ONLY the provided context from meeting notes and \
reference documents. Be precise and cite which document each fact comes from (e.g. "per \
03_client_call.pdf"). If the context does not contain the answer, say clearly that the \
documents don't cover this -- do not guess or make anything up.

Context:
{context}

Question: {question}

Answer:"""

SUMMARY_PROMPT = """You summarize information from meeting notes and reference documents using \
ONLY the provided context. Be concise, use bullet points where helpful, and note which \
document each point comes from. If the context is insufficient to answer, say so plainly.

Context:
{context}

Request: {question}

Summary:"""

GROUNDING_PROMPT = """You check whether an ANSWER is actually supported by the given CONTEXT. \
Reply with ONLY "yes" or "no".

CONTEXT:
{context}

ANSWER:
{answer}

Is the ANSWER fully supported by the CONTEXT? Reply "yes" or "no" only."""


def _format_context(docs: list[Document]) -> str:
    if not docs:
        return "(no relevant documents found)"
    parts = []
    for doc in docs:
        parts.append(f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _classify(question: str) -> str:
    llm = get_llm()
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    question_type = response.content.strip().lower()
    return question_type if question_type in ("factual", "summary") else "factual"


def process_sub_question(state: dict) -> dict:
    """
    Retriever + Synthesizer + Critique agents for one sub-question, run as a
    single fanned-out branch. Retries retrieve -> generate internally up to
    MAX_RETRIES times if the citation check fails, then gives up gracefully.
    """
    question = state["question"]
    retriever = _get_retriever()
    llm = get_llm()

    question_type = _classify(question)
    prompt_template = SUMMARY_PROMPT if question_type == "summary" else FACTUAL_PROMPT

    docs: list[Document] = []
    answer = ""
    grounded = False

    for attempt in range(MAX_RETRIES + 1):
        docs = retriever.retrieve(question, top_k=config.TOP_K)
        context = _format_context(docs)

        prompt = prompt_template.format(context=context, question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()

        check_prompt = GROUNDING_PROMPT.format(context=context, answer=answer)
        check_response = llm.invoke([HumanMessage(content=check_prompt)])
        grounded = "yes" in check_response.content.strip().lower()

        if grounded:
            break

    sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})

    result: SubQuestionResult = {
        "question": question,
        "question_type": question_type,
        "answer": answer,
        "grounded": grounded,
        "sources": sources,
    }
    return {"results": [result]}


# --- Aggregator agent ---

AGGREGATE_PROMPT = """You combine answers to several sub-questions into a single coherent \
response to the user's original question. Keep each source's citation. Be concise -- don't \
repeat information unnecessarily.

Original question: {question}

Sub-answers:
{sub_answers}

Combined answer:"""


def aggregator_node(state: GraphState) -> dict:
    results = state["results"]

    if len(results) == 1:
        return {"final_answer": results[0]["answer"]}

    sub_answers = "\n\n".join(
        f"Q: {r['question']}\nA: {r['answer']}" for r in results
    )
    llm = get_llm()
    prompt = AGGREGATE_PROMPT.format(question=state["question"], sub_answers=sub_answers)
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_answer": response.content.strip()}


