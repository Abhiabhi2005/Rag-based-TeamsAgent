import os
import json
import time
import numpy as np
from google import genai
from langgraph.graph import StateGraph, END
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from agents import (
    GraphState,
    planner_node,
    fan_out_to_sub_questions,
    process_sub_question,
    aggregator_node,
)
import database
import cache as garnet_cache
import config

# Initialize Gemini Client for Embeddings
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Langfuse
langfuse = get_client()
langfuse_handler = CallbackHandler()

def create_embedding(text: str) -> list[float]:
    result = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=f"task: sentence similarity | query: {text}"
    )
    return result.embeddings[0].values

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("process_sub_question", process_sub_question)
    graph.add_node("aggregator", aggregator_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", fan_out_to_sub_questions, ["process_sub_question"])
    graph.add_edge("process_sub_question", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()

import time

def ask(question: str) -> dict:
    """Run question through Semantic Cache (Garnet -> SQLite) before hitting the multi-agent graph with timing and tracing."""
    start_time = time.time()  # Start timer
    
    database.initialize_db()
    query_embedding = create_embedding(question)

    # --- Step 1: Check Garnet (Tier 1) ---
    keys = garnet_cache.get_garnet_keys()
    best_score = 0
    best_answer = None

    for key in keys:
        data = garnet_cache.get_garnet_value(key)
        if data and "embedding" in data:
            score = cosine_similarity(query_embedding, data["embedding"])
            if score > best_score:
                best_score = score
                best_answer = data["answer"]

    if best_score >= 0.90 and best_answer:
        elapsed_time = time.time() - start_time
        print(f"-> GARNET SEMANTIC CACHE HIT (Score: {best_score:.3f}) | Time taken: {elapsed_time:.4f} seconds")
        return {"question": question, "sub_questions": [question], "results": [], "final_answer": best_answer}

    # --- Step 2: Check SQLite (Tier 2) ---
    rows = database.get_all_cached_queries()
    best_score = 0
    best_answer = None

    for row in rows:
        stored_embedding = json.loads(row[2])
        score = cosine_similarity(query_embedding, stored_embedding)
        if score > best_score:
            best_score = score
            best_answer = row[3]

    print(f"SQLite best similarity: {best_score:.3f}")

    if best_score >= 0.90 and best_answer:
        elapsed_time = time.time() - start_time
        print(f"-> SQLITE SEMANTIC CACHE HIT (Score: {best_score:.3f}). Backfilling Garnet... | Time taken: {elapsed_time:.4f} seconds")
        garnet_cache.save_to_garnet(question, query_embedding, best_answer)
        return {"question": question, "sub_questions": [question], "results": [], "final_answer": best_answer}

    print("-> SQLITE CACHE MISS")

    # --- Step 3: Run Multi-Agent Graph (Tier 3 - LLM Call) ---
    print("-> CACHE MISS. Running Multi-Agent Graph...")
    app = build_graph()
    initial_state: GraphState = {
        "question": question,
        "sub_questions": [],
        "results": [],
        "final_answer": "",
    }
    
    final_state = app.invoke(
        initial_state, 
        config={"callbacks": [langfuse_handler]}
    )
    final_answer = final_state["final_answer"]

    # --- Step 4: Save to Caches ---
    database.save_to_sqlite(question, query_embedding, final_answer)
    garnet_cache.save_to_garnet(question, query_embedding, final_answer)

    langfuse.flush()

    elapsed_time = time.time() - start_time
    print(f"-> LLM & GRAPH EXECUTION COMPLETE | Time taken: {elapsed_time:.4f} seconds")

    return final_state