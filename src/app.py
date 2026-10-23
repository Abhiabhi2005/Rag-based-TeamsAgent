# # app.py
# from os import environ
# from dotenv import load_dotenv
# from cache import get_cached_response, set_cached_response
# from database import (
#     initialize_db,
#     save_query,
#     get_query,
# )

# load_dotenv()  # make sure .env is loaded before we read from environ below
# initialize_db()
# from microsoft_agents.activity import load_configuration_from_env
# from microsoft_agents.hosting.core import (
#     AgentApplication,
#     TurnState,
#     TurnContext,
#     MemoryStorage,
# )
# from microsoft_agents.hosting.aiohttp import CloudAdapter
# from microsoft_agents.authentication.msal import MsalConnectionManager
# from start_server import start_server

# import graph  
# # your LangGraph pipeline (router -> retrieve -> generate -> citation_check)


# import hashlib

# from cache import (
#     get_cached_response,
#     set_cached_response,
# )

# from database import (
#     initialize_db,
#     save_query,
# )
# # --- SDK configuration & auth ---
# agents_sdk_config = load_configuration_from_env(environ)
# CONNECTION_MANAGER = MsalConnectionManager(**agents_sdk_config)
# ADAPTER = CloudAdapter(connection_manager=CONNECTION_MANAGER)

# AGENT_APP = AgentApplication[TurnState](
#     storage=MemoryStorage(),
#     adapter=ADAPTER,
#     connection_manager=CONNECTION_MANAGER,
#     **agents_sdk_config,
# )


# async def _help(context: TurnContext, _: TurnState):
#     await context.send_activity(
#         "Welcome to the Meeting Knowledge Assistant 🧠. "
#         "Ask me anything about the Atlas Launch meeting notes, or type /help."
#     )


# AGENT_APP.conversation_update("membersAdded")(_help)
# AGENT_APP.message("/help")(_help)


# @AGENT_APP.activity("message")
# async def on_message(context: TurnContext, _):

#     question = context.activity.text

#     cache_key = hashlib.md5(
#         question.lower().strip().encode()
#     ).hexdigest()

#     # ----------------------
#     # CACHE CHECK
#     # ----------------------
#     cached_answer = get_cached_response(cache_key)

#     if cached_answer:
#         print(f"[CACHE HIT] {question}")

#         await context.send_activity(
#             f"✅ Cache Hit\n\n{cached_answer}"
#         )
#         return

#     print(f"[CACHE MISS] {question}")

#     # ----------------------
#     # DB CHECK
#     # ----------------------
#     db_answer = get_query(question)

#     if db_answer:
#         print(f"[DB HIT] {question}")

#         set_cached_response(
#             cache_key,
#             db_answer
#         )

#         await context.send_activity(
#             f"✅ Database Hit\n\n{db_answer}"
#         )
#         return

#     print(f"[DB MISS] {question}")

#     # ----------------------
#     # LLM CALL
#     # ----------------------
#     print(f"[LLM CALLED] {question}")

#     result = graph.ask(question)

#     answer = result["final_answer"]

#     # ----------------------
#     # SAVE TO CACHE
#     # ----------------------
#     set_cached_response(
#         cache_key,
#         answer
#     )

#     print("[CACHE WRITE]")

#     # ----------------------
#     # SAVE TO DB
#     # ----------------------
#     save_query(
#         question,
#         answer
#     )

#     print("[DB WRITE]")

#     await context.send_activity(
#         f"🤖 LLM Generated Response\n\n{answer}"
#     )
# if __name__ == "__main__":
#     try:
#         start_server(
#             agent_application=AGENT_APP,
#             auth_configuration=CONNECTION_MANAGER.get_default_connection_configuration(),
#         )
#     except Exception as error:
#         raise error

# app.py
from os import environ
from dotenv import load_dotenv

load_dotenv()  # make sure .env is loaded before we read from environ below

from microsoft_agents.activity import load_configuration_from_env
from microsoft_agents.hosting.core import (
    AgentApplication,
    TurnState,
    TurnContext,
    MemoryStorage,
)
from microsoft_agents.hosting.aiohttp import CloudAdapter
from microsoft_agents.authentication.msal import MsalConnectionManager
from start_server import start_server

import graph  # Your LangGraph pipeline with semantic caching integrated

# --- SDK configuration & auth ---
agents_sdk_config = load_configuration_from_env(environ)
CONNECTION_MANAGER = MsalConnectionManager(**agents_sdk_config)
ADAPTER = CloudAdapter(connection_manager=CONNECTION_MANAGER)

AGENT_APP = AgentApplication[TurnState](
    storage=MemoryStorage(),
    adapter=ADAPTER,
    connection_manager=CONNECTION_MANAGER,
    **agents_sdk_config,
)


async def _help(context: TurnContext, _: TurnState):
    await context.send_activity(
        "Welcome to the Meeting Knowledge Assistant 🧠. "
        "Ask me anything about the Atlas Launch meeting notes, or type /help."
    )


AGENT_APP.conversation_update("membersAdded")(_help)
AGENT_APP.message("/help")(_help)


@AGENT_APP.activity("message")
async def on_message(context: TurnContext, _):
    question = context.activity.text.strip()

    if not question:
        return

    print(f"\n[USER QUESTION]: {question}")

    try:
        # graph.ask() handles:
        # 1. Garnet (Tier 1) Semantic Cache lookup (threshold >= 0.85)
        # 2. SQLite (Tier 2) Semantic Cache lookup + Garnet backfill
        # 3. Multi-Agent Graph execution (Planner -> Retrieve -> Generate -> Critique -> Aggregate)
        # 4. Saving new answers to both caches with Gemini embeddings & Langfuse tracing
        result = graph.ask(question)

        answer = result.get("final_answer", "Sorry, I couldn't generate an answer.")

        await context.send_activity(
            f"🤖 Assistant Response:\n\n{answer}"
        )

    except Exception as e:
        print(f"[ERROR]: {e}")
        await context.send_activity(
            "⚠️ An error occurred while processing your request. Please try again."
        )


if __name__ == "__main__":
    try:
        print("Starting Semantic Cache PDF Assistant Server for AgentsPlayground...")
        start_server(
            agent_application=AGENT_APP,
            auth_configuration=CONNECTION_MANAGER.get_default_connection_configuration(),
        )
    except Exception as error:
        raise error