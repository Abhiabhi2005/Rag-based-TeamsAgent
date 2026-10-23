#llm.py
"""
Groq LLM wrapper. Centralizes model instantiation so every agent uses the
same configured client.
"""
from langchain_groq import ChatGroq

import config


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Return a configured ChatGroq client. Low temperature by default -- this
    system prioritizes grounded, consistent answers over creative variation."""
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Check your .env file has GROQ_API_KEY=your-key-here"
        )
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=config.GROQ_MODEL_NAME,
        temperature=temperature,
    )