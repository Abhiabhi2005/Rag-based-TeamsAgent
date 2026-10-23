import redis
import json
import uuid
import config

cache = redis.Redis(
    host=config.GARNET_HOST,
    port=config.GARNET_PORT,
    decode_responses=True
)

def get_garnet_keys():
    return cache.keys("semantic_cache:*")

def get_garnet_value(key):
    val = cache.get(key)
    return json.loads(val) if val else None

def save_to_garnet(question, embedding, answer):
    key = f"semantic_cache:{uuid.uuid4()}"
    data = {
        "question": question,
        "embedding": embedding,
        "answer": answer
    }
    cache.setex(
        key,
        config.CACHE_TTL,
        json.dumps(data)
    )