import sqlite3
import json
import config

def initialize_db():
    conn = sqlite3.connect(config.SQLITE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            embedding TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def get_all_cached_queries():
    conn = sqlite3.connect(config.SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT id, question, embedding, answer FROM cache")
    rows = cursor.fetchall()
    conn.close()
    return rows

def save_to_sqlite(question, embedding, answer):
    conn = sqlite3.connect(config.SQLITE_DB)
    conn.execute(
        """
        INSERT INTO cache (question, embedding, answer)
        VALUES (?, ?, ?)
        """,
        (question, json.dumps(embedding), answer)
    )
    conn.commit()
    conn.close()