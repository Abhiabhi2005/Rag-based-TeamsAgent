import sqlite3

conn = sqlite3.connect("data/chat.db")

cursor = conn.cursor()

cursor.execute(
    """
    SELECT id,
           question,
           substr(response,1,100),
           created_at
    FROM query_log
    ORDER BY id DESC
    """
)

rows = cursor.fetchall()

print("\n=== QUERY LOG ===\n")

for row in rows:
    print(f"ID: {row[0]}")
    print(f"QUESTION: {row[1]}")
    print(f"ANSWER: {row[2]}...")
    print(f"TIME: {row[3]}")
    print("-" * 60)

conn.close()