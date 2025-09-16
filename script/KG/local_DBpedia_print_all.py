import sqlite3

DB_FILE = "dbpedia_entities.db"
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

# Mostra i primi 10 monumenti nel DB
c.execute("SELECT id, label, class FROM entities LIMIT 10")
rows = c.fetchall()

for r in rows:
    print(r)

conn.close()
