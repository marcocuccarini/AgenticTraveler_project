import sqlite3

# -----------------------------
# Connessione al DB
# -----------------------------
DB_FILE = "dbpedia_entities.db"
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

# -----------------------------
# Funzione: cerca monumento per nome
# -----------------------------
def search_monument(name):
    query = """
    SELECT id, uri, label, class
    FROM entities
    WHERE LOWER(label) LIKE LOWER(?)
    """
    c.execute(query, (f"%{name}%",))
    return c.fetchall()

# -----------------------------
# Funzione: ottieni proprietà di un monumento
# -----------------------------
def get_monument_properties(entity_id):
    query = """
    SELECT key, value
    FROM properties
    WHERE entity_id = ?
    """
    c.execute(query, (entity_id,))
    return c.fetchall()

# -----------------------------
# Esempio d'uso con più monumenti
# -----------------------------
test_monuments = ["Colosseum", "Eiffel Tower", "Pantheon", "Louvre"]

for name in test_monuments:
    print(f"\n🔎 Risultati per: {name}")
    monuments = search_monument(name)

    if not monuments:
        print("   ❌ Nessun risultato trovato")
        continue

    for m in monuments:
        print(f"\n🏛 Monumento: {m[2]} ({m[1]}) [Classi: {m[3]}]")
        
        # 2. Ottieni le proprietà dettagliate
        props = get_monument_properties(m[0])
        if props:
            for k, v in props:
                print(f"   - {k}: {v}")
        else:
            print("   (nessuna proprietà registrata)")

conn.close()
