import json
import sqlite3
from SPARQLWrapper import SPARQLWrapper, JSON
import time

# -----------------------------
# Step 0: Setup SPARQL endpoint
# -----------------------------
endpoint = "https://dbpedia.org/sparql"
sparql = SPARQLWrapper(endpoint)

# -----------------------------
# Step 1: Define target classes
# -----------------------------
TARGET_CLASSES = [
    "Monument", "HistoricPlace", "Artwork", "Museum",
    "Church", "Temple", "Castle", "Palace", "ArchaeologicalSite",
    "HistoricBuilding", "Bridge", "Statue", "Memorial", "HeritageSite",
    "Park", "Library", "Theatre", "Mosque", "Synagogue", "Cathedral"
]

# -----------------------------
# Step 2: Query DBpedia
# -----------------------------
def run_query(q):
    sparql.setQuery(q)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

entities = {}
total_samples = 0  # total number of rows retrieved

BATCH_SIZE = 1000  # fetch 1000 rows at a time to avoid timeouts

for cls in TARGET_CLASSES:
    print(f"🔍 Querying DBpedia for class {cls} ...")
    offset = 0
    while True:
        query = f"""
        SELECT ?s ?label ?p ?o
        WHERE {{
          ?s a dbo:{cls} ;
             rdfs:label ?label .
          OPTIONAL {{ ?s ?p ?o }}
          FILTER (lang(?label) = 'en')
        }}
        LIMIT {BATCH_SIZE}
        OFFSET {offset}
        """
        results = run_query(query)
        rows = results["results"]["bindings"]

        if not rows:  # no more data
            break

        for r in rows:
            total_samples += 1
            uri = r["s"]["value"]
            label = r["label"]["value"]
            prop = r.get("p", {}).get("value", None)
            val = r.get("o", {}).get("value", None)

            if uri not in entities:
                entities[uri] = {"uri": uri, "label": label, "class": [cls], "properties": {}}
            else:
                if cls not in entities[uri]["class"]:
                    entities[uri]["class"].append(cls)

            if prop and val:
                key = prop.replace("http://dbpedia.org/ontology/", "")
                if key not in entities[uri]["properties"]:
                    entities[uri]["properties"][key] = val
                else:
                    if not isinstance(entities[uri]["properties"][key], list):
                        entities[uri]["properties"][key] = [entities[uri]["properties"][key]]
                    entities[uri]["properties"][key].append(val)

        offset += BATCH_SIZE
        time.sleep(0.1)  # small delay to avoid overloading DBpedia

print(f"✅ Retrieved {len(entities)} unique entities from DBpedia")
print(f"📊 Retrieved {total_samples} total samples (rows) from DBpedia")

# -----------------------------
# Step 3: Save JSON
# -----------------------------
JSON_FILE = "dbpedia_entities.json"
with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(list(entities.values()), f, indent=2)
print(f"💾 Saved JSON to {JSON_FILE}")

# -----------------------------
# Step 4: Save SQLite
# -----------------------------
SQLITE_FILE = "dbpedia_entities.db"
conn = sqlite3.connect(SQLITE_FILE)
c = conn.cursor()

# Create tables
c.execute("DROP TABLE IF EXISTS entities")
c.execute("DROP TABLE IF EXISTS properties")
c.execute("""
CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    uri TEXT UNIQUE,
    label TEXT,
    class TEXT
)
""")
c.execute("""
CREATE TABLE properties (
    entity_id INTEGER,
    key TEXT,
    value TEXT,
    FOREIGN KEY(entity_id) REFERENCES entities(id)
)
""")

# Insert data
for e in entities.values():
    class_str = ",".join(e["class"])
    c.execute("INSERT OR IGNORE INTO entities (uri, label, class) VALUES (?, ?, ?)",
              (e["uri"], e.get("label", ""), class_str))
    entity_id = c.lastrowid
    for k, v in e["properties"].items():
        if isinstance(v, list):
            for item in v:
                c.execute("INSERT INTO properties (entity_id, key, value) VALUES (?, ?, ?)",
                          (entity_id, k, item))
        else:
            c.execute("INSERT INTO properties (entity_id, key, value) VALUES (?, ?, ?)",
                      (entity_id, k, v))

conn.commit()
conn.close()
print(f"💾 Saved SQLite DB to {SQLITE_FILE} with {len(entities)} entities")
