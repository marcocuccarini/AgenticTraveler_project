from SPARQLWrapper import SPARQLWrapper, JSON

# Endpoints in ordine di priorità
ENDPOINTS = [
    "http://wit.istc.cnr.it/arco/virtuoso/sparql",  # ArCo ufficiale
    "https://dati.beniculturali.it/sparql"          # Alternativo MiC
]

def query_by_name(nome):
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?entity ?label
    WHERE {{
      ?entity rdfs:label ?label .
      FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{nome}")))
    }}
    LIMIT 20
    """
    for endpoint in ENDPOINTS:
        sparql = SPARQLWrapper(endpoint)
        sparql.setReturnFormat(JSON)
        sparql.setQuery(query)
        try:
            print(f"Tento l'endpoint: {endpoint}")
            response = sparql.query().convert()
            results = response["results"]["bindings"]
            if results:
                print(f"✅ Risultati trovati su {endpoint}")
                return results
            else:
                print(f"⚠ Nessun risultato su {endpoint}, passo al successivo...")
        except Exception as e:
            print(f"❌ Errore su {endpoint}: {e}")
            print("Passo all'endpoint successivo...")
    return []

if __name__ == "__main__":
    monumenti = [
        "Loggia dei Lanzi",
        "Perseo e medusa",
        "Palazzo Vecchio",
        "Galleria degli Uffizi",
        "Nettuno di Firenze",
        "Statua Cosimo de Medici",
        "David Michelangelo piazza signoria",
        "Ratto delle sabine",
        "Ercole e il centauro",
        "Ratto di Polissesena",
        "Patroclo e Menelao"
    ]

    for nome in monumenti:
        print(f"\n=== Ricerca: {nome} ===")
        results = query_by_name(nome)
        if results:
            for res in results:
                print(res["entity"]["value"], "-", res["label"]["value"])
        else:
            print("Nessun risultato trovato.")
