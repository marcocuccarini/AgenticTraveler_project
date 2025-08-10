from SPARQLWrapper import SPARQLWrapper, JSON

# Endpoints in ordine di priorità
ENDPOINTS = [
    "http://wit.istc.cnr.it/arco/virtuoso/sparql",  # ArCo ufficiale
    "https://dati.beniculturali.it/sparql"          # Mirror / alternativo
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
                print(f"Risultati trovati su {endpoint}")
                return results
            else:
                print(f"Nessun risultato su {endpoint}, passo al successivo...")
        except Exception as e:
            print(f"Errore su {endpoint}: {e}")
            print("Passo all'endpoint successivo...")
    return []

if __name__ == "__main__":
    # Esempi di nomi sicuri da provare
    nomi_test = [
        "Arco della Pace",
        "Arco di Tito",
        "Colosseo",
        "Ponte Vecchio"
    ]
    
    for nome in nomi_test:
        print(f"\n=== Ricerca: {nome} ===")
        results = query_by_name(nome)
        for res in results:
            print(res["entity"]["value"], "-", res["label"]["value"])
