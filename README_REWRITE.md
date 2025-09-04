# Query Rewriting & Voting System - AgenticTraveler

## Overview

La nuova funzionalità `--rewrite` implementa un sistema avanzato di riscrittura delle query e voting per migliorare significativamente i risultati del sistema RAG.

## Come Funziona

### 1. **Query Rewriting con LLM**
Quando viene utilizzato il flag `--rewrite`, il sistema:
- Prende la query originale (Monument + Description)
- Usa un LLM (Smolagents) per generare 2 riscritture alternative
- Ogni riscrittura interpreta la domanda dell'utente da angolazioni diverse:
  - Contesto storico
  - Dettagli architettonici 
  - Significato culturale
  - Informazioni turistiche
  - Eventi storici

### 2. **Multi-Query Search**
Il sistema esegue ricerche RAG per:
- **Query Originale**: `"Monument: Colosseo. Description: Ancient Roman amphitheatre"`
- **Rewrite 1**: Es. `"Historical architecture construction Roman Colosseum amphitheatre cultural heritage"`  
- **Rewrite 2**: Es. `"Tourist gladiatorial contests Vespasian Titus Roman Forum entertainment venue"`

### 3. **Voting Mechanism**
Sistema di voti per selezionare i migliori passaggi:
- **Peso per posizione**: I primi risultati valgono più punti
- **Conteggio voti**: Passaggi trovati da più query ottengono più voti
- **Tiebreaker**: In caso di pareggio, si preferiscono i risultati della query originale
- **Top-K**: Seleziona i 3 passaggi con più voti

## Usage

### Comando Base
```bash
python main.py --image monument.jpg --question "Tell me about this monument"
```

### Con Query Rewriting
```bash  
python main.py --image monument.jpg --question "Tell me about this monument" --rewrite
```

### Con Verbose per vedere il processo
```bash
python main.py --image monument.jpg --question "What's the history?" --rewrite --verbose
```

## Esempio di Output

```
🖼️ Analyzing image: colosseum.jpg
❓ Question: What is the history of this monument?
🔄 Mode: Enhanced RAG with Query Rewriting & Voting

🎯 STEP 3: RAG Analysis with Query Rewriting & Voting
🔄 Generating query rewrites using LLM...
✅ Generated 2 query rewrites
🔍 Testing 3 queries total
  Original: Monument: Colosseo. Description: Ancient Roman amphitheatre
  Rewrite 1: Historical construction Vespasian Titus Roman architecture amphitheatre
  Rewrite 2: Gladiatorial contests entertainment spectacles Roman Forum cultural heritage

🎯 Original found 5 results  
🎯 Rewrite 1 found 5 results
🎯 Rewrite 2 found 5 results

🗳️ Voting results summary:
  1. Votes: 9, Score: 0.8756
  2. Votes: 7, Score: 0.8234  
  3. Votes: 6, Score: 0.8123

🗳️ RAG Analysis with Voting Mechanism

**Query Rewriting Process:**
• Original query used
• 2 alternative interpretations generated  
• Voting mechanism applied to select best passages

**Top 3 Selected Texts (via voting):**
**1. Confidence: 0.8756**
Construction began under emperor Vespasian in AD 72 and completed in AD 80...

**🧠 AI Answer:**
The Colosseum has a rich history spanning nearly 2000 years...
```

## Vantaggi

1. **Coverage migliore**: Le riscritture catturano aspetti diversi delle informazioni
2. **Robustezza**: Meno dipendente dalla formulazione esatta della query
3. **Qualità**: Il voting seleziona i passaggi più consistentemente rilevanti
4. **Trasparenza**: Mostra il processo di rewriting e voting
5. **Fallback**: Funziona anche se le riscritture falliscono

## Algoritmo di Voting

```python
# Per ogni passaggio trovato:
for rank, (score, passage) in enumerate(results):
    position_weight = max(0, top_k - rank)  # 3, 2, 1, 0, 0...
    original_bonus = 0.1 if is_original_query else 0.0
    
    passage_votes[passage] += position_weight
    passage_scores[passage] = max(current_score, score + original_bonus)

# Ordinamento finale per:
# 1. Numero di voti (primario)
# 2. Punteggio similarity (secondario)  
# 3. Bonus query originale (tiebreaker)
```

## Configurazione

Nessuna configurazione aggiuntiva richiesta. Il sistema utilizza:
- Stesso RAG system (rag_system_smolagent.py)
- Stessi modelli embedding
- Stesso database di testi predefiniti
- Stessa infrastruttura Smolagents

L'unica differenza è l'aggiunta del flag `--rewrite` che attiva il processo avanzato.