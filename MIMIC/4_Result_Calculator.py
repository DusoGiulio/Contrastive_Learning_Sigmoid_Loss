
import json
import numpy as np

# Carica il file JSON con i risultati di similarità (query -> top immagini simili)
with open("Milvus_Base_ritrived_labeled.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Funzione per determinare se due label binarie sono rilevanti tra loro
def correct_label(label1, label2):
    #entrambe etichette completamente vuote -> considerate rilevanti
    if label1 == "00000000000000" and label2 == "00000000000000":
        return True
    # Se esiste almeno una patologia condivisa -> considerate rilevanti
    for i in range(len(label1)):
        if label1[i] == '1' and label2[i] == '1':
            return True
    return False

# Precision@k = (# rilevanti nei primi k) / k
def k_precision(correct, k):
    count = 0
    for i in range(k):
        if i < len(correct):
            if correct[i]:
                count += 1
    return count / k

# Recall@k = (# rilevanti nei primi k) / (# totali rilevanti)
def k_recall(correct, tot, k):
    if tot == 0:
        return 0.0
    count = 0
    for i in range(k):
        if i < len(correct):
            if correct[i]:
                count += 1
    return count / tot

# Average Precision (AP): media delle precisioni calcolate nei punti in cui troviamo un elemento rilevante
# Formula: AP = (1/m) * sum_{k in R} Precision@k, con R gli indici delle posizioni rilevanti
def average_precision(correct):
    ap = 0.0
    j = 0  # Numero di rilevanti trovati finora
    for i in range(len(correct)):
        if correct[i]:
            j += 1
            ap += j / (i + 1)  # Precision@i
    if j == 0:
        return 0.0
    return ap / j

# DCG@k (Discounted Cumulative Gain)
# Formula: DCG = sum_{i=1}^{k} (rel_i / log2(i+1)), dove rel_i ∈ {0,1}
def dcg(correct, k):
    score = 0.0
    for i in range(k):
        if i < len(correct):
            score += correct[i] / np.log2(i + 2)
    return score

# nDCG@k = DCG@k / IDCG@k (Ideal DCG, cioè il massimo possibile se l’ordinamento fosse perfetto)
def k_ndcg(correct, k):
    # Costruisci una lista ideale ordinata decrescente dei valori rilevanti
    max_correct = [x for x in correct]
    for i in range(len(max_correct)):
        for j in range(i + 1, len(max_correct)):
            if max_correct[j] > max_correct[i]:
                max_correct[i], max_correct[j] = max_correct[j], max_correct[i]
    ideal_dcg = dcg(max_correct, k)
    actual_dcg = dcg(correct, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg

# Parametri: k rappresenta le posizioni nei ranking su cui calcolare le metriche
ks = [1, 3, 5]
map_total = []  # Lista per memorizzare tutti i valori di Average Precision (per ogni query)
precision_totals = {}  # Precision@k per ogni k
recall_totals = {}     # Recall@k per ogni k
ndcg_totals = {}       # nDCG@k per ogni k

# Inizializza le liste per ogni valore di k
for k in ks:
    precision_totals[k] = []
    recall_totals[k] = []
    ndcg_totals[k] = []

# Analisi: per ogni query nel dataset
for i in range(len(data)):
    entry = data[i]
    query_label = entry["label"]  # Etichetta della query
    retrieved = entry["top_similar_images"]  # Lista dei top-k risultati

    # Costruisci la lista di rilevanza binaria per ogni elemento restituito (1 = rilevante, 0 = no)
    correct_labels = []
    for j in range(len(retrieved)):
        r_label = retrieved[j]["label"]
        correct_labels.append(correct_label(query_label, r_label))

    # Conta il numero totale di elementi rilevanti nella lista restituita
    tot = 0
    for l in correct_labels:
        if l:
            tot += 1

    # Calcola precision, recall e nDCG per ciascun valore di k
    for k in ks:
        p = k_precision(correct_labels, k)
        r = k_recall(correct_labels, tot, k)
        n = k_ndcg(correct_labels, k)

        precision_totals[k].append(p)
        recall_totals[k].append(r)
        ndcg_totals[k].append(n)

    # Calcola Average Precision (AP) per questa query
    map_total.append(average_precision(correct_labels))

# Costruisci dizionario dei risultati medi finali
results = {
    "MAP": round(np.mean(map_total), 5),  # Mean Average Precision = media di tutte le AP
    "Metriche K": {}  # Precision/Recall/nDCG per ogni k
}

# Calcola media per ogni metrica a ciascun k
for k in ks:
    results["Metriche K"][str(k)] = {
        "precision": round(np.mean(precision_totals[k]), 5),
        "recall": round(np.mean(recall_totals[k]), 5),
        "ndcg": round(np.mean(ndcg_totals[k]), 5)
    }

# Salva i risultati in un file JSON
with open("Base_METRICS.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)
