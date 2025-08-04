import json
import numpy as np

# Percorso al file salvato da Milvus
input_file = "syntetic/results/Milvus_Syn_01_retrieved.json"
output_metrics_file = "syntetic/results/Metrics_Milvus_Syn_01.json"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

def k_precision(correct, k):
    return sum(correct[:k]) / k

def k_recall(correct, total_relevant, k):
    if total_relevant == 0:
        return 0.0
    return sum(correct[:k]) / total_relevant

def average_precision(correct):
    hits, ap = 0, 0.0
    for i, val in enumerate(correct):
        if val:
            hits += 1
            ap += hits / (i + 1)
    return ap / hits if hits > 0 else 0.0

def dcg(correct, k):
    return sum(correct[i] / np.log2(i + 2) for i in range(min(k, len(correct))))

def k_ndcg(correct, k):
    ideal = sorted(correct, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(correct, k) / idcg if idcg > 0 else 0.0

ks = [1, 3, 5]
map_scores = []
precision_scores = {k: [] for k in ks}
recall_scores = {k: [] for k in ks}
ndcg_scores = {k: [] for k in ks}

for entry in data:
    query_label = entry["text_label"]
    retrieved = entry["top_similar_images"]

    correct_labels = [int(query_label == r["image_label"]) for r in retrieved]
    total_relevant = sum(correct_labels)

    for k in ks:
        precision_scores[k].append(k_precision(correct_labels, k))
        recall_scores[k].append(k_recall(correct_labels, total_relevant, k))
        ndcg_scores[k].append(k_ndcg(correct_labels, k))

    map_scores.append(average_precision(correct_labels))

# Risultati medi
results = {
    "MAP": round(np.mean(map_scores), 5),
    "Metrics@k": {
        str(k): {
            "precision": round(np.mean(precision_scores[k]), 5),
            "recall": round(np.mean(recall_scores[k]), 5),
            "ndcg": round(np.mean(ndcg_scores[k]), 5)
        } for k in ks
    }
}

# Salva su file
with open(output_metrics_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)

