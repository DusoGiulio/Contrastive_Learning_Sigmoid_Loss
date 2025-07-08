import json
import numpy as np

# Carica il file JSON
with open("Cosine_Matrix_Text_Label.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def correct_label(label1, label2):
    if label1 == "00000000000000" and label2 == "00000000000000":
        return True
    for i in range(len(label1)):
        if label1[i] == '1' and label2[i] == '1':
            return True
    return False

def k_precision(correct, k):
    count = 0
    for i in range(k):
        if i < len(correct):
            if correct[i]:
                count += 1
    return count / k

def k_recall(correct, tot, k):
    if tot == 0:
        return 0.0
    count = 0
    for i in range(k):
        if i < len(correct):
            if correct[i]:
                count += 1
    return count / tot

def average_precision(correct):
    ap = 0.0
    j = 0
    for i in range(len(correct)):
        if correct[i]:
            j += 1
            ap += j / (i + 1)
    if j == 0:
        return 0.0
    return ap / j

def dcg(correct, k):
    score = 0.0
    for i in range(k):
        if i < len(correct):
            score += correct[i] / np.log2(i + 2)
    return score

def k_ndcg(correct, k):
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

# Parametri
ks = [1, 3, 5]
map_total = []
precision_totals = {}
recall_totals = {}
ndcg_totals = {}

for k in ks:
    precision_totals[k] = []
    recall_totals[k] = []
    ndcg_totals[k] = []

# Analisi
for i in range(len(data)):
    entry = data[i]
    query_label = entry["label"]
    retrieved = entry["top_similar_images"]

    correct_labels = []
    for j in range(len(retrieved)):
        r_label = retrieved[j]["label"]
        correct_labels.append(correct_label(query_label, r_label))

    tot = 0
    for l in correct_labels:
        if l:
            tot += 1

    for k in ks:
        p = k_precision(correct_labels, k)
        r = k_recall(correct_labels, tot, k)
        n = k_ndcg(correct_labels, k)

        precision_totals[k].append(p)
        recall_totals[k].append(r)
        ndcg_totals[k].append(n)

    map_total.append(average_precision(correct_labels))

results = {
    "MAP": round(np.mean(map_total), 5),
    "Metriche K": {}
}

for k in ks:
    results["Metriche K"][str(k)] = {
        "precision": round(np.mean(precision_totals[k]), 5),
        "recall": round(np.mean(recall_totals[k]), 5),
        "ndcg": round(np.mean(ndcg_totals[k]), 5)
    }

with open("mesure_result_matrix_cosine.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

