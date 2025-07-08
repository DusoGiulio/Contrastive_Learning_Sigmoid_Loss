######################CREA BITMASK 01#############################
# import pandas as pd

# df = pd.read_csv("mimic_data/mimic-cxr-2.0.0-chexpert.csv")

# label_cols = [
#     "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
#     "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
#     "Lung Opacity", "No Finding", "Pleural Effusion",
#     "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
# ]
# # Funzione per creare la bitmask
# def make_bitmask(row):
#     return ''.join(['1' if row[col] == 0 else '0' for col in label_cols])

# df["bitmask_0_1"] = df.apply(make_bitmask, axis=1)

# df.to_csv("mimic_data/mimic-cxr-2.0.0-chexpert.csv", index=False)
###############################################################
##################MERGE BITMASK e TEST
# import pandas as pd

# # Forza bitmask_0_1 come stringa per non perdere zeri iniziali
# df_chexpert = pd.read_csv("mimic_data/mimic-cxr-2.0.0-chexpert.csv", dtype={"bitmask_0_1": str})
# df_test = pd.read_csv("mimic_data/test.csv")
# # Merge sulle chiavi corrette
# df_merged = pd.merge(
#     df_test,
#     df_chexpert[["subject_id", "study_id", "bitmask_0_1"]],
#     on=["subject_id", "study_id"],
#     how="left"
# )

# # Salva il file preservando gli zeri iniziali
# df_merged.to_csv("mimic_data/test.csv", index=False)

# # Mostra l'anteprima
# print(df_merged[["subject_id", "study_id", "bitmask_0_1"]].head())

#######################################################################

import json
import pandas as pd

# File paths
input_result_file = "milvus_search_data_test_t10_cosine.json"
csv_report_file = "mimic_data/test.csv"
output_file = "Cosine_Matrix_Text_Label.json"

# Carica i risultati delle ricerche Milvus
with open(input_result_file, 'r', encoding='utf-8') as f:
    search_results = json.load(f)

# Carica il CSV contenente i report
df_reports = pd.read_csv(csv_report_file,dtype={"bitmask_0_1": str})

# Verifica che la colonna 'report' esista
if "report" not in df_reports.columns:
    raise ValueError("La colonna 'report' non è presente in test.csv")

# Lista dei report posizionali
reports = df_reports["report"].tolist()
labels = df_reports["bitmask_0_1"]
# Costruisci il nuovo output con i report testuali associati
enriched_results = []

for entry in search_results:
    query_index = entry["query_text_row_id"]

    # Recupera il report del testo (query)
    query_report = reports[query_index] if query_index < len(reports) else "[Indice fuori range]"

    enriched_entry = {
        "query_text_row_id": query_index,
        "query_text_report": query_report,
        "label":str(labels[query_index]),
        "top_similar_images": []
        
    }

    for match in entry.get("top_similar_images", []):
        image_index = match["found_image_id"]
        similarity = match["similarity_distance_cosine"]

        image_report = reports[image_index] if image_index < len(reports) else "[Indice fuori range]"

        enriched_entry["top_similar_images"].append({
            "found_image_id": image_index,
            "similarity_distance_cosine": similarity,
            "associated_report": image_report,
            "label":str(labels[image_index])
        })

    enriched_results.append(enriched_entry)

# Salva il file JSON arricchito
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(enriched_results, f, indent=4)

