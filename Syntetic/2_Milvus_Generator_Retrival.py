import json
from pymilvus import MilvusClient
from pymilvus.client.types import DataType
from pymilvus.orm.schema import CollectionSchema, FieldSchema
import os

#__________________________________
#json_file_path = r"syntetic/results/test_Syn_cosine.json"
#output_json_file_path = "syntetic/results/Milvus_Syn_Cosine_retrieved.json"
#COLLECTION_NAME = "Cosine_Syn"
#_________________________________
#json_file_path = r"syntetic/results/test_Syn_01.json"
#output_json_file_path = "syntetic/results/Milvus_Syn_01_retrieved.json"
#COLLECTION_NAME = "Bitmask_Syn"
#_________________________________
json_file_path = r"syntetic_1/test_syn_without_siamese.json"
output_json_file_path = "syntetic_1/Milvus_syn_without_siamese_retrieved.json"
COLLECTION_NAME = "syn_without_siamese"

VECTOR_DIMENSION = 1024

client = MilvusClient(uri="http://localhost:19530")

id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
image_embedding_field = FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)

schema = CollectionSchema(
    fields=[id_field, image_embedding_field],
    description="Collezione di embedding immagine per retrieval cross-modale",
    enable_dynamic_field=False
)


if client.has_collection(collection_name=COLLECTION_NAME):
    client.drop_collection(collection_name=COLLECTION_NAME)
client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="image_embedding",
    metric_type="COSINE",
    index_type="HNSW",
    params={"M": 8, "efConstruction": 64}
)

client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)

with open(json_file_path, 'r', encoding='utf-8') as f:
    full_json_data = json.load(f)

id_to_label_map = {}
entities_to_insert = []

for item in full_json_data:
    if "Immagine-testo iesimo" in item and "image_embedding" in item:
        if len(item["image_embedding"]) == VECTOR_DIMENSION:
            sample_id = item["Immagine-testo iesimo"]
            sample_label = item.get("label")
            id_to_label_map[sample_id] = sample_label
            entities_to_insert.append({
                "id": sample_id,
                "image_embedding": item["image_embedding"]
            })

if entities_to_insert:
    insert_result = client.insert(collection_name=COLLECTION_NAME, data=entities_to_insert)
    client.load_collection(collection_name=COLLECTION_NAME)
else:
    client.close()
    exit()

all_search_results = []

for query_item in full_json_data:
    query_id = query_item.get("Immagine-testo iesimo")
    query_label = query_item.get("label")
    text_vector = query_item.get("text_embedding")

    if text_vector and len(text_vector) == VECTOR_DIMENSION:
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            data=[text_vector],
            limit=5,
            search_field="image_embedding",
            output_fields=["id"],
            metric_type="COSINE"
        )

        formatted_hits = []
        if search_results and len(search_results[0]) > 0:
            for hit in search_results[0]:
                found_id = hit['id']
                image_label = id_to_label_map.get(found_id, None)
                formatted_hits.append({
                    "found_image_id": found_id,
                    "similarity_distance_cosine": hit['distance'],
                    "image_label": image_label
                })

        all_search_results.append({
            "query_text_row_id": query_id,
            "text_label": query_label,
            "top_similar_images": formatted_hits
        })
    else:
        print(f"Embedding di testo non valido per ID {query_id}.")

# === Salvataggio risultati ===
try:
    os.makedirs(os.path.dirname(output_json_file_path), exist_ok=True)
    with open(output_json_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_search_results, f, indent=4)
    print(f"Risultati salvati in: {output_json_file_path}")
except Exception as e:
    print(f"Errore nel salvataggio del JSON: {e}")

client.close()
