import torch
import numpy as np
import json
import os
from Siamese_Network_for_syntetic import SiameseNetwork
from torch.nn.functional import cosine_similarity

def load_data_from_npz(npz_path):
    data = np.load(npz_path)
    text_embeddings = data["report_embeddings"]
    image_embeddings = data["image_embeddings"]
    labels = data["labels"]
    return text_embeddings, image_embeddings, labels

def test_siamese(model_path, test_npz_path, output_json_path, device):
    # Carica modello
    model = SiameseNetwork(text_input_dim=768, image_input_dim=1024)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Carica dati
    text_data, image_data, labels = load_data_from_npz(test_npz_path)
    num_samples = text_data.shape[0]

    results = []

    with torch.no_grad():
        for i in range(num_samples):
            text_tensor = torch.tensor(text_data[i], dtype=torch.float32, device=device).unsqueeze(0)
            image_tensor = torch.tensor(image_data[i], dtype=torch.float32, device=device).unsqueeze(0)

            text_emb, image_emb = model(text_tensor, image_tensor)

            # Normalizzazione per cosine similarity
            text_emb_norm = torch.nn.functional.normalize(text_emb, p=2, dim=1)
            image_emb_norm = torch.nn.functional.normalize(image_emb, p=2, dim=1)

            cos_sim = cosine_similarity(text_emb_norm, image_emb_norm).item()

            results.append({
                "Immagine-testo iesimo": i,
                "image_embedding": image_emb_norm.squeeze(0).cpu().numpy().tolist(),
                "text_embedding": text_emb_norm.squeeze(0).cpu().numpy().tolist(),
                "cosine_similarity": cos_sim,
                "label": int(labels[i]) if np.isscalar(labels[i]) else labels[i].tolist()
            })

    # Salva JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Testing completato. Risultati salvati in: {output_json_path}")

if __name__ == "__main__":
    model_path = "syntetic/models/syn_bitmask_01_model.pth"
    test_npz_path = "Test_Syntetic.npz"
    output_json_path = "syntetic/results/test_Syn_01.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_siamese(model_path, test_npz_path, output_json_path, device)
