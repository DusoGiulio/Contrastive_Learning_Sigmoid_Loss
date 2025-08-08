import torch
import torch.nn as nn
import numpy as np
import json
import os
from torch.nn.functional import cosine_similarity

def load_data_from_npz(npz_path):
    data = np.load(npz_path)
    text_embeddings = data["report_embeddings"]  # (N, 768)
    image_embeddings = data["image_embeddings"]  # (N, 1024)
    labels = data["labels"]
    return text_embeddings, image_embeddings, labels

class RandomProjection(nn.Module):
    def __init__(self, text_input_dim=768, image_input_dim=1024):
        super().__init__()
        self.projection = nn.Linear(text_input_dim, image_input_dim)

    def forward(self, x):
        return self.projection(x)

def test_random_projection(test_npz_path, output_json_path, device):
    # Modello di proiezione casuale
    projector = RandomProjection()
    projector.to(device)
    projector.eval()

    # Carica dati
    text_data, image_data, labels = load_data_from_npz(test_npz_path)
    num_samples = text_data.shape[0]

    results = []

    with torch.no_grad():
        for i in range(num_samples):
            text_tensor = torch.tensor(text_data[i], dtype=torch.float32, device=device).unsqueeze(0)
            image_tensor = torch.tensor(image_data[i], dtype=torch.float32, device=device).unsqueeze(0)

            # Proiezione casuale
            text_proj = projector(text_tensor)

            # Normalizzazione
            text_norm = torch.nn.functional.normalize(text_proj, p=2, dim=1)
            image_norm = torch.nn.functional.normalize(image_tensor, p=2, dim=1)

            # Similarità coseno
            cos_sim = cosine_similarity(text_norm, image_norm).item()

            results.append({
                "Immagine-testo iesimo": i,
                "image_embedding": image_norm.squeeze(0).cpu().numpy().tolist(),
                "text_embedding": text_norm.squeeze(0).cpu().numpy().tolist(),
                "cosine_similarity": cos_sim,
                "label": int(labels[i]) if np.isscalar(labels[i]) else labels[i].tolist()
            })

    # Salva JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Testing completato. Risultati salvati in: {output_json_path}")

if __name__ == "__main__":
    test_npz_path = "Syntetic_dataset/Test_Syntetic.npz"
    output_json_path = "syntetic_1/test_syn_without_siamese.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_random_projection(test_npz_path, output_json_path, device)
