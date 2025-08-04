import numpy as np
import torch
import torch.nn.functional as F
import os
#Genera la matrice i similarità coseno partendo dal dataset
def load_text_embeddings(npz_path):
    data = np.load(npz_path)
    if "report_embeddings" not in data:
        raise KeyError("Il file non contiene 'text_embeds'")
    text_embeds = data["report_embeddings"]  # shape: [N, D]
    return torch.tensor(text_embeds, dtype=torch.float32)

def compute_similarity_matrix(text_embeds, normalize=True):
    if normalize:
        text_embeds = F.normalize(text_embeds, dim=1)  # cosine similarity
    similarity_matrix = torch.matmul(text_embeds, text_embeds.T)  # [N, N]
    return similarity_matrix

def main():
    npz_path = "Validation_Syntetic.npz"  
    output_path = "Val_Similarity_Matrix.pt"
    text_embeds = load_text_embeddings(npz_path)
    print(f"Loaded text embeddings: {text_embeds.shape}")  # [N, D]

    similarity_matrix = compute_similarity_matrix(text_embeds, normalize=True)
    print(f"Computed similarity matrix of shape: {similarity_matrix.shape}")  # [N, N]

    torch.save(similarity_matrix, output_path)
    print(f"Saved similarity matrix to: {output_path}")

if __name__ == "__main__":
    main()