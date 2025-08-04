import numpy as np
import torch

# Genera la matrice di similarità: 1 se stessa classe, -1 altrimenti
def generate_label_matrix(npz_path, save_path):
    data = np.load(npz_path)
    labels = data["labels"]

    # Confronto tra ogni coppia i-j
    label_matrix = np.where(
        labels[:, None] == labels[None, :], 
        1.0, 
        -1.0
    ).astype(np.float32)

    torch.save(torch.tensor(label_matrix), save_path)
    print(f"Saved label matrix to {save_path}, shape: {label_matrix.shape}")


if __name__ == "__main__":
    generate_label_matrix("Train_Syntetic.npz", "train_labels_Syn_matrix.pt")
    generate_label_matrix("Validation_Syntetic.npz", "val_labels_Syn_matrix.pt")
