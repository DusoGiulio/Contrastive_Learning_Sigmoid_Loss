import numpy as np
import torch

#Genera la matrice di similarità binaria partendo dal dataset
def generate_label_matrix(npz_path, save_path):
    data = np.load(npz_path)
    labels = data["labels"]

    # Confronto tra ogni coppia i-j: 1 se stessa classe, 0 altrimenti
    label_matrix = (labels[:, None] == labels[None, :]).astype(np.float32)

    torch.save(torch.tensor(label_matrix), save_path)
    print(f"Saved label matrix to {save_path}, shape: {label_matrix.shape}")


if __name__ == "__main__":
    generate_label_matrix("Train_Syntetic.npz", "train_labels_Syn_matrix.pt")
    generate_label_matrix("Validation_Syntetic.npz", "val_labels_Syn_matrix.pt")
