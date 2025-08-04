import torch
import torch.optim as optim
import numpy as np
import os
import time
import gc

from Siamese_Network_for_syntetic import SiameseNetwork, Similarity_Loss_Sigmoid_Vectorized


def load_data_from_npz(npz_path):
    data = np.load(npz_path)
    text_embeddings = data["report_embeddings"]
    image_embeddings = data["image_embeddings"]
    labels = data["labels"]
    return text_embeddings, image_embeddings, labels


def train_leave_one_out_from_npz(model, 
                                 train_npz_path, 
                                 val_npz_path, 
                                 loss_fn, 
                                 optimizer, 
                                 batch_size, 
                                 epochs, 
                                 device, 
                                 patience=5, 
                                 save_folder="results"):
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model.to(device)
    model.train()

    # Caricamento dati
    text_train, image_train, labels_train = load_data_from_npz(train_npz_path)
    text_val, image_val, labels_val = load_data_from_npz(val_npz_path)

    num_samples_train = text_train.shape[0]
    num_samples_val = text_val.shape[0]

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()
        total_loss = 0

        indices = np.arange(num_samples_train)

        for start_idx in range(0, num_samples_train, batch_size):
            end_idx = min(start_idx + batch_size, num_samples_train)
            batch_indices = indices[start_idx:end_idx]

            batch_text = torch.tensor(text_train[batch_indices], dtype=torch.float32, device=device)
            batch_image = torch.tensor(image_train[batch_indices], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            text_emb, image_emb = model(batch_text, batch_image)
            loss = loss_fn(text_emb, image_emb, mod=0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            gc.collect()
            torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start_time
        avg_train_loss = total_loss / (num_samples_train)
        

        # Valutazione
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for start_idx in range(0, num_samples_val, batch_size):
                end_idx = min(start_idx + batch_size, num_samples_val)
                batch_text = torch.tensor(text_val[start_idx:end_idx], dtype=torch.float32, device=device)
                batch_image = torch.tensor(image_val[start_idx:end_idx], dtype=torch.float32, device=device)
                text_emb, image_emb = model(batch_text, batch_image)
                loss = loss_fn(text_emb, image_emb, mod=1)
                val_loss += loss.item()

            avg_val_loss = val_loss / (num_samples_val )
            print(f"{avg_train_loss:.6f};  {avg_val_loss:.6f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(save_folder, "best_model.pth"))
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"Epochs without improvement: {epochs_without_improvement}")
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered.")
                    break

        torch.save(model.state_dict(), os.path.join(save_folder, f"model_epoch_{epoch + 1}.pth"))
if __name__ == "__main__":
    train_npz = "Train_Syntetic.npz"
    val_npz = "Validation_Syntetic.npz"
    save_folder = "syntetic/models"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(text_input_dim=768, image_input_dim=1024).to(device)
    loss_fn = Similarity_Loss_Sigmoid_Vectorized(temperature_init=10,
                                                 train_matrix_path="train_labels_Syn_matrix.pt",
                                                 val_matrix_path="val_labels_Syn_matrix.pt").to(device)

    optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=0.001)

    train_leave_one_out_from_npz(model, 
                                 train_npz, 
                                 val_npz, 
                                 loss_fn, 
                                 optimizer, 
                                 batch_size=10000, 
                                 epochs=500, 
                                 device=device, 
                                 patience=5, 
                                 save_folder=save_folder)
