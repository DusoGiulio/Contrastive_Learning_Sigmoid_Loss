import torch
import torch.optim as optim
import pandas as pd
import os
import time
from torchvision import transforms
from PIL import Image
import gc 
from Siamese_Network import SiameseNetwork, Similarity_Loss_Sigmoid_Vectorized


def load_dataframe(df, image_folder, row_ids, transform, device):
    images = []
    texts = []
    for row_id in row_ids:
        img_name = f"image_{row_id}.png"
        img_path = os.path.join(image_folder, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            if transform:
                img = transform(img)
            images.append(img)
            text = df.loc[df['Row_ID'] == row_id, 'report'].iloc[0]
            texts.append(text)
        except FileNotFoundError:
            print(f"Image not found: {img_path} (for Row_ID {row_id})")
            images.append(torch.zeros(3, 224, 224, device=device))
            texts.append("")
        except (OSError, IndexError) as e:
            print(f"Error for Row_ID {row_id}: {e}")
            images.append(torch.zeros(3, 224, 224, device=device))
            texts.append("")
    return torch.stack(images), texts
#__________________________________________________________________________________________
#Funzione di valutazione
def evaluate(model, df_val, image_folder, loss_fn, batch_size, device):
    model.eval() 
    total_loss = 0
    total_rows = len(df_val)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Disabilita il calcolo dei gradienti per l'inferenza
    with torch.no_grad():
        for start_idx in range(0, total_rows, batch_size):
            batch_df = df_val.iloc[start_idx:min(start_idx + batch_size, total_rows)]
            current_batch_row_ids = batch_df["Row_ID"].tolist()

            batch_text_embeddings = []
            batch_image_embeddings = []

            # Ciclo sulle singole coppie immagine-testo all'interno del batch di valutazione
            for i,row_id in enumerate(current_batch_row_ids):
                # Carica una singola immagine e il suo testo
                single_image, single_text_list = load_dataframe(df_val, image_folder, [row_id], transform, device)
                
                single_text = single_text_list[0]
                single_image = single_image.to(device)

                # Calcola gli embedding per la singola coppia
                text_emb, image_emb = model([single_text], single_image)
                
                batch_text_embeddings.append(text_emb)
                batch_image_embeddings.append(image_emb)
                
                if i%1500==0:
                    # Svuota la cache CUDA
                    gc.collect()
                    torch.cuda.empty_cache()

            # Concatena gli embedding raccolti per l'intero batch di valutazione
            if len(batch_text_embeddings) > 0 and len(batch_image_embeddings) > 0:
                batch_text_embeddings_tensor = torch.cat(batch_text_embeddings, dim=0)
                batch_image_embeddings_tensor = torch.cat(batch_image_embeddings, dim=0)

                # Calcola la loss sul batch di embedding
                loss = loss_fn(batch_text_embeddings_tensor, batch_image_embeddings_tensor,1)
                
                total_loss += loss.item()
            gc.collect()
            torch.cuda.empty_cache()


    avg_loss = total_loss / total_rows


    model.train() # Riporta il modello in modalità addestramento
    return avg_loss
#__________________________________________________________________________________________
# Funzione di addestramento
def train_leave_one_out_from_folder(model, df_train, df_val, image_folder, loss_fn, optimizer, batch_size, epochs, device, patience=5, save_folder="1_vs_all"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model.to(device)
    model.train()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    total_rows = len(df_train)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        
        # Ciclo sui batch
        for start_idx in range(0, total_rows, batch_size):
            batch_df = df_train.iloc[start_idx:min(start_idx + batch_size, total_rows)]
            current_batch_row_ids = batch_df["Row_ID"].tolist()

            batch_text_embeddings = []
            batch_image_embeddings = []

            optimizer.zero_grad() 

            # Ciclo sulle singole coppie immagine-testo all'interno del batch
            for i, row_id in enumerate(current_batch_row_ids):
                # Carica una singola immagine e il suo testo
                single_image, single_text_list = load_dataframe(df_train, image_folder, [row_id], transform, device)
                
                single_text = single_text_list[0] 
                single_image = single_image.to(device) 

                # Calcola gli embedding per la singola coppia
                text_emb, image_emb = model([single_text], single_image) 
                
                batch_text_embeddings.append(text_emb)
                batch_image_embeddings.append(image_emb)
                if i%1500==0:
                    gc.collect()
                    torch.cuda.empty_cache()

            gc.collect()
            torch.cuda.empty_cache()    
            # Concatena gli embedding raccolti per l'intero batch
            if len(batch_text_embeddings) > 0 and len(batch_image_embeddings) > 0:
                batch_text_embeddings_tensor = torch.cat(batch_text_embeddings, dim=0)
                batch_image_embeddings_tensor = torch.cat(batch_image_embeddings, dim=0)

                # Calcola la loss sul batch di embedding
                loss = loss_fn(batch_text_embeddings_tensor, batch_image_embeddings_tensor,0 )

                # Otimizzazione
                loss.backward() 
                total_loss += loss.item()

            batches_per_epoch = (total_rows)
            
            optimizer.step() 
            
            optimizer.zero_grad() 
            gc.collect()
            torch.cuda.empty_cache() 

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Calcola la loss media 
        avg_loss = total_loss / (batches_per_epoch)
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Epoch Time: {epoch_duration:.2f} seconds")

        # Validazione dei risultati
        val_loss = evaluate(model, df_val, image_folder, loss_fn, batch_size, device)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.6f}")

        # Early stopping 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_folder, "best_model_test.pth"))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"Epochs without improvement: {epochs_without_improvement}")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # Salvataggio
        torch.save(model.state_dict(), os.path.join(save_folder, f"model_epoch_{epoch}.pth"))

if __name__ == '__main__':
    csv_path_train = "mimic_data/train.csv"
    csv_path_val = "mimic_data/validation.csv"
    image_folder = "mimic_data/train_mimic"
    save_folder = "modello"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_train = pd.read_csv(csv_path_train)
    df_val = pd.read_csv(csv_path_val)
    loss_fn = Similarity_Loss_Sigmoid_Vectorized(10)
    loss_fn = loss_fn.to(device)
    model = SiameseNetwork(device=device).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()), lr=0.001)
    
    train_leave_one_out_from_folder(model, df_train, df_val, image_folder, loss_fn, optimizer, batch_size=11413, epochs=100,device=device, patience=5, save_folder=save_folder)