import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from chexnet.chexnet import DenseNet121
import gc
import json

# RADBERT
class TextEncoder(nn.Module):
    def __init__(self, model_path="./bert/radbert_local", device='cpu'):
        super(TextEncoder, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert = AutoModelForMaskedLM.from_pretrained(model_path)

        self.mlp1 = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1024),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.bert(**tokens, output_hidden_states=True)
        cls_embedding = outputs.hidden_states[-1][:, 0, :]
        return F.normalize(self.mlp2(self.mlp1(cls_embedding)), p=2, dim=1)


# Modello CheXNet per le immagini
class ImageEncoder(nn.Module):
    def __init__(self, model_path="chexnet/model.pth", device='cpu'):
        super(ImageEncoder, self).__init__()
        self.device = device
        self.model = DenseNet121(out_size=1024)

        state_dict = torch.load(model_path, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        self.model.densenet121.classifier = nn.Identity()

        self.mlp1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image):
        image = image.to(self.device)
        with torch.no_grad():
            features = self.model(image)
        return F.normalize(self.mlp2(self.mlp1(features)), p=2, dim=1)


# Modello Siamese completo
class SiameseNetwork(nn.Module):
    def __init__(self, device):
        super(SiameseNetwork, self).__init__()
        self.device = device
        self.text_encoder = TextEncoder(device=self.device).to(self.device)
        self.image_encoder = ImageEncoder(device=self.device).to(self.device)

    def forward(self, text, image):
        text_embedding = self.text_encoder(text)
        image_embedding = self.image_encoder(image)
        return text_embedding, image_embedding



# Funzione di loss con matrici di similarità coseno
class Similarity_Loss_Sigmoid_Vectorized(nn.Module):
    def __init__(self, temperature, json_train="train_matrix_cosine_MIMIC.json", json_val="validation_matrix_cosine_MIMIC.json"):
        super(Similarity_Loss_Sigmoid_Vectorized, self).__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(temperature, dtype=torch.float))
        self.bias = nn.Parameter(torch.tensor([-10.0])) 

        embedding_list=[]
        with open(json_train, 'r') as f:
            embeddings = json.load(f)
  
        for k,v in embeddings.items():
            embedding_list.append(torch.tensor(v, dtype=torch.float))

        self.register_buffer("train_embeddings", torch.stack(embedding_list))
        embedding_list=[]    
        with open(json_val, 'r') as f:
            embeddings = json.load(f)
  
        for k,v in embeddings.items():
            embedding_list.append(torch.tensor(v, dtype=torch.float))

        self.register_buffer("val_embeddings", torch.stack(embedding_list))    
        
    def forward(self, text_embeddings, image_embeddings,mod):
        # 1. Normalizzazione degli embedding
        normalized_image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        normalized_text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        B = text_embeddings.shape[0]
        # 2. Calcolo della matrice di similarità 
        # La moltiplicazione di matrici (dot product) tra (B, D) e (D, B) risulta in (B, B)
        # dove B è la dimensione del batch e D è la dimensione dell'embedding.
        similarity_matrix = torch.matmul(normalized_image_embeddings, normalized_text_embeddings.T)
        # 3. Applicazione del logit e del bias
        logit_matrix = (similarity_matrix * -self.temperature) + self.bias

        # 4. Creazione della matrice delle label
        if mod==0:
            labels_matrix = torch.matmul(self.train_embeddings, self.train_embeddings.T)
        else:
            labels_matrix = torch.matmul(self.val_embeddings, self.val_embeddings.T)

        # 5. Calcolo della loss vettorizzata
        loss_matrix = torch.log(1 / (1 + torch.exp(labels_matrix * logit_matrix)))

        # 6. Somma e media della los
        total_loss = torch.sum(loss_matrix)
        return -(total_loss / B)
##############################################################################################################################
#Funznioen di loss con matrici di similarità con label binarie 0 1
class Similarity_Loss_Sigmoid_Vectorized_01(nn.Module):
    def __init__(self, temperature, train_matrix_path="train_matrix_bitmask_01_MIMIC.pt", val_matrix_path="validation_matrix_bitmask_01_MIMIC.pt"):
        super(Similarity_Loss_Sigmoid_Vectorized, self).__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(temperature, dtype=torch.float))
        self.bias = nn.Parameter(torch.tensor([-10.0])) 

        # Carica le matrici binarie delle etichette
        self.register_buffer("train_labels_matrix", torch.load(train_matrix_path).float())
        self.register_buffer("val_labels_matrix", torch.load(val_matrix_path).float())
        
    def forward(self, text_embeddings, image_embeddings, mod):
        normalized_image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        normalized_text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        B = text_embeddings.shape[0]

        # Similarità coseno tra immagini e testi
        similarity_matrix = torch.matmul(normalized_image_embeddings, normalized_text_embeddings.T)

        # Logit scalato
        logit_matrix = (similarity_matrix * -self.temperature) + self.bias

        # Estrai la matrice label corretta (sub-matrice corrispondente al batch)
        if mod == 0:
            labels_matrix_full = self.train_labels_matrix
        else:
            labels_matrix_full = self.val_labels_matrix

       
        labels_matrix = labels_matrix_full[:B, :B]

        # Calcolo della loss
        loss_matrix = torch.log(1 / (1 + torch.exp(labels_matrix * logit_matrix)))
        total_loss = torch.sum(loss_matrix)
        return -(total_loss / B)
