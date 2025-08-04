import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder testuale solo con MLP
class TextEncoder(nn.Module):
    def __init__(self, input_dim=768):  
        super(TextEncoder, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
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

    def forward(self, x):  
        return F.normalize(self.mlp2(self.mlp1(x)), p=2, dim=1)


# Encoder visivo solo con MLP
class ImageEncoder(nn.Module):
    def __init__(self, input_dim=1024):  
        super(ImageEncoder, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
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

    def forward(self, x):  
        return F.normalize(self.mlp2(self.mlp1(x)), p=2, dim=1)


# Rete Siamese con solo MLP
class SiameseNetwork(nn.Module):
    def __init__(self, text_input_dim=768, image_input_dim=1024):
        super(SiameseNetwork, self).__init__()
        self.text_encoder = TextEncoder(input_dim=text_input_dim)
        self.image_encoder = ImageEncoder(input_dim=image_input_dim)

    def forward(self, text_embedding, image_embedding):
        text_proj = self.text_encoder(text_embedding)
        image_proj = self.image_encoder(image_embedding)
        return text_proj, image_proj


#####################################################################################################
class Similarity_Loss_Sigmoid_Vectorized(nn.Module):
    def __init__(self, temperature_init, train_matrix_path, val_matrix_path):
        super(Similarity_Loss_Sigmoid_Vectorized, self).__init__()
        
        # Parametri appresi
        self.temperature = nn.Parameter(torch.tensor(temperature_init, dtype=torch.float))
        self.bias = nn.Parameter(torch.tensor([-10.0], dtype=torch.float))

        # Caricamento matrici binarie (NxN)
        self.register_buffer("train_labels_matrix", torch.load(train_matrix_path).float())
        self.register_buffer("val_labels_matrix", torch.load(val_matrix_path).float())

    def forward(self, text_embeddings, image_embeddings, mod):

        # Normalizza (già fatto nel modello, ma lo ricontrolliamo per sicurezza)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

        B = text_embeddings.size(0)

        # Similarità coseno batch-wise
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T)  # (B, B)

        # Logit scalato con temperatura appresa e bias
        logit_matrix = (similarity_matrix * -self.temperature) + self.bias  # (B, B)

        # Estrai sottoblocco (batch x batch) dalla matrice globale
        if mod == 0:
            labels_full = self.train_labels_matrix
        else:
            labels_full = self.val_labels_matrix

        labels_matrix = labels_full[:B, :B]  # compatibile con il batch

        # Calcola la loss sigmoid leave-one-out
        loss_matrix = torch.log(1 / (1 + torch.exp(labels_matrix * logit_matrix)))
        loss = -torch.sum(loss_matrix) / B

        return loss
