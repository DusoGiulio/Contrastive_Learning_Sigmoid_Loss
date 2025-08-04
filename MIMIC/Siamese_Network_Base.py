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
        
        self.projection = nn.Linear(768, 1024)

    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.bert(**tokens, output_hidden_states=True)
        cls_embedding = outputs.hidden_states[-1][:, 0, :]  
        
        projected = self.projection(cls_embedding)         
        return F.normalize(projected, p=2, dim=1)


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


    def forward(self, image):
        image = image.to(self.device)
        with torch.no_grad():
            features = self.model(image)
        return F.normalize(features, p=2, dim=1)


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