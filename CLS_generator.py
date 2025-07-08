import torch
import json
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm

# Configurazione
model_name = "bert-base-uncased"  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
csv_path = "mimic_data/test.csv"  

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

df = pd.read_csv(csv_path) 

output = {}

with torch.no_grad():
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        row_id = row["Row_ID"]
        text = str(row["report"])

        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = model(**tokens, output_hidden_states=False)
        cls_vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        cls_vec_normalized = torch.nn.functional.normalize(cls_vec, p=2, dim=0)  # usa dim=0 perché è un solo vettore
        output[int(row_id)] = cls_vec_normalized.tolist()   

# Salvataggio in JSON
with open("CLS_text_test.json", "w") as f:
    json.dump(output, f)

print(f"Salvati {len(output)} embedding CLS in 'CLS_text.json'")
