import torch
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from torch.nn.functional import normalize
import json 

from Siamese_Network_Base import SiameseNetwork


def test_model(model, df_test, image_folder, transform, device, output_json_path):
    model.eval()
    results = []

    with torch.no_grad():
        for index, row in df_test.iterrows():
            img_name = f"image_{index}.png"
            image_path = os.path.join(image_folder, img_name)

            try:
                img = Image.open(image_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Errore con {img_name}: {e}")
                continue

            text_report = row['report']
            text_input = [text_report]

            text_emb, image_emb = model(text_input, img_tensor)
            text_emb = normalize(text_emb, p=2, dim=1)
            image_emb = normalize(image_emb, p=2, dim=1)

            similarity = torch.dot(text_emb.squeeze(0), image_emb.squeeze(0)).item()

            results.append({
                "Row_ID": index,
                "image_embedding": image_emb.squeeze(0).cpu().tolist(),
                "text_embedding": text_emb.squeeze(0).cpu().tolist(),
                "positive_cosine_similarity": similarity
            })

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Risultati salvati in: {output_json_path}")


if __name__ == '__main__':
    csv_path = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/Base_evaluetion/mimic_data/test.csv"
    image_folder = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/Base_evaluetion/mimic_data/test_mimic"
    output_path = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/testset_Base.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(csv_path)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    model = SiameseNetwork(device=device).to(device)
    model.eval()

    test_model(model, df, image_folder, transform, device, output_json_path=output_path)
