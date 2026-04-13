import torch
import numpy as np
from transformers import AutoTokenizer
from model import SentimentModel
from dataset import SentimentDataset
from torch.utils.data import DataLoader
import pandas as pd
import os

def predict(input_path="input.txt", output_path="output.txt", model_dir="./sentiment_model",
            cfg=None, tokenizer=None, feat_mean=None, feat_std=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer is None:
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        if os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Please run training first.")

    if feat_mean is None or feat_std is None:
        mean_path = os.path.join(model_dir, "feat_mean.npy")
        std_path = os.path.join(model_dir, "feat_std.npy")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            feat_mean = np.load(mean_path)
            feat_std = np.load(std_path)
        else:
            feat_mean = np.zeros(14)
            feat_std = np.ones(14)

    model = SentimentModel(cfg.model_name)
    model_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with open(input_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    df = pd.DataFrame({"text": texts, "score": 5.0, "has_score": 0, "is_irony": 0})
    dataset = SentimentDataset(df, tokenizer, cfg.max_length, feat_mean, feat_std)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"], batch["features"])
            scores.extend(outputs["score"].cpu().numpy().tolist())

    with open(output_path, "w", encoding="utf-8") as f:
        for s in scores:
            f.write(f"{s:.6f}\n")
    print(f"Predictions saved to {output_path}")