import os
from config import Config
from data_loader import load_all_data
from transformers import AutoTokenizer
from dataset import SentimentDataset
from torch.utils.data import DataLoader
from torch import load
from model import SentimentModel
from train import train_model
from predict import predict
import numpy as np

def main():
    cfg = Config()
    cfg.device = "cpu"

    train_df, val_df = load_all_data(cfg)

    from features import extract_features
    feats = np.vstack([extract_features(t) for t in train_df["text"]])
    feat_mean = feats.mean(axis=0)
    feat_std = feats.std(axis=0) + 1e-8

    tokenizer_path = os.path.join(cfg.output_dir, "tokenizer")
    if os.path.exists(tokenizer_path):
        print("Loading tokenizer from local folder...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        print("Downloading tokenizer from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")

    model_path = os.path.join(cfg.output_dir, "best_model.pt")
    model = SentimentModel(cfg.model_name)

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(load(model_path, map_location=cfg.device))
        model.to(cfg.device)
        print("Model loaded, skipping training.")
    else:
        print("No saved model found. Training from scratch...")
        train_dataset = SentimentDataset(train_df, tokenizer, cfg.max_length, feat_mean, feat_std)
        val_dataset = SentimentDataset(val_df, tokenizer, cfg.max_length, feat_mean, feat_std)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
        best_rmse = train_model(model, train_loader, val_loader, cfg)
        print(f"Training finished. Best RMSE: {best_rmse:.4f}")

    predict(input_path="input.txt", output_path="output.txt",
            model_dir=cfg.output_dir, cfg=cfg,
            tokenizer=tokenizer, feat_mean=feat_mean, feat_std=feat_std)

if __name__ == "__main__":
    main()