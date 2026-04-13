import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer
from features import extract_features

class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, feat_mean=None, feat_std=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feat_mean = feat_mean if feat_mean is not None else np.zeros(14)
        self.feat_std = feat_std if feat_std is not None else np.ones(14)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        feats = extract_features(text)
        feats = (feats - self.feat_mean) / (self.feat_std + 1e-8)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "features": torch.tensor(feats, dtype=torch.float32),
            "score": torch.tensor(row["score"], dtype=torch.float32),
            "has_score": torch.tensor(row["has_score"], dtype=torch.float32),
            "is_irony": torch.tensor(row["is_irony"], dtype=torch.float32),
        }