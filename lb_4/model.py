import torch
import torch.nn as nn
from transformers import AutoModel

class SentimentModel(nn.Module):
    def __init__(self, model_name, feature_dim=14, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion = nn.Linear(hidden_size + 64, hidden_size)
        self.score_head = nn.Linear(hidden_size, 1)
        self.irony_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, features):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_out.last_hidden_state[:, 0]
        feat_emb = self.feat_proj(features)
        combined = torch.cat([cls_emb, feat_emb], dim=1)
        fused = self.fusion(combined)
        fused = torch.relu(fused)

        score_logit = self.score_head(fused).squeeze(-1)
        score = torch.sigmoid(score_logit) * 9 + 1

        irony_logit = self.irony_head(fused).squeeze(-1)
        return {"score": score, "irony_logit": irony_logit}