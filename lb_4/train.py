# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_rmse = float("inf")
    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"], batch["features"])
            loss_score = nn.MSELoss()(outputs["score"], batch["score"])
            loss_irony = nn.BCEWithLogitsLoss()(outputs["irony_logit"], batch["is_irony"])
            loss = loss_score + cfg.irony_weight * loss_irony

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), loss_s=loss_score.item(), loss_i=loss_irony.item())

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Valid"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch["input_ids"], batch["attention_mask"], batch["features"])
                mask = batch["has_score"].cpu().numpy() > 0.5
                if mask.sum() == 0:
                    continue
                pred = outputs["score"].detach().cpu().numpy()[mask]
                true = batch["score"].cpu().numpy()[mask]
                preds.extend(pred)
                targets.extend(true)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        print(f"Epoch {epoch+1} | Val RMSE: {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))
    return best_rmse