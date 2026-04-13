import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from features import POS_WORDS, NEG_WORDS

def load_all_data(cfg):
    df_rev = pd.read_json("train-ru-reviews-classification.jsonl", lines=True)
    df_rev = df_rev[df_rev["label_text"].isin(["negative", "neutral", "positive"])]
    score_map = {"negative": 2.0, "neutral": 5.5, "positive": 9.0}
    df_rev["score"] = df_rev["label_text"].map(score_map)
    df_rev["is_irony"] = 0
    df_rev["source"] = "ru_reviews"
    # сэмплируем
    df_rev = df_rev.sample(min(len(df_rev), cfg.max_ru_reviews), random_state=cfg.seed)

    df_bank = pd.read_csv("russian_bank_reviews.csv")
    if "rating_value" in df_bank.columns:
        df_bank["rating"] = pd.to_numeric(df_bank["rating_value"], errors="coerce")
    elif "rating" in df_bank.columns:
        df_bank["rating"] = pd.to_numeric(df_bank["rating"], errors="coerce")
    else:
        raise KeyError("No rating column found in bank reviews CSV")
    df_bank = df_bank.dropna(subset=["rating"])
    df_bank["score"] = 1 + (df_bank["rating"] - 1) * 9 / 4
    title_col = "review_title" if "review_title" in df_bank.columns else "title"
    text_col = "review" if "review" in df_bank.columns else "text"
    df_bank["text"] = df_bank[title_col].fillna("") + " " + df_bank[text_col].fillna("")
    df_bank["is_irony"] = 0
    df_bank["source"] = "bank"
    df_bank = df_bank.sample(min(len(df_bank), cfg.max_bank), random_state=cfg.seed)

    df_gis = pd.read_json("org-reviews.jsonl", lines=True)
    df_gis["rating"] = pd.to_numeric(df_gis["rating"], errors="coerce")
    df_gis = df_gis.dropna(subset=["rating"])
    df_gis["score"] = 1 + (df_gis["rating"] - 1) * 9 / 4
    df_gis["is_irony"] = 0
    df_gis["source"] = "2gis"
    df_gis = df_gis.sample(min(len(df_gis), cfg.max_2gis), random_state=cfg.seed)

    df_tweet = pd.read_parquet("train-RuSentiTweet.parquet")
    df_tweet = df_tweet[df_tweet["label"].isin(["negative", "neutral", "positive"])]
    tweet_map = {"negative": 2.5, "neutral": 5.5, "positive": 8.5}
    df_tweet["score"] = df_tweet["label"].map(tweet_map)
    df_tweet["is_irony"] = 0
    df_tweet["source"] = "tweet"
    df_tweet = df_tweet.sample(min(len(df_tweet), cfg.max_rusentitweet), random_state=cfg.seed)

    import json
    with open("russian_jokes.json", "r", encoding="utf-8") as f:
        jokes_data = json.load(f)
    if isinstance(jokes_data, list):
        df_jokes = pd.DataFrame({"text": jokes_data})
    elif isinstance(jokes_data, dict) and "jokes" in jokes_data:
        df_jokes = pd.DataFrame({"text": jokes_data["jokes"]})
    else:
        raise ValueError("Unexpected jokes JSON structure")
    df_jokes["score"] = np.nan
    df_jokes["is_irony"] = 1
    df_jokes["source"] = "jokes"
    df_jokes = df_jokes.sample(min(len(df_jokes), cfg.max_jokes), random_state=cfg.seed)

    main = pd.concat([df_rev, df_bank, df_gis, df_tweet], ignore_index=True)

    main["contradiction"] = main["text"].apply(
        lambda x: 1 if (any(w in str(x).lower() for w in POS_WORDS) and
                        any(w in str(x).lower() for w in NEG_WORDS)) else 0
    )
    ironic_main = main[main["contradiction"] == 1].copy()
    ironic_main["is_irony"] = 1
    ironic_main["source"] += "_ironic"
    ironic_main = ironic_main.sample(min(5000, len(ironic_main)), random_state=cfg.seed)

    full = pd.concat([main, df_jokes, ironic_main], ignore_index=True)
    full["has_score"] = full["score"].notna().astype(float)
    full["score"] = full["score"].fillna(5.0)

    train_df, val_df = train_test_split(
        full, test_size=0.1, random_state=cfg.seed,
        stratify=full["has_score"].astype(int) if len(full["has_score"].unique()) > 1 else None
    )
    return train_df, val_df