import evaluate
import pandas as pd
import json
from typing import List
from model_utils import load_tokenizer_and_model
from translation import translate_sentences
from data_loader import load_and_normalize
import config

def bleu_score(predictions: List[str], references: List[str]) -> float:
    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=predictions,
                            references=[[r] for r in references])
    return result["score"]

def save_examples(model_name: str, test_name: str,
                  sources: List[str], preds: List[str], refs: List[str], n=10):
    records = []
    for src, pred, ref in zip(sources[:n], preds[:n], refs[:n]):
        records.append({"source": src, "prediction": pred, "reference": ref})
    path = config.PREDICTIONS_PATH / f"{model_name}_{test_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def evaluate_single(model_name: str, checkpoint: str, test_name: str, test_ds):
    model, tokenizer = load_tokenizer_and_model(checkpoint)
    sources = test_ds["en"]
    references = test_ds["ru"]
    predictions = translate_sentences(sources, model, tokenizer)
    score = bleu_score(predictions, references)
    save_examples(model_name, test_name, sources, predictions, references)
    return {"model": model_name, "test_dataset": test_name, "BLEU": round(score, 2)}

def run_full_evaluation():
    _, news_test = load_and_normalize("news")
    _, books_test = load_and_normalize("books")

    models = {
        "baseline": config.BASE_CHECKPOINT,
        "news_tuned": str(config.MODEL_WEIGHTS["finetuned_news"]),
        "books_tuned": str(config.MODEL_WEIGHTS["finetuned_books"]),
        "mixed_tuned": str(config.MODEL_WEIGHTS["finetuned_mixed"]),
    }
    tests = {"news": news_test, "books": books_test}

    results = []
    for model_key, ckpt in models.items():
        for test_key, test_ds in tests.items():
            result = evaluate_single(model_key, ckpt, test_key, test_ds)
            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(config.METRICS_PATH / "bleu_results.csv", index=False)
    print(df)
    return df