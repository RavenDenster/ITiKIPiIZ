import random
import numpy as np
import torch
from pathlib import Path

SEED = 12345
MAX_TOKENS = 128
BASE_CHECKPOINT = str(Path(__file__).parent / "opus-mt-en-ru")

PROJECT_ROOT = Path(__file__).parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
METRICS_PATH = OUTPUT_ROOT / "metrics"
PREDICTIONS_PATH = OUTPUT_ROOT / "predictions"
MODEL_WEIGHTS = {
    "finetuned_news": OUTPUT_ROOT / "finetuned_news",
    "finetuned_books": OUTPUT_ROOT / "finetuned_books",
    "finetuned_mixed": OUTPUT_ROOT / "finetuned_mixed",
}

DATASETS_INFO = {
    "news": {
        "source": "Helsinki-NLP/news_commentary",
        "pair": "en-ru",
        "train_count": 3000,
        "valid_count": 300,
    },
    "books": {
        "source": "Helsinki-NLP/opus_books",
        "pair": "en-ru",
        "train_count": 3000,
        "valid_count": 300,
    },
}

def set_random_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_dirs():
    OUTPUT_ROOT.mkdir(exist_ok=True)
    METRICS_PATH.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)
    for path in MODEL_WEIGHTS.values():
        path.mkdir(parents=True, exist_ok=True)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")