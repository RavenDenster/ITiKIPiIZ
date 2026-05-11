from datasets import Dataset, load_dataset, concatenate_datasets
from typing import Tuple
import config

def _parse_translation(example: dict, col_name: str) -> dict:
    trans = example[col_name]
    if isinstance(trans, list):
        trans = trans[0]
    if not isinstance(trans, dict):
        raise ValueError("Неожиданный формат поля перевода")
    return {"en": trans["en"], "ru": trans["ru"]}

def _find_translation_column(dataset: Dataset) -> str:
    for name in ("translation", "translations"):
        if name in dataset.column_names:
            return name
    raise ValueError("Столбец с переводом не найден")

def load_and_normalize(dataset_key: str) -> Tuple[Dataset, Dataset]:
    cfg = config.DATASETS_INFO[dataset_key]
    raw = None
    for loader in [
        lambda: load_dataset(cfg["source"], cfg["pair"]),
        lambda: load_dataset(cfg["source"], name=cfg["pair"]),
        lambda: load_dataset(cfg["source"], lang_pair=cfg["pair"]),
    ]:
        try:
            raw = loader()
            break
        except Exception:
            continue
    if raw is None:
        raise RuntimeError(f"Не удалось загрузить {cfg['source']} с парой {cfg['pair']}")

    split_name = "train" if "train" in raw else list(raw.keys())[0]
    base = raw[split_name]
    trans_col = _find_translation_column(base)

    normalized = base.map(lambda x: _parse_translation(x, trans_col),
                          remove_columns=base.column_names)
    normalized = normalized.shuffle(seed=config.SEED)

    sample_size = min(len(normalized), cfg["train_count"])
    subset = normalized.select(range(sample_size))
    split_dataset = subset.train_test_split(test_size=0.2, seed=config.SEED)

    test_size = min(len(split_dataset["test"]), cfg["valid_count"])
    train_ds = split_dataset["train"]
    test_ds = split_dataset["test"].select(range(test_size))
    return train_ds, test_ds

def create_mixed_datasets():
    news_train, news_test = load_and_normalize("news")
    books_train, books_test = load_and_normalize("books")
    mixed_train = concatenate_datasets([news_train, books_train]).shuffle(seed=config.SEED)
    mixed_test = concatenate_datasets([news_test, books_test]).shuffle(seed=config.SEED)
    return mixed_train, mixed_test