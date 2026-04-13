from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    model_name: str = "cointegrated/rubert-tiny2"
    max_length: int = 128
    batch_size: int = 32
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    irony_weight: float = 0.3

    max_ru_reviews: int = 50000
    max_bank: int = 10000
    max_2gis: int = 2500
    max_rusentitweet: int = 10000
    max_jokes: int = 15000

    output_dir: str = "./sentiment_model"
    device: str = "cuda"