from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from typing import List, Dict
from datasets import Dataset
import config

def load_tokenizer_and_model(checkpoint: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    except ValueError:
        from transformers import MarianTokenizer
        tokenizer = MarianTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model.to(config.get_device())
    return model, tokenizer

def tokenize_function(batch: Dict[str, List[str]], tokenizer: AutoTokenizer):
    inputs = tokenizer(
        batch["en"],
        max_length=config.MAX_TOKENS,
        truncation=True,
        padding=False,
    )
    targets = tokenizer(
        text_target=batch["ru"],
        max_length=config.MAX_TOKENS,
        truncation=True,
        padding=False,
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

def prepare_tokenized_datasets(train_ds: Dataset, eval_ds: Dataset, tokenizer: AutoTokenizer):
    train_tokenized = train_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_tokenized = eval_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=eval_ds.column_names,
    )
    return train_tokenized, eval_tokenized

def create_data_collator(tokenizer, model):
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)