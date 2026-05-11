from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from pathlib import Path
from data_loader import load_and_normalize, create_mixed_datasets
from model_utils import (
    load_tokenizer_and_model,
    prepare_tokenized_datasets,
    create_data_collator,
)
import config
import torch

def fine_tune_model(train_ds, eval_ds, output_dir: Path, num_epochs: int = 1):
    model, tokenizer = load_tokenizer_and_model(config.BASE_CHECKPOINT)
    train_tokenized, eval_tokenized = prepare_tokenized_datasets(train_ds, eval_ds, tokenizer)
    collator = create_data_collator(tokenizer, model)

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=config.SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        processing_class=tokenizer,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def train_news():
    news_train, news_test = load_and_normalize("news")
    fine_tune_model(news_train, news_test, config.MODEL_WEIGHTS["finetuned_news"])

def train_books():
    books_train, books_test = load_and_normalize("books")
    fine_tune_model(books_train, books_test, config.MODEL_WEIGHTS["finetuned_books"])

def train_mixed():
    mixed_train, mixed_test = create_mixed_datasets()
    fine_tune_model(mixed_train, mixed_test, config.MODEL_WEIGHTS["finetuned_mixed"])