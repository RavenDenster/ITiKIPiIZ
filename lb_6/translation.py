from typing import List
import torch
import config
from model_utils import load_tokenizer_and_model

def translate_sentences(texts: List[str], model, tokenizer, batch_size: int = 16) -> List[str]:
    device = config.get_device()
    model.eval()
    outputs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_TOKENS,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_length=config.MAX_TOKENS,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend(decoded)
    return outputs

def translate_single(text: str, model_path: str) -> str:
    model, tokenizer = load_tokenizer_and_model(model_path)
    return translate_sentences([text], model, tokenizer, batch_size=1)[0]