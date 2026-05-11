from translation import translate_sentences, load_tokenizer_and_model
import config

MODEL_PATH = str(config.MODEL_WEIGHTS["finetuned_news"])

def main():
    model, tokenizer = load_tokenizer_and_model(MODEL_PATH)

    my_texts = [
        "I’m gonna make him an offer he can’t refuse.",
        "May the Force be with you.",
        "I can't believe you did that!",
    ]

    translations = translate_sentences(my_texts, model, tokenizer)

    for src, tgt in zip(my_texts, translations):
        print(f"[EN] {src}")
        print(f"[RU] {tgt}")
        print("-" * 50)

if __name__ == "__main__":
    main()