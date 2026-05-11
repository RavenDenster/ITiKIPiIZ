import config
from train import train_news, train_books, train_mixed
from evaluation import run_full_evaluation
from translation import translate_single

def main():
    config.set_random_seed()
    config.create_dirs()
    print(f"Устройство: {config.get_device()}")

    # print("Обучение на новостях...")
    # train_news()
    # print("Обучение на книгах...")
    # train_books()
    # print("Обучение на смешанном корпусе...")
    # train_mixed()

    print("Оценка всех моделей...")
    df = run_full_evaluation()
    print("Результаты сохранены в output/metrics/bleu_results.csv")

    example = "Machine learning is transforming our world."
    translation = translate_single(example, str(config.MODEL_WEIGHTS["finetuned_news"]))
    print(f"Пример перевода: {example} -> {translation}")

if __name__ == "__main__":
    main()