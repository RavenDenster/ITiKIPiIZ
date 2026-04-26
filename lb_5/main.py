from pathlib import Path
from ruwordnet import RuWordNet
from thesaurus import RuWordNetThesaurus
from utils import get_context_lemmas, lemmatize
from lesk import simple_lesk, weighted_lesk


def print_results(result: dict, target_word: str) -> None:
    best = result["best_sense"]
    print("ЗАДАЧА СНЯТИЯ ЛЕКСИЧЕСКОЙ НЕОДНОЗНАЧНОСТИ (WSD)")
    print(f"Слово: {target_word}")
    print(f"Лемма: {lemmatize(target_word)}")
    print(f"Леммы контекста: {', '.join(result['context'])}")
    print()

    if result.get("all_zero"):
        print("Пересечения с сигнатурами отсутствуют. Результат случаен.")
    elif result.get("is_tie"):
        print("Несколько значений имеют одинаковый максимальный вес. Результат неоднозначен.")
    print()

    print("Ранжирование значений:\n")
    for i, item in enumerate(result["ranking"], 1):
        print(f"[{i}] Sense ID: {item['sense_id']}")
        print(f"    Синонимы: {', '.join(item['synonyms'])}")
        print(f"    Определение: {item['definition']}")
        print(f"    Вес (score): {item['score']:.2f}")
        if "parts_detail" in item and item["parts_detail"]:
            for part, detail in item["parts_detail"].items():
                print(f"      - {part}: пересечение {detail['overlap']} (вклад {detail['contribution']:.2f})")
        elif "overlap" in item:
            if item["overlap"]:
                print(f"    Пересечение: {', '.join(item['overlap'])}")
        if not item.get("overlap"):
            print("    Пересечение: —")
        print()

    print("=" * 80)
    print("ВЫБРАННОЕ ЗНАЧЕНИЕ")
    print(f"ID синсета: {best.id}")
    print(f"Синонимы: {', '.join(best.synonyms)}")
    print(f"Определение: {best.definition}")
    print(f"Гиперонимы: {', '.join(best.hypernyms) if best.hypernyms else '—'}")
    print(f"Гипонимы: {', '.join(best.hyponyms) if best.hyponyms else '—'}")
    if result.get("best_overlap"):
        print(f"Совпавшие слова: {', '.join(result['best_overlap'])}")
    print("=" * 80)


def main():
    db_path = Path(__file__).parent / "ruwordnet-2021.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"Файл базы данных не найден: {db_path}\n"
            f"Поместите файл ruwordnet-2021.db в папку проекта."
        )
    print("Загрузка RuWordNet...")
    wn = RuWordNet(str(db_path))
    thesaurus = RuWordNetThesaurus(wn)

    examples = [
        {
            "sentence": "Я буду сдавать лабораторные работы вовремя и получу хорошую оценку за зачёт.",
            "target": "оценку"
        },
        {
            "sentence": "Директор дал высокую оценку работе сотрудников.",
            "target": "оценку"
        },
        {
            "sentence": "На даче мы посадили зелёный лук и чеснок.",
            "target": "лук"
        },
        {
            "sentence": "Лук это хорошое оружие.",
            "target": "лук"
        },
                {
            "sentence": "Лук это хорошое оружие.",
            "target": "лук"
        },
        {"sentence": "Мастер заменил дверную ручку на стальную.", "target": "ручку"},
        {"sentence": "Высший свет собрался на бал к княгине.", "target": "свет"},

    ]

    for ex in examples:
        sent = ex["sentence"]
        target = ex["target"]
        print(f"\nАнализ предложения: «{sent}»")
        print(f"Целевое слово: «{target}»\n")

        ctx = get_context_lemmas(sent, target)

        senses = thesaurus.get_senses(target)
        if not senses:
            print(f"Значений слова «{target}» в тезаурусе не найдено.\n")
            continue

        res = weighted_lesk(ctx, senses, weights={
            "synonyms": 3.0,
            "def_examples": 2.0,
            "hypernyms": 1.0,
            "hyponyms": 0.5
        })

        print_results(res, target)


if __name__ == "__main__":
    main()