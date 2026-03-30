import re
import math
import collections
from typing import List, Optional

MAX_SUMMARY_LENGTH = 300
USE_IDF = True
MIN_SENTENCE_LENGTH = 10
WEIGHT_FIRST_SENTENCE = 1.5
WEIGHT_LAST_SENTENCE = 1.3

try:
    import pymorphy3
    MORPH = pymorphy3.MorphAnalyzer()
    LEMMATIZATION_AVAILABLE = True
except ImportError:
    print("Предупреждение: pymorphy3 не установлен. Лемматизация отключена.")
    LEMMATIZATION_AVAILABLE = False
    def dummy_lemma(word: str) -> str:
        return word.lower()
    MORPH = dummy_lemma

STOP_WORDS = set([
    'и', 'в', 'во', 'не', 'на', 'я', 'он', 'она', 'оно', 'они', 'мы', 'ты', 'вы',
    'это', 'этот', 'эта', 'эти', 'тот', 'та', 'те', 'такой', 'также', 'ещё', 'уже',
    'если', 'то', 'только', 'было', 'была', 'были', 'был', 'будет', 'будем', 'будете',
    'будут', 'быть', 'как', 'так', 'вот', 'для', 'по', 'за', 'из', 'от', 'до', 'без',
    'через', 'при', 'над', 'под', 'между', 'же', 'бы', 'что', 'чтобы', 'который',
    'которая', 'которое', 'которые', 'все', 'весь', 'вся', 'всех', 'всё', 'свой',
    'его', 'её', 'их', 'ее', 'него', 'нему', 'ней', 'нем', 'ними', 'этих', 'этим',
    'этими', 'этих', 'этом', 'этом', 'эту', 'этой'
])

def split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])', text)
    return [s.strip() for s in parts if s.strip()]

def lemmatize_word(word: str) -> str:
    if LEMMATIZATION_AVAILABLE:
        return MORPH.parse(word)[0].normal_form
    return word.lower()

def tokenize_sentence(sent: str) -> List[str]:
    words = re.findall(r'\b\w+\b', sent.lower())
    lemmas = [lemmatize_word(w) for w in words]
    return [l for l in lemmas if l not in STOP_WORDS and len(l) > 1]

def compute_idf(corpus_words_lists: List[List[str]]) -> dict:
    N = len(corpus_words_lists)
    df = collections.Counter()
    for words in corpus_words_lists:
        for w in set(words):
            df[w] += 1
    idf = {}
    for word, freq in df.items():
        idf[word] = math.log((N + 1) / (freq + 1)) + 1
    return idf

def summarize_document(
    text: str,
    max_length: int,
    idf: Optional[dict] = None,
    min_sent_len: int = MIN_SENTENCE_LENGTH,
    weight_first: float = WEIGHT_FIRST_SENTENCE,
    weight_last: float = WEIGHT_LAST_SENTENCE
) -> str:
    sentences = split_sentences(text)
    sentences = [s for s in sentences if len(s) >= min_sent_len]
    if not sentences:
        return ""

    sent_words = [tokenize_sentence(s) for s in sentences]
    all_words = [w for sw in sent_words for w in sw]

    if idf is not None:
        tf = collections.Counter(all_words)
        word_weights = {w: tf.get(w, 0) * idf.get(w, 0) for w in set(all_words)}
    else:
        word_weights = collections.Counter(all_words)

    sent_weights = [sum(word_weights.get(w, 0) for w in sw) for sw in sent_words]

    if sent_weights:
        sent_weights[0] *= weight_first
        if len(sent_weights) > 1:
            sent_weights[-1] *= weight_last

    sorted_idx = sorted(range(len(sent_weights)), key=lambda i: sent_weights[i], reverse=True)
    selected = set()
    current_len = 0
    for idx in sorted_idx:
        sent_len = len(sentences[idx])
        if current_len + sent_len <= max_length:
            selected.add(idx)
            current_len += sent_len

    if not selected:
        first = sentences[0]
        return first[:max_length]

    ordered = sorted(selected)
    summary = ' '.join(sentences[i] for i in ordered)
    if len(summary) > max_length:
        summary = summary[:max_length]
    return summary

def summarize_all(texts: List[str]) -> List[str]:
    if not texts:
        return []
    if USE_IDF:
        corpus_words = []
        for text in texts:
            sentences = split_sentences(text)
            sent_words = [tokenize_sentence(s) for s in sentences]
            all_words = [w for sw in sent_words for w in sw]
            corpus_words.append(all_words)
        idf = compute_idf(corpus_words)
    else:
        idf = None
    return [summarize_document(t, MAX_SUMMARY_LENGTH, idf) for t in texts]

# ======================== ВСТРОЕННАЯ ОЦЕНКА ROUGE ========================
def get_ngrams(tokens, n):
    return set(zip(*[tokens[i:] for i in range(n)]))

def rouge_n(reference: str, candidate: str, n: int = 1) -> float:
    ref_tokens = re.findall(r'\b\w+\b', reference.lower())
    cand_tokens = re.findall(r'\b\w+\b', candidate.lower())
    if not cand_tokens or not ref_tokens:
        return 0.0
    ref_ngrams = get_ngrams(ref_tokens, n)
    cand_ngrams = get_ngrams(cand_tokens, n)
    overlap = len(ref_ngrams & cand_ngrams)
    precision = overlap / len(cand_ngrams) if cand_ngrams else 0
    recall = overlap / len(ref_ngrams) if ref_ngrams else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def rouge_l(reference: str, candidate: str) -> float:
    ref_tokens = re.findall(r'\b\w+\b', reference.lower())
    cand_tokens = re.findall(r'\b\w+\b', candidate.lower())
    if not cand_tokens or not ref_tokens:
        return 0.0
    dp = [[0] * (len(cand_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i, r in enumerate(ref_tokens, 1):
        for j, c in enumerate(cand_tokens, 1):
            if r == c:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[len(ref_tokens)][len(cand_tokens)]
    precision = lcs_len / len(cand_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def evaluate_rouge(reference_summaries: List[str], candidate_summaries: List[str]) -> dict:
    r1, r2, rl = [], [], []
    for ref, cand in zip(reference_summaries, candidate_summaries):
        r1.append(rouge_n(ref, cand, 1))
        r2.append(rouge_n(ref, cand, 2))
        rl.append(rouge_l(ref, cand))
    return {
        'rouge1': sum(r1) / len(r1),
        'rouge2': sum(r2) / len(r2),
        'rougeL': sum(rl) / len(rl)
    }

def read_text_from_console(prompt: str) -> str:
    print(prompt)
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return " ".join(lines)

def read_texts_from_console() -> List[str]:
    n = int(input("Введите количество документов: "))
    texts = []
    for i in range(1, n+1):
        text = read_text_from_console(f"\nВведите текст {i} (завершите пустой строкой):")
        if text:
            texts.append(text)
        else:
            print(f"Текст {i} пуст, пропускаем.")
    return texts

def read_texts_from_files() -> List[str]:
    n = int(input("Введите количество документов: "))
    texts = []
    for i in range(1, n+1):
        path = input(f"Введите путь к файлу с текстом {i}: ").strip()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            if text:
                texts.append(text)
            else:
                print(f"Файл {path} пуст, пропускаем.")
        except Exception as e:
            print(f"Ошибка чтения файла {path}: {e}. Пропускаем.")
    return texts

def read_references_from_console(n: int) -> List[str]:
    refs = []
    print("Введите эталонные рефераты.")
    for i in range(1, n+1):
        ref = read_text_from_console(f"\nЭталон для документа {i} (завершите пустой строкой):")
        if ref:
            refs.append(ref)
        else:
            print(f"Эталон для документа {i} пуст, будет пропущен.")
            refs.append("")
    return refs

def read_references_from_files(n: int) -> List[str]:
    refs = []
    print("Введите пути к файлам с эталонными рефератами.")
    for i in range(1, n+1):
        path = input(f"Путь к эталону для документа {i}: ").strip()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                ref = f.read().strip()
            if ref:
                refs.append(ref)
            else:
                print(f"Файл {path} пуст, эталон пропущен.")
                refs.append("")
        except Exception as e:
            print(f"Ошибка чтения файла {path}: {e}. Эталон пропущен.")
            refs.append("")
    return refs

def main():
    print("=== Автоматическое построение рефератов ===\n")
    print("Выберите способ ввода текстов:")
    print("1 - Ввести тексты вручную (по одному)")
    print("2 - Загрузить тексты из файлов")
    choice = input("Ваш выбор (1/2): ").strip()
    if choice == "1":
        texts = read_texts_from_console()
    elif choice == "2":
        texts = read_texts_from_files()
    else:
        print("Неверный выбор. Завершение.")
        return
    if not texts:
        print("Нет текстов для обработки.")
        return
    print("\nОбработка текстов...")
    summaries = summarize_all(texts)
    print("\n" + "="*60)
    print("РЕФЕРАТЫ:")
    print("="*60)
    for i, summ in enumerate(summaries, 1):
        print(f"\nДокумент {i} (длина: {len(summ)} символов):")
        print(summ)
        print("-"*40)
    print("\nХотите оценить качество рефератов с помощью ROUGE? (y/n)")
    answer = input().strip().lower()
    if answer == 'y':
        print("Выберите способ ввода эталонных рефератов:")
        print("1 - Ввести вручную")
        print("2 - Загрузить из файлов")
        subchoice = input("Ваш выбор (1/2): ").strip()
        if subchoice == "1":
            refs = read_references_from_console(len(texts))
        elif subchoice == "2":
            refs = read_references_from_files(len(texts))
        else:
            print("Неверный выбор. Оценка отменена.")
            return
        valid_pairs = [(r, c) for r, c in zip(refs, summaries) if r]
        if valid_pairs:
            valid_refs, valid_cands = zip(*valid_pairs)
            avg_scores = evaluate_rouge(list(valid_refs), list(valid_cands))
            print("\n=== Оценка ROUGE (F-мера) ===")
            for metric, score in avg_scores.items():
                print(f"{metric.upper()}: {score:.4f}")
        else:
            print("Нет валидных эталонных рефератов.")
    print("\nРабота завершена.")

if __name__ == "__main__":
    main()