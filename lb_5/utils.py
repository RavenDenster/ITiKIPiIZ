import re
from razdel import tokenize
from pymorphy3 import MorphAnalyzer

_morph = MorphAnalyzer()

STOP_WORDS = {
    "и", "в", "во", "на", "с", "со", "а", "но", "или", "что", "как",
    "к", "ко", "из", "за", "по", "под", "у", "о", "об", "от", "до",
    "не", "ни", "же", "ли", "бы", "я", "ты", "он", "она", "оно", "мы",
    "вы", "они", "это", "тот", "та", "те", "этот", "эта", "эти",
    "буду", "будет", "будут", "был", "была", "были", "есть", "весь", "мой", "твой", "свой"
}


def is_russian_token(token: str) -> bool:
    return bool(re.fullmatch(r"[а-яёА-ЯЁ\-]+", token))


def lemmatize(token: str) -> str:
    return _morph.parse(token.lower())[0].normal_form


def tokenize_and_lemmatize(text: str, drop_stopwords: bool = True) -> list[str]:
    result = []
    for tok in tokenize(text):
        word = tok.text.lower()
        if not is_russian_token(word):
            continue
        lemma = lemmatize(word)
        if drop_stopwords and lemma in STOP_WORDS:
            continue
        result.append(lemma)
    return result


def get_context_lemmas(sentence: str, target_word: str, exclude_target: bool = True) -> list[str]:
    all_lemmas = tokenize_and_lemmatize(sentence)
    target_lemma = lemmatize(target_word.lower())

    if exclude_target:
        all_lemmas = [w for w in all_lemmas if w != target_lemma]

    return all_lemmas