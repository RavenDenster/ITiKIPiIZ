import re
import numpy as np

POS_WORDS = {"отлично", "прекрасно", "супер", "класс", "замечательно", "идеально",
             "понравилось", "доволен", "спасибо", "рекомендую", "люблю", "хорошо"}
NEG_WORDS = {"ужасно", "отвратительно", "кошмар", "плохо", "жаль", "обман", "хамство",
             "никогда", "бесполезно", "хуже", "проблема", "сломалось"}
IRONY_MARKERS = {"ага", "ну да", "конечно", "спасибо огромное", "просто супер", "молодцы"}
CONTRAST = {"но", "зато", "хотя", "однако"}

def extract_features(text: str) -> np.ndarray:
    text = str(text).lower()
    tokens = re.findall(r"[а-яa-zё]+", text)

    pos_cnt = sum(1 for w in tokens if w in POS_WORDS)
    neg_cnt = sum(1 for w in tokens if w in NEG_WORDS)
    irony_cnt = sum(1 for m in IRONY_MARKERS if m in text)
    contrast_cnt = sum(1 for c in CONTRAST if c in text)

    excls = text.count('!')
    quests = text.count('?')
    quotes = text.count('"') + text.count('«') + text.count('»')
    smile = text.count(':)') + text.count(':-)') + text.count('❤️')
    sad = text.count(':(') + text.count(':-(') + text.count('💩')

    upper_ratio = sum(1 for ch in text if ch.isupper()) / max(1, sum(1 for ch in text if ch.isalpha()))
    long_word_ratio = sum(1 for w in tokens if len(w) >= 8) / max(1, len(tokens))

    contradiction = 1.0 if (pos_cnt > 0 and neg_cnt > 0) else 0.0

    return np.array([
        len(text), len(tokens), pos_cnt, neg_cnt, irony_cnt, contrast_cnt,
        excls, quests, quotes, smile, sad, upper_ratio, long_word_ratio, contradiction
    ], dtype=np.float32)