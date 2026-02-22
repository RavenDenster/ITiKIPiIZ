import re

def tokenize(sentence: str) -> list:
    pattern = r"[а-яА-ЯёЁ\-]+"
    return re.findall(pattern, sentence)