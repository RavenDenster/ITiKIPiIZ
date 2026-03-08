import re
import pymorphy3
from enum import Enum

# Словари терминов
ACTIONS = ["найти", "показать", "вывести", "отобразить", "искать", "посмотреть", "найди", "покажи", "выведи"]
ENTITIES = ["книга", "статья", "журнал", "доклад", "сборник", "книги", "статьи", "журналы"]
PERSONALITIES = ["толстой", "достоевский", "пушкин", "булгаков", "чехов", "тургенев", "гоголь", "лермонтов",
                 "иванов", "петров", "сидоров", "кузнецов", "смирнов"]
SUBJECTS = ["программирование", "лингвистика", "математика", "медицина", "алгоритм",
            "физика", "история", "химия", "информатика", "биология", "алгоритмы",
            "поэзия", "философия"]
PREPS = ["по", "на", "о", "про"]
CONNECTORS = ["и", "или"]
POST = ["после", "с"]
PRIOR = ["до", "по"]
INSIDE = ["в", "за"]
YEAR_TERMS = ["год", "года", "году"]
QUANTIFIERS = ["все", "всё"]
PUBLISH_MARKS = ["изданные", "выпущенные", "опубликованные", "издать"]

morph = pymorphy3.MorphAnalyzer(lang='ru')

class LexemeType(Enum):
    ACTION = 1
    ENTITY = 2
    PERSON = 3
    SUBJECT = 4
    PREP = 5
    CONNECTOR = 6
    YEAR_NUM = 7
    POST = 8
    PRIOR = 9
    INSIDE = 10
    YEAR_TERM = 11
    QUANTIFIER = 12
    PUBLISH_MARK = 13
    COMMA = 14
    UNKNOWN = 15

class Lexeme:
    def __init__(self, type_, value):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"{self.type.name}('{self.value}')"

def analyze(text):
    words = re.findall(r'[а-яёa-z0-9]+|[,]', text.lower())
    lexemes = []
    for w in words:
        if w == ',':
            lexemes.append(Lexeme(LexemeType.COMMA, ','))
            continue
        if w.isdigit():
            lexemes.append(Lexeme(LexemeType.YEAR_NUM, w))
            continue

        parsed = morph.parse(w)[0]
        lemma = parsed.normal_form.lower()

        if lemma in ACTIONS or w in ACTIONS:
            lexemes.append(Lexeme(LexemeType.ACTION, w))
        elif lemma in ENTITIES or w in ENTITIES:
            lexemes.append(Lexeme(LexemeType.ENTITY, w))
        elif lemma in PERSONALITIES or w in PERSONALITIES:
            lexemes.append(Lexeme(LexemeType.PERSON, w))
        elif lemma in SUBJECTS or w in SUBJECTS:
            lexemes.append(Lexeme(LexemeType.SUBJECT, w))
        elif lemma in PREPS or w in PREPS:
            lexemes.append(Lexeme(LexemeType.PREP, w))
        elif lemma in CONNECTORS or w in CONNECTORS:
            lexemes.append(Lexeme(LexemeType.CONNECTOR, w))
        elif lemma in POST or w in POST:
            lexemes.append(Lexeme(LexemeType.POST, w))
        elif lemma in PRIOR or w in PRIOR:
            lexemes.append(Lexeme(LexemeType.PRIOR, w))
        elif lemma in INSIDE or w in INSIDE:
            lexemes.append(Lexeme(LexemeType.INSIDE, w))
        elif lemma in YEAR_TERMS or w in YEAR_TERMS:
            lexemes.append(Lexeme(LexemeType.YEAR_TERM, w))
        elif lemma in QUANTIFIERS or w in QUANTIFIERS:
            lexemes.append(Lexeme(LexemeType.QUANTIFIER, w))
        elif lemma in PUBLISH_MARKS or w in PUBLISH_MARKS:
            lexemes.append(Lexeme(LexemeType.PUBLISH_MARK, w))
        else:
            # Если существительное в родительном падеже — считаем персоной
            if 'NOUN' in parsed.tag and 'gent' in parsed.tag:
                lexemes.append(Lexeme(LexemeType.PERSON, w))
            else:
                lexemes.append(Lexeme(LexemeType.UNKNOWN, w))
    return lexemes