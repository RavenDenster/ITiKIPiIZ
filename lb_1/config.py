import os

DICT_XML_PATH = 'dict.opcorpora.xml'

CACHE_PATH = 'dict_cache.pkl'

POS_MAP = {
    'NOUN': 'S', # существительное
    'ADJF': 'A', # прилогательное
    'ADJS': 'A',
    'COMP': 'A',
    'VERB': 'V', # глагов
    'INFN': 'V',
    'PRTF': 'V',
    'PRTS': 'V',
    'GRND': 'V',
    'ADVB': 'ADV', # наречие
    'CONJ': 'CONJ', # союз
    'PREP': 'PR', # предлог
    'NPRO': 'NI', # неизменяемые части речи (местоимение-существительное)
    'NUMR': 'NI', # (числительное)
    'PRED': 'NI', # (предикатив)
    'PRCL': 'NI', # (частица)
    'INTJ': 'NI' # (междометие)
}

DEFAULT_POS = 'NI'

APPROX_MAX_LENGTH_DIFF = 2
APPROX_MAX_DIST = 3     