import os

DICT_XML_PATH = 'dict.opcorpora.xml'

CACHE_PATH = 'dict_cache.pkl'

POS_MAP = {
    'NOUN': 'S',
    'ADJF': 'A',
    'ADJS': 'A',
    'COMP': 'A',
    'VERB': 'V',
    'INFN': 'V',
    'PRTF': 'V',
    'PRTS': 'V',
    'GRND': 'V',
    'ADVB': 'ADV',
    'CONJ': 'CONJ',
    'PREP': 'PR',
    'NPRO': 'NI',
    'NUMR': 'NI',
    'PRED': 'NI',
    'PRCL': 'NI',
    'INTJ': 'NI'
}

DEFAULT_POS = 'NI'