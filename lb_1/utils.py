import re
from config import POS_MAP, DEFAULT_POS

def normalize(word: str) -> str:
    return word.lower().replace('ё', 'е')

def map_pos(opencorpora_pos: str) -> str:
    return POS_MAP.get(opencorpora_pos, DEFAULT_POS)