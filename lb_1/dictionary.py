import os
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from utils import normalize

class Dictionary:
    def __init__(self):
        self.word_map = defaultdict(list)

    def load_from_xml(self, xml_path: str):
        known_pos = {'NOUN', 'ADJF', 'ADJS', 'COMP', 'VERB', 'INFN',
                     'PRTF', 'PRTS', 'GRND', 'ADVB', 'CONJ', 'PREP',
                     'NPRO', 'NUMR', 'PRED', 'PRCL', 'INTJ'}

        context = ET.iterparse(xml_path, events=('end',))
        for event, elem in context:
            if elem.tag == 'lemma':
                l_elem = elem.find('l')
                if l_elem is None:
                    elem.clear()
                    continue
                lemma = l_elem.get('t')
                freq = elem.get('freq')
                freq = int(freq) if freq is not None else 0

                pos = None
                for g in l_elem.findall('g'):
                    gram = g.get('v')
                    if gram in known_pos:
                        pos = gram
                        break
                if pos is None:
                    elem.clear()
                    continue

                for f_elem in elem.findall('f'):
                    wordform = f_elem.get('t')
                    norm = normalize(wordform)
                    self.word_map[norm].append({
                        'lemma': lemma,
                        'pos': pos,
                        'freq': freq
                    })

                elem.clear()

        for norm in self.word_map:
            self.word_map[norm].sort(key=lambda c: c['freq'], reverse=True)

    def save_cache(self, cache_path: str):
        with open(cache_path, 'wb') as f:
            pickle.dump(dict(self.word_map), f)

    def load_cache(self, cache_path: str):
        with open(cache_path, 'rb') as f:
            self.word_map = defaultdict(list, pickle.load(f))

    def get_candidates(self, word: str) -> list:
        norm = normalize(word)
        return self.word_map.get(norm, [])