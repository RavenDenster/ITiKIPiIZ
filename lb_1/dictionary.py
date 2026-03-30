import os
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from utils import normalize, levenshtein_distance
import config

class Dictionary:
    def __init__(self):
        self.word_map = defaultdict(list)
        self.words_by_length = defaultdict(list)

    def _build_length_index(self):
        self.words_by_length.clear()
        for norm in self.word_map:
            self.words_by_length[len(norm)].append(norm)

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

        self._build_length_index()

    def save_cache(self, cache_path: str):
        with open(cache_path, 'wb') as f:
            pickle.dump(dict(self.word_map), f)

    def load_cache(self, cache_path: str):
        with open(cache_path, 'rb') as f:
            self.word_map = defaultdict(list, pickle.load(f))
        self._build_length_index()

    def _get_approx_candidates(self, word: str) -> list:
        norm = normalize(word)
        best_dist = None
        best_norms = []

        min_len = max(1, len(norm) - config.APPROX_MAX_LENGTH_DIFF)
        max_len = len(norm) + config.APPROX_MAX_LENGTH_DIFF

        for length in range(min_len, max_len + 1):
            for candidate_norm in self.words_by_length.get(length, []):
                dist = levenshtein_distance(norm, candidate_norm)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_norms = [candidate_norm]
                elif dist == best_dist:
                    best_norms.append(candidate_norm)

        if best_dist is None or best_dist > config.APPROX_MAX_DIST:
            return []

        result = []
        for cn in best_norms:
            result.extend(self.word_map[cn])

        result.sort(key=lambda c: c['freq'], reverse=True)
        return result

    def get_candidates(self, word: str) -> list:
        norm = normalize(word)
        # print(self.word_map.get('крикнул', 'не найдено'))
        exact = self.word_map.get(norm, [])
        if exact:
            return exact
        return self._get_approx_candidates(word)