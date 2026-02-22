from utils import map_pos

class Lemmatizer:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def lemmatize_sentence(self, tokens: list) -> list:
        results = []
        for i, token in enumerate(tokens):
            candidates = self.dictionary.get_candidates(token)
            if not candidates:
                results.append((token, token, 'UNK'))
                continue

            if len(candidates) == 1:
                cand = candidates[0]
                results.append((token, cand['lemma'], map_pos(cand['pos'])))
                continue

            chosen = self._disambiguate(token, candidates, results, i)
            if chosen is None:
                chosen = candidates[0]
            results.append((token, chosen['lemma'], map_pos(chosen['pos'])))
        return results

    def _disambiguate(self, token: str, candidates: list, left_results: list, idx: int):
        left_pos = left_results[idx-1][2] if idx > 0 else None

        pos_to_cands = {}
        for c in candidates:
            sp = map_pos(c['pos'])
            pos_to_cands.setdefault(sp, []).append(c)

        # Правило 1: для однобуквенных слов предпочитаем союз (CONJ)
        if len(token) == 1 and 'CONJ' in pos_to_cands:
            return max(pos_to_cands['CONJ'], key=lambda c: c['freq'])

        # Правило 2: для двухбуквенных слов предпочитаем предлог (PR)
        if len(token) == 2 and 'PR' in pos_to_cands:
            return max(pos_to_cands['PR'], key=lambda c: c['freq'])

        # Правило 3: в начале предложения предпочитаем глагол
        if idx == 0 and 'V' in pos_to_cands:
            return max(pos_to_cands['V'], key=lambda c: c['freq'])

        # Правило 4: после предлога выбираем существительное
        if left_pos == 'PR' and 'S' in pos_to_cands:
            return max(pos_to_cands['S'], key=lambda c: c['freq'])

        # Правило 5: после прилагательного выбираем существительное
        if left_pos == 'A' and 'S' in pos_to_cands:
            return max(pos_to_cands['S'], key=lambda c: c['freq'])

        # Правило 6: после глагола предпочитаем наречие, затем существительное
        if left_pos == 'V':
            if 'ADV' in pos_to_cands:
                return max(pos_to_cands['ADV'], key=lambda c: c['freq'])
            if 'S' in pos_to_cands:
                return max(pos_to_cands['S'], key=lambda c: c['freq'])

        # Правило 7: после существительного выбираем прилагательное
        if left_pos == 'S' and 'A' in pos_to_cands:
            return max(pos_to_cands['A'], key=lambda c: c['freq'])

        # Правило 8: выбор между глаголом и существительным (без предлога) – глагол
        if 'V' in pos_to_cands and 'S' in pos_to_cands and left_pos != 'PR':
            return max(pos_to_cands['V'], key=lambda c: c['freq'])

        return None