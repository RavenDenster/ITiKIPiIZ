from ruwordnet import RuWordNet
from utils import lemmatize, tokenize_and_lemmatize


class Sense:
    def __init__(self, synset_id: str, synonyms: list[str], definition: str,
                 examples: list[str], hypernyms: list[str], hyponyms: list[str]):
        self.id = synset_id
        self.synonyms = synonyms
        self.definition = definition
        self.examples = examples
        self.hypernyms = hypernyms
        self.hyponyms = hyponyms

    def __repr__(self):
        return f"Sense(id={self.id}, lemmas={self.synonyms})"


class RuWordNetThesaurus:

    def __init__(self, wn: RuWordNet):
        self.wn = wn

    @staticmethod
    def _safe_str(value) -> str:
        return str(value).strip() if value else ""

    def _synset_title(self, synset) -> str:
        if hasattr(synset, "title") and synset.title:
            return self._safe_str(synset.title)
        return ""

    def _synset_definition(self, synset) -> str:
        for attr in ("definition", "description", "text", "title"):
            val = getattr(synset, attr, None)
            if val:
                return self._safe_str(val)
        return ""

    def _synset_examples(self, synset) -> list[str]:
        if hasattr(synset, "examples") and synset.examples:
            return [self._safe_str(e) for e in synset.examples if self._safe_str(e)]
        return []

    def _synset_synonyms(self, synset) -> list[str]:
        names = []
        if hasattr(synset, "senses"):
            for sense in synset.senses:
                name = getattr(sense, "name", None)
                if name:
                    names.append(self._safe_str(name).lower())
        seen = set()
        out = []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _related_names(self, synsets) -> list[str]:
        res = []
        for syn in synsets:
            title = self._synset_title(syn).lower()
            if title:
                res.append(title)
            else:
                if hasattr(syn, "senses"):
                    names = [self._safe_str(s.name).lower() for s in syn.senses if getattr(s, "name", None)]
                    if names:
                        res.append(", ".join(names))
        seen = set()
        uniq = []
        for item in res:
            if item not in seen:
                seen.add(item)
                uniq.append(item)
        return uniq

    def get_senses(self, word: str) -> list[Sense]:
        lemma = lemmatize(word)
        raw_senses = self.wn.get_senses(lemma)
        result = []

        for raw in raw_senses:
            syn = raw.synset
            sid = self._safe_str(getattr(raw, "id", "")) or self._safe_str(getattr(syn, "id", ""))

            synonyms = self._synset_synonyms(syn)
            if not synonyms:
                raw_name = getattr(raw, "name", None)
                synonyms = [self._safe_str(raw_name).lower()] if raw_name else [lemma]

            definition = self._synset_definition(syn)
            examples = self._synset_examples(syn)

            hypernyms = self._related_names(getattr(syn, "hypernyms", []))
            hyponyms = self._related_names(getattr(syn, "hyponyms", []))

            result.append(Sense(
                synset_id=sid,
                synonyms=synonyms,
                definition=definition,
                examples=examples,
                hypernyms=hypernyms,
                hyponyms=hyponyms,
            ))
        return result