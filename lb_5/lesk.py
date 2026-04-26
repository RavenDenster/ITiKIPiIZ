from utils import tokenize_and_lemmatize
from thesaurus import RuWordNetThesaurus, Sense


def _build_signature_bag(sense: Sense) -> set[str]:
    parts = []
    parts.extend(sense.synonyms)
    parts.append(sense.definition)
    parts.extend(sense.examples)
    parts.extend(sense.hypernyms)
    parts.extend(sense.hyponyms)

    combined = " ".join(p for p in parts if p)
    return set(tokenize_and_lemmatize(combined))


def _overlap_size(bag_a: set[str], bag_b: set[str]) -> int:
    return len(bag_a & bag_b)


def simple_lesk(context_lemmas: list[str], senses: list[Sense]) -> dict:
    context_set = set(context_lemmas)
    best_sense = None
    best_score = -1
    best_intersection = set()
    ranking = []

    for sense in senses:
        sig = _build_signature_bag(sense)
        inter = context_set & sig
        score = len(inter)

        ranking.append({
            "sense_id": sense.id,
            "synonyms": sense.synonyms,
            "definition": sense.definition,
            "score": score,
            "overlap": sorted(inter),
            "signature": sorted(sig),
        })

        if score > best_score:
            best_score = score
            best_sense = sense
            best_intersection = inter

    all_zero = all(item["score"] == 0 for item in ranking)
    is_tie = sum(1 for item in ranking if item["score"] == best_score) > 1

    return {
        "context": sorted(context_set),
        "best_sense": best_sense,
        "best_score": best_score,
        "best_overlap": sorted(best_intersection),
        "ranking": ranking,
        "all_zero": all_zero,
        "is_tie": is_tie,
    }


def _build_weighted_signature(sense: Sense) -> dict[str, set[str]]:
    def bag(text: str) -> set[str]:
        return set(tokenize_and_lemmatize(text)) if text.strip() else set()

    parts = {
        "synonyms": bag(" ".join(sense.synonyms)),
        "def_examples": bag(" ".join([sense.definition] + sense.examples)),
        "hypernyms": bag(" ".join(sense.hypernyms)),
        "hyponyms": bag(" ".join(sense.hyponyms)),
    }
    return parts


def weighted_lesk(context_lemmas: list[str], senses: list[Sense],
                  weights: dict[str, float] = None) -> dict:
    if weights is None:
        weights = {
            "synonyms": 3.0,
            "def_examples": 2.0,
            "hypernyms": 1.0,
            "hyponyms": 0.5,
        }

    context_set = set(context_lemmas)
    best_sense = None
    best_score = -1.0
    best_intersection = set()
    ranking = []

    for sense in senses:
        parts = _build_weighted_signature(sense)
        score = 0.0
        total_inter = set()
        details = {}

        for part_name, part_bag in parts.items():
            inter = context_set & part_bag
            w = weights.get(part_name, 0.0)
            score += len(inter) * w
            if inter:
                details[part_name] = {
                    "overlap": sorted(inter),
                    "weight": w,
                    "contribution": len(inter) * w,
                }
            total_inter.update(inter)

        ranking.append({
            "sense_id": sense.id,
            "synonyms": sense.synonyms,
            "definition": sense.definition,
            "score": score,
            "overlap": sorted(total_inter),
            "parts_detail": details,
            "signature": sorted(set().union(*parts.values())),
        })

        if score > best_score:
            best_score = score
            best_sense = sense
            best_intersection = total_inter

    all_zero = all(item["score"] == 0.0 for item in ranking)
    is_tie = sum(1 for item in ranking if abs(item["score"] - best_score) < 1e-9) > 1

    return {
        "context": sorted(context_set),
        "best_sense": best_sense,
        "best_score": best_score,
        "best_overlap": sorted(best_intersection),
        "ranking": ranking,
        "all_zero": all_zero,
        "is_tie": is_tie,
        "weights": weights,
    }