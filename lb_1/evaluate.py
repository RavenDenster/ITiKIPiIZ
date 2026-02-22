import sys
import re

def parse_output_line(line):
    tokens = line.strip().split()
    result = []
    for token in tokens:
        match = re.match(r'(.+)\{(.+)=(.+)\}', token)
        if match:
            word, lemma, tag = match.groups()
            result.append((word, lemma, tag))
    return result

def load_gold(filepath):
    gold_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                gold_data.append(parse_output_line(line))
    return gold_data

def load_predicted(filepath):
    pred_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pred_data.append(parse_output_line(line))
    return pred_data

def evaluate(gold, pred):
    total_tokens = 0
    correct_lemmas = 0
    correct_tags = 0
    details = []

    for sent_g, sent_p in zip(gold, pred):
        if len(sent_g) != len(sent_p):
            print(f"Warning: sentence length mismatch", file=sys.stderr)
            continue
        for (word_g, lemma_g, tag_g), (word_p, lemma_p, tag_p) in zip(sent_g, sent_p):
            if word_g != word_p:
                print(f"Warning: token mismatch: '{word_g}' vs '{word_p}'", file=sys.stderr)
                continue
            total_tokens += 1
            if lemma_g == lemma_p:
                correct_lemmas += 1
            if tag_g == tag_p:
                correct_tags += 1
            details.append((word_g, lemma_g, tag_g, lemma_p, tag_p))

    lemma_acc = correct_lemmas / total_tokens if total_tokens else 0
    tag_acc = correct_tags / total_tokens if total_tokens else 0

    return {
        'total_tokens': total_tokens,
        'lemma_accuracy': lemma_acc,
        'tag_accuracy': tag_acc,
        'details': details
    }

def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <gold_file> <pred_file>")
        sys.exit(1)
    gold_file = sys.argv[1]
    pred_file = sys.argv[2]
    gold = load_gold(gold_file)
    pred = load_predicted(pred_file)
    metrics = evaluate(gold, pred)
    print(f"Total tokens: {metrics['total_tokens']}")
    print(f"Lemma accuracy: {metrics['lemma_accuracy']:.4f}")
    print(f"Tag accuracy: {metrics['tag_accuracy']:.4f}")
    with open('evaluation_details.txt', 'w', encoding='utf-8') as f:
        f.write("word\tgold_lemma\tgold_tag\tpred_lemma\tpred_tag\tlemma_correct\ttag_correct\n")
        for w, lg, tg, lp, tp in metrics['details']:
            f.write(f"{w}\t{lg}\t{tg}\t{lp}\t{tp}\t{lg==lp}\t{tg==tp}\n")

if __name__ == '__main__':
    main()