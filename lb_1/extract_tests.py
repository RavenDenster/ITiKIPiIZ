import os
import pickle
import xml.etree.ElementTree as ET
import sys
from utils import map_pos

def parse_all_sentences(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sentences_data = []
    for text in root.findall('text'):
        for para in text.findall('.//paragraph'):
            for sentence in para.findall('sentence'):
                source = sentence.find('source')
                if source is None or source.text is None:
                    continue
                input_text = source.text.strip()
                tokens_elem = sentence.find('tokens')
                if tokens_elem is None:
                    continue
                gold_tokens = []
                for token in tokens_elem.findall('token'):
                    text_token = token.get('text')
                    tfr = token.find('tfr')
                    if tfr is None:
                        continue
                    v = tfr.find('v')
                    if v is None:
                        continue
                    l = v.find('l')
                    if l is None:
                        continue
                    lemma = l.get('t')
                    pos = None
                    for g in l.findall('g'):
                        gram = g.get('v')
                        if gram in {'NOUN','ADJF','ADJS','COMP','VERB','INFN','PRTF','PRTS','GRND','ADVB','CONJ','PREP','NPRO','NUMR','PRED','PRCL','INTJ'}:
                            pos = gram
                            break
                    if pos is None:
                        continue
                    short_tag = map_pos(pos)
                    gold_tokens.append(f"{text_token}{{{lemma}={short_tag}}}")
                if gold_tokens:
                    sentences_data.append((input_text, ' '.join(gold_tokens)))
    return sentences_data

def extract_sentences_cached(xml_path, output_gold, output_input, max_sentences=20, cache_file=None):
    if cache_file is None:
        cache_file = os.path.splitext(xml_path)[0] + '_cache.pkl'

    rebuild = True
    if os.path.exists(cache_file):
        cache_mtime = os.path.getmtime(cache_file)
        xml_mtime = os.path.getmtime(xml_path)
        if cache_mtime >= xml_mtime:
            try:
                with open(cache_file, 'rb') as f:
                    all_sentences = pickle.load(f)
                rebuild = False
                print(f"Loaded {len(all_sentences)} sentences from cache.", file=sys.stderr)
            except Exception:
                rebuild = True
    
    if rebuild:
        print(f"Parsing XML {xml_path}...", file=sys.stderr)
        all_sentences = parse_all_sentences(xml_path)
        print(f"Parsed {len(all_sentences)} sentences. Saving cache...", file=sys.stderr)
        with open(cache_file, 'wb') as f:
            pickle.dump(all_sentences, f)

    selected = all_sentences[:max_sentences]
    input_lines = [s[0] for s in selected]
    gold_lines = [s[1] for s in selected]
    
    with open(output_gold, 'w', encoding='utf-8') as f:
        f.write('\n'.join(gold_lines))
    with open(output_input, 'w', encoding='utf-8') as f:
        f.write('\n'.join(input_lines))
    print(f"Extracted {len(selected)} sentences.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python extract_tests.py <annot.opcorpora.xml> <output_prefix>")
        sys.exit(1)
    xml_path = sys.argv[1]
    prefix = sys.argv[2]
    extract_sentences_cached(xml_path, f"{prefix}_gold.txt", f"{prefix}_input.txt", max_sentences=1000)