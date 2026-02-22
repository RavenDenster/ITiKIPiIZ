# main.py
import sys
import os
import io
from dictionary import Dictionary
from tokenizer import tokenize
from lemmatizer import Lemmatizer
import config

def process_lines(lines_iter, lemmatizer, out_file):
    for line in lines_iter:
        line = line.strip()
        if not line:
            continue
        tokens = tokenize(line)
        results = lemmatizer.lemmatize_sentence(tokens)
        output = ' '.join(f"{token}{{{lemma}={pos}}}" for token, lemma, pos in results)
        print(output, file=out_file)

def main():
    dict_path = config.DICT_XML_PATH
    cache_path = config.CACHE_PATH

    dictionary = Dictionary()

    if os.path.exists(cache_path):
        print(f"Loading dictionary from cache {cache_path}...", file=sys.stderr)
        dictionary.load_cache(cache_path)
        print("Cache loaded.", file=sys.stderr)
    else:
        print(f"Cache not found. Loading dictionary from {dict_path}...", file=sys.stderr)
        if not os.path.exists(dict_path):
            print(f"Error: dictionary file {dict_path} not found.", file=sys.stderr)
            sys.exit(1)
        dictionary.load_from_xml(dict_path)
        print("Saving cache...", file=sys.stderr)
        dictionary.save_cache(cache_path)
        print("Cache saved.", file=sys.stderr)

    lemmatizer = Lemmatizer(dictionary)

    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        try:
            with open(input_file, 'r', encoding='utf-8') as inf, \
                 open(output_file, 'w', encoding='utf-8') as outf:
                process_lines(inf, lemmatizer, outf)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif len(sys.argv) > 1:
        input_file = sys.argv[1]
        try:
            with open(input_file, 'r', encoding='utf-8') as inf:
                process_lines(inf, lemmatizer, sys.stdout)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        process_lines(sys.stdin, lemmatizer, sys.stdout)

if __name__ == '__main__':
    main()

# python extract_tests.py annot.opcorpora.xml tests - Сгенерируйте тестовые файлы
# python main.py tests_input.txt tests_pred.txt - Запустите лемматизатор
# python evaluate.py tests_gold.txt tests_pred.txt - Оцените точность
