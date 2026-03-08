import sys
from lexical_analyzer import analyze
from query_parser import QueryParser

def main(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, line in enumerate(lines, 1):
        print(f"Запрос {i}: {line}")
        lexemes = analyze(line)
        print(f"Лексемы: {lexemes}")
        parser = QueryParser(lexemes)
        try:
            
            tree = parser.parse()
            print("Результат: УСПЕХ")
            print("Дерево разбора:")
            print(tree)
        except SyntaxError as e:
            print("Результат: НЕУДАЧА")
            print("Диагностика:", e)
        print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Укажите имя файла с запросами.")