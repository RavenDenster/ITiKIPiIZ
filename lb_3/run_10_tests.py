import subprocess
import sys

def main():
    text_files = [f"hp{i}.txt" for i in range(1, 11)]
    ref_files = [f"hp{i}_ref.txt" for i in range(1, 11)]

    input_lines = []
    input_lines.append("2")
    input_lines.append(str(len(text_files)))
    input_lines.extend(text_files) 
    input_lines.append("y")     
    input_lines.append("2")          
    input_lines.extend(ref_files)     

    stdin_data = "\n".join(input_lines) + "\n"

    result = subprocess.run(
        [sys.executable, "main.py"],
        input=stdin_data,
        text=True,
        encoding='utf-8',
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\n[Ошибка] main.py завершился с кодом {result.returncode}")
    else:
        print("\nТестирование успешно завершено.")

if __name__ == "__main__":
    main()


