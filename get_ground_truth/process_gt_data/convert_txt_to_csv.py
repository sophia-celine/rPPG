import argparse
import csv
from pathlib import Path
import re
import tempfile

import numpy as np


SCIENTIFIC_NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)[Ee][+-]?\d+$"
)

def convert_dataset(root_dir):
    root_dir = Path(root_dir)
    txt_files = list(root_dir.glob("*/*.txt"))

    converted = 0
    errors = []

    for txt_file in txt_files:
        try:
            signal = np.loadtxt(txt_file, dtype=float)

            csv_file = txt_file.with_suffix(".csv")

            formatted_signal = [
                str(int(x)) if np.isfinite(x) else "nan"
                for x in np.atleast_1d(signal)
            ]

            np.savetxt(
                csv_file,
                formatted_signal,
                fmt="%s",
                delimiter=","
            )

            converted += 1
            print(f"Convertido: {txt_file} -> {csv_file}")

        except Exception as e:
            errors.append((txt_file, str(e)))
            print(f"Erro em {txt_file}: {e}")

    print(f"\nConversão concluída: {converted} arquivos.")

    if errors:
        print(f"Arquivos com erro: {len(errors)}")
        for file, error in errors:
            print(f"{file}: {error}")


def convert_scientific_csv_values(root_dir, dry_run=False):
    """Convert scientific-notation values in CSV files to integers.

    Files are searched recursively in ``root_dir``. Existing integers and
    decimal values not written in scientific notation are preserved.
    Scientific values are converted by truncating toward zero, matching
    ``int(float(value))``.
    """
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Pasta não encontrada: {root_dir}")

    converted_files = 0
    converted_values = 0
    errors = []
    csv_files = sorted(root_dir.rglob("*.csv"))

    for csv_file in csv_files:
        try:
            with csv_file.open("r", newline="", encoding="utf-8-sig") as source:
                rows = list(csv.reader(source))

            file_changed = False
            for row in rows:
                for column, value in enumerate(row):
                    text = value.strip()
                    if not SCIENTIFIC_NUMBER_PATTERN.fullmatch(text):
                        continue

                    number = float(text)
                    if not np.isfinite(number):
                        continue

                    integer_text = str(int(number))
                    row[column] = integer_text
                    converted_values += 1
                    file_changed = True

            if not file_changed or dry_run:
                continue

            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=csv_file.parent,
                delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                writer = csv.writer(destination, lineterminator="\n")
                writer.writerows(rows)

            temporary_path.replace(csv_file)
            converted_files += 1
            print(f"Convertido: {csv_file} ({converted_values} valores no total)")

        except Exception as exc:
            errors.append((csv_file, str(exc)))
            print(f"Erro em {csv_file}: {exc}")

    action = "Seriam convertidos" if dry_run else "Conversão concluída"
    print(
        f"\n{action}: {converted_values} valores em "
        f"{len(csv_files)} arquivos CSV."
    )
    print(f"Arquivos alterados: {converted_files}.")

    if errors:
        print(f"Arquivos com erro: {len(errors)}")
        for file, error in errors:
            print(f"{file}: {error}")

    return converted_files, converted_values, errors


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Converte valores em notação científica de arquivos CSV "
            "para números inteiros."
        )
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        help="Pasta raiz a percorrer recursivamente.",
    )
    parser.add_argument(
        "--txt-to-csv",
        action="store_true",
        help="Também executa a conversão existente de TXT para CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra a quantidade que seria convertida sem alterar arquivos.",
    )
    args = parser.parse_args()

    if args.txt_to_csv:
        convert_dataset(args.root_dir)

    convert_scientific_csv_values(
        args.root_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
