
from pathlib import Path
import numpy as np

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


if __name__ == "__main__":
    convert_dataset("/home/soph/ssd/dataset_raw")
