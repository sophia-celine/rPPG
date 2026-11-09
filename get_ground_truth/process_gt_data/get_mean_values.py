import math
import sys
from pathlib import Path

import pandas as pd

from config import APP_SETTINGS


DATASET_CSV_PATH = APP_SETTINGS.dataset_raw_file
MEASUREMENTS = {
    "sinal_FC": "Frequência cardíaca média durante a gravação",
    "sinal_FR_impedancia": "Taxa de respiração média durante a gravação",
    "sinal_SpO2": "Saturação média durante a gravação",
}


def patient_folder_name(patient_id):
    normalized_id = str(patient_id).strip().lower()
    if "." in normalized_id:
        base_id, index = normalized_id.rsplit(".", 1)
        if base_id and index.isdigit():
            normalized_id = f"{base_id}_{index}"
    return "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in normalized_id
    )


def signal_file_path(patient_id, signal_reference):
    filename = Path(str(signal_reference).strip().replace("\\", "/")).name
    return Path(APP_SETTINGS.patient_data_root) / patient_folder_name(patient_id) / filename


def mean_signal_value(signal_path):
    values = []
    with signal_path.open("r", encoding="utf-8-sig") as signal_file:
        for line_number, line in enumerate(signal_file, start=1):
            value_text = line.strip()
            if not value_text:
                continue
            try:
                value = float(value_text)
            except ValueError as exc:
                raise ValueError(
                    f"valor inválido na linha {line_number}: {value_text!r}"
                ) from exc
            if math.isfinite(value):
                values.append(value)

    if not values:
        raise ValueError("o arquivo não contém valores numéricos válidos")
    return sum(values) / len(values)


def update_dataset(csv_path=DATASET_CSV_PATH):
    csv_path = Path(csv_path)
    dataframe = pd.read_csv(csv_path)
    required_columns = {"Id do paciente", *MEASUREMENTS.keys()}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes na planilha: "
            + ", ".join(sorted(missing_columns))
        )

    for measurement_column in MEASUREMENTS.values():
        if measurement_column not in dataframe.columns:
            dataframe[measurement_column] = pd.NA
        dataframe[measurement_column] = pd.NA

    errors = []
    updated_values = 0
    for row_index, row in dataframe.iterrows():
        patient_id = row["Id do paciente"]
        if pd.isna(patient_id) or not str(patient_id).strip():
            errors.append(f"Linha {row_index + 2}: ID do paciente vazio.")
            continue

        for signal_column, measurement_column in MEASUREMENTS.items():
            signal_reference = row[signal_column]
            if pd.isna(signal_reference) or not str(signal_reference).strip():
                errors.append(
                    f"Linha {row_index + 2} ({patient_id}): "
                    f"referência {signal_column} vazia."
                )
                continue

            signal_path = signal_file_path(patient_id, signal_reference)
            try:
                mean_value = mean_signal_value(signal_path)
            except (OSError, ValueError) as exc:
                errors.append(
                    f"Linha {row_index + 2} ({patient_id}), "
                    f"{signal_column} ({signal_path}): {exc}"
                )
                continue

            dataframe.at[row_index, measurement_column] = mean_value
            updated_values += 1

    dataframe.to_csv(csv_path, index=False)
    return len(dataframe), updated_values, errors


def main():
    try:
        patient_count, updated_values, errors = update_dataset()
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Erro ao processar a planilha {DATASET_CSV_PATH}: {exc}")
        return 1

    print(
        f"Planilha atualizada: {patient_count} linha(s), "
        f"{updated_values} média(s) calculada(s)."
    )
    if errors:
        print(f"{len(errors)} problema(s) encontrado(s):")
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
