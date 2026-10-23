import csv
from pathlib import Path
from rPPGAnalysis import rPPGAnalysis
from sklearn.metrics import mean_absolute_error, mean_squared_error
from copy import deepcopy
from openpyxl import Workbook

HR_WINDOW_SIZE = 60
RESPIRATION_WINDOW_SIZE = 60
# RESULTS_CSV = r"C:\Users\Sophia\Documents\rPPG_data\results\results_mp.csv"
RESULTS_CSV = "/home/soph/rppg/rPPG_data/results/results_mp.csv"

def save_results(results: dict, output_path: str | Path = RESULTS_CSV) -> None:
    """Save per-patient metrics grouped by metric, then by method."""
    patient_names = list(results)
    metric_rows = []

    metric_groups = (
        ("hr_results", ("mae", "rmse", "mape", "snr"), "hr"),
        ("rr_results", ("mae", "rmse", "mape"), "rr"),
        ("correlation_results", ("pearson_coef", "best_lag"), ""),
    )

    for result_key, metric_names, suffix in metric_groups:
        methods = list(dict.fromkeys(
            method
            for patient_results in results.values()
            for method in patient_results.get(result_key, {})
        ))
        for metric in metric_names:
            for method in methods:
                row_name = (
                    f"{metric}_{suffix}_{method}"
                    if suffix
                    else f"{metric}_{method}"
                )
                values = [
                    results[patient]
                    .get(result_key, {})
                    .get(method, {})
                    .get(metric, "")
                    for patient in patient_names
                ]
                metric_rows.append([row_name, *values])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", *patient_names])
        writer.writerows(metric_rows)

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Resultados"
    worksheet.append(["metric", *patient_names])
    for row in metric_rows:
        worksheet.append(row)
    workbook.save(output_path.with_suffix(".xlsx"))


def find_patient_info(
    root_path: str | Path,
) -> tuple[str, str, str | None, str | None, str]:
    root_path = Path(root_path).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_path}")

    paths = sorted(root_path.rglob("*"))
    files = [path for path in paths if path.is_file()]
    directories = [path for path in paths if path.is_dir()]

    def find_unique(description: str, candidates: list[Path]) -> Path:
        if not candidates:
            raise FileNotFoundError(f"No {description} found under {root_path}")
        if len(candidates) > 1:
            matches = ", ".join(str(path) for path in candidates)
            raise ValueError(f"Multiple {description} found under {root_path}: {matches}")
        return candidates[0]

    def find_optional_unique(
        description: str,
        candidates: list[Path],
    ) -> str | None:
        if len(candidates) > 1:
            matches = ", ".join(str(path) for path in candidates)
            raise ValueError(f"Multiple {description} found under {root_path}: {matches}")
        return str(candidates[0]) if candidates else None

    return (
        str(find_unique(
            "AVI video",
            [path for path in files if path.suffix.casefold() == ".avi"],
        )),
        str(find_unique(
            "ECG CSV file",
            [
                path for path in files
                if path.name.casefold().startswith("ecg")
                and path.suffix.casefold() == ".csv"
            ],
        )),
        find_optional_unique(
            "PPG TXT file",
            [
                path for path in files
                if path.name.casefold().startswith("ppg")
                and path.suffix.casefold() == ".txt"
            ],
        ),
        find_optional_unique(
            "respiration TXT file",
            [
                path for path in files
                if path.name.casefold().startswith("resp")
                and path.suffix.casefold() == ".txt"
            ],
        ),
        str(find_unique(
            "BVP directory",
            [path for path in directories if path.name.casefold() == "bvp_results"],
        )),
    )

def list_patients(root_path: Path) -> list[str]:
    root_path = Path(root_path).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_path}")

    patient_dirs = [path for path in root_path.iterdir() if path.is_dir()]
    return sorted(str(path) for path in patient_dirs)


def list_patients_from_batches(root_path: Path) -> list[str]:
    root_path = Path(root_path).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_path}")

    patient_dirs = [
        patient_dir
        for batch_dir in root_path.iterdir()
        if batch_dir.is_dir() and batch_dir.name.casefold().startswith("batch")
        for patient_dir in batch_dir.iterdir()
        if patient_dir.is_dir()
    ]
    return sorted(str(path) for path in patient_dirs)


# patients_paths = list_patients(r"A:\dataset_raw\batch3")
# patients_paths = list_patients_from_batches(r"A:\dataset_raw")
patients_paths = list_patients_from_batches("/home/soph/ssd/dataset_raw")
# patients_paths = list_patients("/home/soph/ssd/dataset_raw/batch7")

patients_info = {}

for patient_path in patients_paths:
    try:
        VIDEO_PATH, ECG_DATA_PATH, PPG_DATA_PATH, RESPIRATION_DATA_PATH, RPPG_FOLDER_PATH = find_patient_info(patient_path)
        patient = Path(str(patient_path).replace("\\", "/").rstrip("/")).name
        patients_info[patient] = {
            "video_path": VIDEO_PATH,   
            "ecg_data_path": ECG_DATA_PATH,
            "ppg_data_path": PPG_DATA_PATH,
            "respiration_data_path": RESPIRATION_DATA_PATH,
            "rppg_folder_path": RPPG_FOLDER_PATH,
        }
    except (FileNotFoundError, ValueError) as e:
        print(f"Error processing patient at {patient_path}: {e}")

print("Patients info:", patients_info)

results = {}
methods = {'ICA': [], 
              'POS': [], 
              'CHROM': [],
              'GREEN': [],
              'PBV': [],
              'LGI': [],
              'OMIT': []}
hr_ecg_all = deepcopy(methods)
hr_rppg_all = deepcopy(methods)
snr_all = deepcopy(methods)
rr_gt_all = deepcopy(methods)
rr_rppg_all = deepcopy(methods)
pearson_coef_all = deepcopy(methods)

for patient in patients_info:
    patient_info = patients_info[patient]
    print(f"Analyzing patient: {patient}")
    patient_analysis = rPPGAnalysis(
        video_path=patient_info["video_path"],
        ecg_data_path=patient_info["ecg_data_path"],
        ppg_data_path=patient_info["ppg_data_path"],
        respiration_data_path=patient_info["respiration_data_path"],
        rPPG_folder_path=patient_info["rppg_folder_path"],
        hr_window_size=HR_WINDOW_SIZE,
        respiration_window_size=RESPIRATION_WINDOW_SIZE,
        auto_run=True
    )
    print("ECG HR values", patient_analysis.ecg_hr_values)
    print("hr results", patient_analysis.hr_results)
    # patient_analysis.plot_spectral_analysis()
    # patient_analysis.plot_gt()
    if patient_analysis.frame_count / patient_analysis.video_fps > HR_WINDOW_SIZE:
        print('duration', patient_analysis.frame_count / patient_analysis.video_fps)
        results[patient] = {
            "ecg_hr_values": patient_analysis.ecg_hr_values,
            "rppg_hr_values": patient_analysis.rppg_hr_values,
            "hr_results": patient_analysis.hr_results,
            "snr": patient_analysis.hr_results,
            "rr_gt_values": patient_analysis.gt_rr_values,
            "rr_rppg_values": patient_analysis.rppg_rr_values,
            "rr_results": patient_analysis.rr_results,
            "correlation_results": patient_analysis.correlation_results
        }
        for method in methods:
            hr_ecg_all[method].extend(patient_analysis.ecg_hr_values)
            hr_rppg_all[method].extend(patient_analysis.rppg_hr_values[method])
            snr_all[method].append(patient_analysis.hr_results[method]['snr'])
            if patient_analysis.gt_rr_values is not None:
                rr_gt_all[method].extend(patient_analysis.gt_rr_values)
            rr_rppg_all[method].extend(patient_analysis.rppg_rr_values[method]['fusion'])
            if method in patient_analysis.correlation_results:
                pearson_coef_all[method].append(patient_analysis.correlation_results[method]['pearson_coef'])

save_results(results)

print('hr ecg all\n', hr_ecg_all)
print('hr rppg all\n', hr_rppg_all)
print('snr all\n', snr_all)
print('rr gt all\n', rr_gt_all)
print('rr rppg all\n', rr_rppg_all)
print('pearson coef all\n', pearson_coef_all)

mae_hr_all = {method: mean_absolute_error(hr_ecg_all[method], hr_rppg_all[method]) for method in methods}
print('MAE all\n', mae_hr_all)
# print(rr_gt_all, 'gt rr all')
# mae_rr_all = {method: mean_absolute_error(rr_gt_all[method], rr_rppg_all[method]) for method in methods}
# print('MAE rr all\n', mae_rr_all)
mean_snr_all = {method: sum(snr_all[method]) / len(snr_all[method]) for method in methods}
print('Mean SNR all\n', mean_snr_all)
greatest_mae, patient_with_greatest_mae, method_with_greatest_mae = max(
    (
        patient_results["snr"][method]["mae"],
        patient,
        method,
    )
    for patient, patient_results in results.items()
    for method in patient_results["snr"]
)
print(
    f"Greatest MAE: {greatest_mae} | "
    f"Patient: {patient_with_greatest_mae} | "
    f"Method: {method_with_greatest_mae}"
)