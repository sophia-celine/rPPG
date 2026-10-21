from pathlib import Path

from rPPGAnalysis import rPPGAnalysis

# VIDEO_PATH = r"/home/soph/rppg/rPPG-Toolbox/data/test/RawData/subject1/vid.avi"
# ECG_DATA_PATH = '/home/soph/ssd/dataset_raw/2356759_2/ecg_20-08-2026_3_16-34-55_16-37-00.csv'
# PPG_DATA_PATH = '/home/soph/ssd/dataset_raw/2356759_2/ppg_20-08-2026_3_16-34-55_16-37-00.txt'
# RESPIRATION_DATA_PATH = '/home/soph/ssd/dataset_raw/2356759_2/respiration_20-08-2026_3_16-34-55_16-37-00.txt'
# RPPG_FOLDER_PATH = '/home/soph/rppg/rPPG-Toolbox/BVPresults'
HR_WINDOW_SIZE = 60
RESPIRATION_WINDOW_SIZE = 60

def find_patient_info(root_path: Path) -> tuple[str, str, str, str, str]:
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
        str(find_unique(
            "PPG TXT file",
            [
                path for path in files
                if path.name.casefold().startswith("ppg")
                and path.suffix.casefold() == ".txt"
            ],
        )),
        str(find_unique(
            "respiration TXT file",
            [
                path for path in files
                if path.name.casefold().startswith("resp")
                and path.suffix.casefold() == ".txt"
            ],
        )),
        str(find_unique(
            "BVP directory",
            [path for path in directories if path.name.casefold().startswith("bvp")],
        )),
    )

# EXEMPLO MAIS DIFÍCIL
# VIDEO_PATH, ECG_DATA_PATH, PPG_DATA_PATH, RESPIRATION_DATA_PATH, RPPG_FOLDER_PATH = find_patient_info(r"A:\dataset_raw\test\10227_1")
# BRADICARDIA 
# VIDEO_PATH, ECG_DATA_PATH, PPG_DATA_PATH, RESPIRATION_DATA_PATH, RPPG_FOLDER_PATH = find_patient_info(r"A:\dataset_raw\batch4\2356759_2")
# VIDEO_PATH, ECG_DATA_PATH, PPG_DATA_PATH, RESPIRATION_DATA_PATH, RPPG_FOLDER_PATH = find_patient_info("/home/soph/ssd/dataset_raw/batch4/2356759_2")

VIDEO_PATH, ECG_DATA_PATH, PPG_DATA_PATH, RESPIRATION_DATA_PATH, RPPG_FOLDER_PATH = find_patient_info(r"A:\dataset_raw\batch1\10227_1")

if __name__ == '__main__':

    patient2_analysis = rPPGAnalysis(video_path=VIDEO_PATH, 
                 ecg_data_path=ECG_DATA_PATH, 
                 ppg_data_path=PPG_DATA_PATH, 
                 respiration_data_path=RESPIRATION_DATA_PATH, 
                 rPPG_folder_path=RPPG_FOLDER_PATH, 
                 hr_window_size=HR_WINDOW_SIZE, 
                 respiration_window_size=RESPIRATION_WINDOW_SIZE
                 )
    patient2_analysis.run()
    # patient2_analysis.plot_gt()
    # print('rppg_signals\n', patient2_analysis.rppg_signals)
    print('ecg_hr_values\n', patient2_analysis.ecg_hr_values)
    # # print('fps\n', patient2_analysis.video_fps)
    print('rppg_hr_values\n', patient2_analysis.rppg_hr_values)
    print('hr_results\n', patient2_analysis.hr_results)
    # print('Correlation_results\n', patient2_analysis.correlation_results)
    # print('Respiratory rate estimation\n', patient2_analysis.rr_results)
    patient2_analysis.plot_spectral_analysis()
    patient2_analysis.plot_rppg_ppg_correlation()
    # patient2_analysis.plot_bvp(same_scale=False)
    # patient2_analysis.plot_all(method='POS')