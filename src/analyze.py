from pathlib import Path

from rPPGAnalysis import rPPGAnalysis
from analyze_dataset import find_patient_info

# VIDEO_PATH = r"/home/soph/rppg/rPPG-Toolbox/data/test/RawData/subject1/vid.avi"
# ECG_DATA_PATH = '/home/soph/ssd/dataset_raw/2356759_2/ecg_20-08-2026_3_16-34-55_16-37-00.csv'
# PPG_DATA_PATH = '/home/soph/ssd/dataset_raw/2356759_2/ppg_20-08-2026_3_16-34-55_16-37-00.txt'
# RESPIRATION_DATA_PATH = '/home/soph/ssd/dataset_raw/2356759_2/respiration_20-08-2026_3_16-34-55_16-37-00.txt'
# RPPG_FOLDER_PATH = '/home/soph/rppg/rPPG-Toolbox/BVPresults'
HR_WINDOW_SIZE = 60
RESPIRATION_WINDOW_SIZE = 60

# EXEMPLO MAIS DIFÍCIL
# VIDEO_PATH, ECG_DATA_PATH, PPG_DATA_PATH, RESPIRATION_DATA_PATH, RPPG_FOLDER_PATH = find_patient_info(r"A:\dataset_raw\test\10227_1")
# BRADICARDIA 
# VIDEO_PATH, ECG_DATA_PATH, PPG_DATA_PATH, RESPIRATION_DATA_PATH, RPPG_FOLDER_PATH = find_patient_info(r"A:\dataset_raw\batch4\2356759_2")
# VIDEO_PATH, ECG_DATA_PATH, PPG_DATA_PATH, RESPIRATION_DATA_PATH, RPPG_FOLDER_PATH = find_patient_info("/home/soph/ssd/dataset_raw/batch4/2356759_2")

# VIDEO_PATH, ECG_DATA_PATH, PPG_DATA_PATH, RESPIRATION_DATA_PATH, RPPG_FOLDER_PATH = find_patient_info('/home/soph/ssd/dataset_raw/batch2/2358200_0')
VIDEO_PATH, ECG_DATA_PATH, PPG_DATA_PATH, RESPIRATION_DATA_PATH, RPPG_FOLDER_PATH = find_patient_info('/home/soph/ssd/dataset_raw/batch6/2357856_0')

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
    patient2_analysis.plot_rppg_ppg_correlation(methods=['POS'])
    # patient2_analysis.plot_rppg_ppg_correlation()
    patient2_analysis.plot_bvp(same_scale=False, methods=['POS'])
    patient2_analysis.plot_all(method='POS')