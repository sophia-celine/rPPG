from rPPGAnalysis import rPPGAnalysis

VIDEO_PATH = '../../rPPG-Toolbox/data/ICU/RawData/subject1/vid.avi'
ECG_DATA_PATH = '../../rPPG_data/pilot/ECG/ecg_signal_L9_16-05-26_16-07-25.csv'
PPG_DATA_PATH = '../../rPPG_data/pilot/spo2/original_spo2_L9_16-05-26_16-07-25.txt'
RESPIRATION_DATA_PATH = '../../rPPG_data/pilot/thoracic_impedance/L9_16-05-26_16-07-25.txt'
RPPG_FOLDER_PATH = '../../rPPG_data/patient2'
HR_WINDOW_SIZE = 15
RESPIRATION_WINDOW_SIZE = 30

if __name__ == '__main__':

    patient2_analysis = rPPGAnalysis(video_path=VIDEO_PATH, 
                 ecg_data_path=ECG_DATA_PATH, 
                 ppg_data_path=PPG_DATA_PATH, 
                 respiration_data_path=RESPIRATION_DATA_PATH, 
                 rPPG_folder_path=RPPG_FOLDER_PATH, 
                 hr_window_size=HR_WINDOW_SIZE, 
                 respiration_window_size=RESPIRATION_WINDOW_SIZE
                 )
    print('rppg_signals\n', patient2_analysis.rppg_signals)
    # print('ecg_hr_values\n', patient2_analysis.ecg_hr_values)
    # print('fps\n', patient2_analysis.video_fps)
    # print('rppg_hr_values\n', patient2_analysis.rppg_hr_values['PhysFormer (PURE)'])
    print('hr_results\n', patient2_analysis.hr_results)
