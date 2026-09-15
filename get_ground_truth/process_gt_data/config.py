from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import platform

SIGNAL_TIME_OFFSET = timedelta(hours=-5)


def signal_timestamp_to_datetime(timestamp):
    return datetime.fromtimestamp(timestamp) + SIGNAL_TIME_OFFSET


@dataclass(frozen=True)
class AppSettings:
    remote_params_url: str = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQi0TexCrRMbHHODBHKEWmoA8ipixOkFQqgVdHiznKbn19cBa6VignR47r90AweuomdhyQFCBInDE9y/pub?output=csv"

    if platform.system() == "Windows":
        uti_data_path = Path(r"\\10.8.0.1\uti\Data")
        dataset_raw_file = Path.home() / "Documents" / "rPPG_data" / "ground_truth" / "dataset_raw.csv"
        patient_data_root = Path(r"A:\dataset_raw")
        evolucoes_path = Path(r"\\10.8.0.1\UTI_Secure\Evolucoes")
    else:
        uti_data_path = "/mnt/10.8.0.1/uti/Data"
        dataset_raw_file = Path.home() / "rppg" / "rPPG_data" / "ground_truth" / "dataset_raw.csv"
        patient_data_root = Path.home() / "ssd" / "dataset_raw"
        evolucoes_path = "/mnt/10.8.0.1/uti_secure/Evolucoes"

    default_video_filename: str = "video_cropped.avi"
    signal_filename_templates: dict[str, str] = field(default_factory=lambda: {
        "ecg": "ecg_{date}_{bed}_{start}_{end}.csv",
        "ppg": "ppg_{date}_{bed}_{start}_{end}.txt",
        "respiration": "respiration_{date}_{bed}_{start}_{end}.txt",
    })
    reference_filename_template: str = "{signal}_{patient_id}.txt"
    video_filename_template: str = "{bed}-{date}-{hour}.avi"


APP_SETTINGS = AppSettings()


@dataclass
class Config:
    file_path: str = ""
    date: str = ""
    start_time: str = "00:00:00"
    end_time: str = "00:00:00"
    bed: str = ""
    output_dir: str = ""
    save_ecg: bool = True
    save_spo2_wave: bool = True
    save_rr: bool = True
    show_plots: bool = True
    data_pack_head: bytes = b"\x02\x0B\x00\x00"
    data_add: int = 36
    ecg_id: int = 65796
    spo2_id: int = 458768
    resp_id: int = 327688
    interactive_select_time: bool = True
    duration_seconds: float = 120.0
    selected_start_ts: float | None = None
    selected_end_ts: float | None = None

    def __post_init__(self):
        self.hora_inicio = self.start_time.replace(':', '-')
        self.hora_fim = self.end_time.replace(':', '-')
        self.output_path = Path(self.output_dir) if self.output_dir else Path(APP_SETTINGS.patient_data_root)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.ecg_dir = self.output_path
        self.spo2_dir = self.output_path
        self.rr_dir = self.output_path
        for folder in (self.ecg_dir, self.spo2_dir, self.rr_dir):
            folder.mkdir(parents=True, exist_ok=True)
