
import pandas as pd

uti_data_path = r"\\10.8.0.1\uti\Data"
dataset_raw_file = r"C:\Users\Sophia\Documents\rPPG_data\ground_truth\dataset_raw.csv"
params_file = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQi0TexCrRMbHHODBHKEWmoA8ipixOkFQqgVdHiznKbn19cBa6VignR47r90AweuomdhyQFCBInDE9y/pub?output=csv"

def get_bed_ip(bed, ip_ids_file):
    """ ipids_file looks like this:
    CNS41720,10.10.10.50
    LEITO 13,10.10.10.141
    LEITO 07,10.10.10.128
    LEITO 06,10.10.10.139
    LEITO 11,10.10.10.119
    LEITO 02,10.10.10.138
    LEITO 04,10.10.10.134
    LEITO 12,10.10.10.122
    LEITO 08,10.10.10.136
    LEITO 10,10.10.10.129
    LEITO 14,10.10.10.131
    """
    with open(ip_ids_file, 'r') as f:
        lines = f.readlines()
        beds = []
        for line in lines:
            beds.append(line.split(',')[0].strip())
            if f"LEITO {bed}" in line:
                return line.split(',')[1].strip()
    raise ValueError(f"Bed {bed} not found in {ip_ids_file}\nBeds found:\n{beds}")

def get_h5_file(date, bed, time, uti_data_path):
    # h5 path will look something like "\\10.8.0.1\uti\Data\20260816\10.10.10.138_20260816_16.h5"

    day = date.split("/")[0]
    month = date.split("/")[1]
    year = date.split("/")[2]
    day_folder = f"{year}{month}{day}"
    ip_ids_file = f"{uti_data_path}/{day_folder}/{day_folder}_{int(time)+1}_onLineDevices.log"

    bed_ip = get_bed_ip(bed, ip_ids_file)
    return f"{uti_data_path}/{day_folder}/{bed_ip}_{day_folder}_{time}.h5"

def fill_data_df(params_file, uti_data_path):
    params_df = pd.read_csv(params_file)
    params_df = params_df.iloc[:, 1:]
    params_df = params_df.T.reset_index()
    params_df.columns = params_df.iloc[0]
    params_df = params_df.iloc[1:].reset_index(drop=True)
    data_df = params_df
    print(data_df)

    """ data_df will look like this:
    0 Id do paciente         Dia Hora Leito Hora monitor Distância horizontal da câmera ao rosto (cm)  ... Oxigenação média durante a gravação Hipoxemia DVA (drogas vasoativas) Cirurgia Data e hora da cirurgia Glasgow
0        1234567  16/08/2026        16    04        16:00                                          170  ...                                 NaN     FALSE                     NaN      NaN                     NaN     NaN
1        1234566  16/08/2026        16    05        16:30                                          170  ...                                 NaN     FALSE                     NaN      NaN                     NaN     NaN
    """

    def _h5_file_for_row(row):
        date = str(row["Dia"]).strip()
        bed = str(row["Leito"]).strip()
        time_value = str(row["Hora"]).strip()
        time = time_value.split(":")[0]
        return get_h5_file(date, bed, time, uti_data_path)

    data_df["h5_file"] = data_df.apply(_h5_file_for_row, axis=1)
    data_df.to_csv(dataset_raw_file, index=False)
    return data_df


if __name__ == "__main__":
    datadf = fill_data_df(params_file, uti_data_path)
    print("datadf", datadf)

