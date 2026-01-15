import h5py
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import sys

def main():
    # Check if a filename was provided as a command-line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Hide the root window for file selection
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select HDF5 file", filetypes=[("HDF5 files", "*.h5")])
        if not file_path:
            print("No file selected. Exiting.")
            exit()

    # Definitions
    DATA_PACK_HEAD = b"\x02\x0B\x00\x00"  # Defined as bytes
    data_add = 36

    # Open the HDF5 file in read mode
    hdf = h5py.File(file_path, 'r')

    data = hdf['data'][:]
    data_timestamps = hdf['data_timestamps'][:]

    datas = []
    ids = []
    seqs = [] 

    # Data processing
    for raw_data, ts in zip(data, data_timestamps):
        pack_id = bytes(raw_data[0:4])
        if pack_id == DATA_PACK_HEAD: 
            frame_len = int.from_bytes(raw_data[4:6], byteorder='big')
            frame_seq = int.from_bytes(raw_data[24:26], byteorder='big')
            npk = 0
            process_next = True
            local_data_add = data_add  # Use a local variable to avoid overwriting the original
            while process_next:
                if local_data_add + 4 <= len(raw_data):
                    data_head = raw_data[local_data_add:local_data_add+4]
                    data_len = int(raw_data[local_data_add+3]) * 2
                    if data_len > 0:
                        if local_data_add + 4 + data_len <= len(raw_data):
                            data_bytes = raw_data[local_data_add+4:local_data_add+4+data_len]
                        else:
                            data_bytes = raw_data[local_data_add+4:]
                        data_array = np.frombuffer(data_bytes, dtype='>i2')  # Convert to np.array of integers
                        datas.append(data_array)
                        ids.append(int.from_bytes(data_head[0:3], byteorder='big'))
                        seqs.append(frame_seq)
                        npk += 1
                        local_data_add = int(local_data_add) + 4 + data_len
                    else:
                        process_next = False
                else:
                    process_next = False

    seqs = np.array(seqs)
    ids = np.array(ids)

    uniqIDs = np.unique(ids)

    # List of IDs and corresponding names
    idECGs = [65540, 65796, 66052, 66308, 66564, 66820, 67076, 67332, 67588, 67844, 68100, 68356]
    idSpO2 = 458768
    idResp = 327688
    idP1 = 4063240
    idPVC = 3473416
    idART = 2883592
    idCO2 = 4784136

    id_para_nome = {id_: f"ECG{i+1}" for i, id_ in enumerate(idECGs)}  # Add ECGs
    id_para_nome.update({
        idSpO2: "SpO2",
        idResp: "Resp",
        idP1: "P1",
        idPVC: "PVC",
        idART: "ART",
        idCO2: "CO2"
    })

    # Filter id_para_nome to include only IDs present in uniqIDs
    id_para_nome = {id_: name for id_, name in id_para_nome.items() if id_ in uniqIDs}

    # Create reverse mapping from names to IDs
    nome_para_id = {name: id_ for id_, name in id_para_nome.items()}

    # Start the GUI for signal selection
    root = tk.Tk()
    root.title("Select Signal to Plot")

    # Adjust window size
    window_width = 300
    window_height = 200  # Adjusted height to accommodate all signal names

    # Set the window size
    root.geometry(f"{window_width}x{window_height}")

    # Create a frame for the listbox and scrollbar
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Add a scrollbar to the listbox
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    listbox = tk.Listbox(frame, selectmode=tk.SINGLE, yscrollcommand=scrollbar.set)

    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Insert signal names into the listbox
    for name in id_para_nome.values():
        listbox.insert(tk.END, name)

    def plot_signal():
        selected_indices = listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No selection", "Please select a signal to plot.")
            return
        selected_index = selected_indices[0]
        selected_name = listbox.get(selected_index)
        # Map the selected name back to ID
        selected_id = nome_para_id.get(selected_name)
        if selected_id is None:
            messagebox.showerror("Error", "Selected signal ID not found.")
            return
        # Retrieve the data
        indices = np.where(ids == selected_id)[0]
        if len(indices) == 0:
            messagebox.showinfo("No data", "No data available for the selected signal.")
            return
        sig = [datas[i] for i in indices]
        sig = np.concatenate(sig)
        plt.figure()
        plt.plot(sig)
        plt.title(selected_name)
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.show()

    plot_button = tk.Button(root, text="Plot", command=plot_signal)
    plot_button.pack(pady=10)

    def on_closing():
        root.destroy()
        exit()  # Ensure the program terminates when the window is closed

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
