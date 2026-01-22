import h5py
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import os

def main():
    # Check if a directory was provided as a command-line argument
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        # Hide the root window for directory selection
        root = tk.Tk()
        root.withdraw()
        directory_path = filedialog.askdirectory(title="Select Directory Containing HDF5 Files")
        if not directory_path:
            print("No directory selected. Exiting.")
            exit()

    # Get the list of .h5 files in the directory
    h5_files = [f for f in os.listdir(directory_path) if f.endswith('.h5')]
    if not h5_files:
        print("No .h5 files found in the selected directory.")
        exit()

    # Start the GUI for file and signal selection
    app = FileSignalSelector(directory_path, h5_files)
    app.run()

class FileSignalSelector:
    def __init__(self, directory_path, h5_files):
        self.directory_path = directory_path
        self.h5_files = h5_files
        self.current_file_path = None
        self.data_loaded = False
        self.datas = []
        self.ids = []
        self.seqs = []
        self.id_para_nome = {}
        self.nome_para_id = {}
        self.root = tk.Tk()
        self.root.title("Select HDF5 File")
        self.create_file_selection_ui()

    def create_file_selection_ui(self):
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create a frame for the listbox and scrollbar
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)

        # Add a scrollbar to the listbox
        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        self.file_listbox = tk.Listbox(frame, selectmode=tk.SINGLE, yscrollcommand=scrollbar.set, height=40, width=80)
        scrollbar.config(command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Insert file names into the listbox
        for file_name in self.h5_files:
            self.file_listbox.insert(tk.END, file_name)

        # Create buttons
        open_button = tk.Button(self.root, text="Open File", command=self.open_selected_file)
        open_button.pack(pady=10)

        exit_button = tk.Button(self.root, text="Exit", command=self.exit_program)
        exit_button.pack(pady=5)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.exit_program)

    def open_selected_file(self):
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No selection", "Please select a file to open.")
            return
        selected_index = selected_indices[0]
        selected_file_name = self.file_listbox.get(selected_index)
        self.current_file_path = os.path.join(self.directory_path, selected_file_name)
        self.load_file_data()
        if self.data_loaded:
            self.create_signal_selection_ui()

    def load_file_data(self):
        try:
            hdf = h5py.File(self.current_file_path, 'r')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")
            return

        # Definitions
        DATA_PACK_HEAD = b"\x02\x0B\x00\x00"  # Defined as bytes
        data_add = 36

        data = hdf['data'][:]
        data_timestamps = hdf['data_timestamps'][:]

        self.datas = []
        self.ids = []
        self.seqs = []

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
                        data_len = raw_data[local_data_add+3] * 2
                        if data_len > 0:
                            if local_data_add + 4 + data_len <= len(raw_data):
                                data_bytes = raw_data[local_data_add+4:local_data_add+4+data_len]
                            else:
                                data_bytes = raw_data[local_data_add+4:]
                            data_array = np.frombuffer(data_bytes, dtype='>i2')  # Convert to np.array of integers
                            self.datas.append(data_array)
                            self.ids.append(int.from_bytes(data_head[0:3], byteorder='big'))
                            self.seqs.append(frame_seq)
                            npk += 1
                            local_data_add = local_data_add + 4 + data_len
                        else:
                            process_next = False
                    else:
                        process_next = False

        self.seqs = np.array(self.seqs)
        self.ids = np.array(self.ids)

        uniqIDs = np.unique(self.ids)

        # List of IDs and corresponding names
        idECGs = [65540, 65796, 66052, 66308, 66564, 66820, 67076, 67332, 67588, 67844, 68100, 68356]
        idSpO2 = 458768
        idResp = 327688
        idP1 = 4063240
        idPVC = 3473416
        idART = 2883592
        idCO2 = 4784136

        self.id_para_nome = {id_: f"ECG{i+1}" for i, id_ in enumerate(idECGs)}  # Add ECGs
        self.id_para_nome.update({
            idSpO2: "SpO2",
            idResp: "Resp",
            idP1: "P1",
            idPVC: "PVC",
            idART: "ART",
            idCO2: "CO2"
        })

        # Filter id_para_nome to include only IDs present in uniqIDs
        self.id_para_nome = {id_: name for id_, name in self.id_para_nome.items() if id_ in uniqIDs}

        if not self.id_para_nome:
            messagebox.showinfo("No Signals", "No recognizable signals found in this file.")
            return

        # Create reverse mapping from names to IDs
        self.nome_para_id = {name: id_ for id_, name in self.id_para_nome.items()}

        self.data_loaded = True

    def create_signal_selection_ui(self):
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Update window title
        self.root.title(f"Select Signal to Plot - {os.path.basename(self.current_file_path)}")

        # Create a frame for the listbox and scrollbar
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)

        # Add a scrollbar to the listbox
        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        self.signal_listbox = tk.Listbox(frame, selectmode=tk.SINGLE, yscrollcommand=scrollbar.set,height=40, width=80)
        scrollbar.config(command=self.signal_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.signal_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Insert signal names into the listbox
        for name in self.id_para_nome.values():
            self.signal_listbox.insert(tk.END, name)

        # Create buttons
        plot_button = tk.Button(self.root, text="Plot Signal", command=self.plot_signal)
        plot_button.pack(pady=5)

        back_button = tk.Button(self.root, text="Back to File Selection", command=self.create_file_selection_ui)
        back_button.pack(pady=5)

        exit_button = tk.Button(self.root, text="Exit", command=self.exit_program)
        exit_button.pack(pady=5)

    def plot_signal(self):
        selected_indices = self.signal_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No selection", "Please select a signal to plot.")
            return
        selected_index = selected_indices[0]
        selected_name = self.signal_listbox.get(selected_index)
        # Map the selected name back to ID
        selected_id = self.nome_para_id.get(selected_name)
        if selected_id is None:
            messagebox.showerror("Error", "Selected signal ID not found.")
            return
        # Retrieve the data
        indices = np.where(self.ids == selected_id)[0]
        if len(indices) == 0:
            messagebox.showinfo("No data", "No data available for the selected signal.")
            return
        sig = [self.datas[i] for i in indices]
        sig = np.concatenate(sig)
        plt.figure()
        plt.plot(sig)
        plt.title(f"{selected_name} - {os.path.basename(self.current_file_path)}")
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.show()

    def exit_program(self):
        self.root.destroy()
        exit()

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    main()
