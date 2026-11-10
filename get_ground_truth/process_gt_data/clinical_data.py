import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd

from config import APP_SETTINGS


PATIENT_COLUMNS = (
    "Id do paciente",
    "Dia",
    "Hora",
    "Leito",
    "evolucoes_path",
)
CLINICAL_COLUMNS = {
    "Diabetes": "Diabetes",
    "Hipertensão": "Hipertensão",
    "Condição cardiovascular": "Problema cardiovascular",
    "Outras comorbidades": "Outras comorbidades",
    "MUC": "MUC",
    "Alergia": "Alergia",
    "Ultima prescrição": "Ultima prescrição",
}
OPTIONAL_CLINICAL_COLUMNS = ("MUC", "Alergia", "Ultima prescrição")
DIABETES_OPTIONS = ("Não", "DM1", "DM2")
HYPERTENSION_OPTIONS = ("Sim", "Não")


class ClinicalDataApp:
    def __init__(self, root, dataframe=None, save_callback=None):
        self.root = root
        self.root.title("Dados clínicos dos pacientes")
        self.root.geometry("1280x640")
        self.root.minsize(1000, 460)
        self.editors = {}
        self.evolution_offsets = {}
        self.last_click = None
        self.save_callback = save_callback

        if dataframe is None:
            try:
                self.df = pd.read_csv(APP_SETTINGS.dataset_raw_file)
                for column in OPTIONAL_CLINICAL_COLUMNS:
                    if column not in self.df.columns:
                        self.df[column] = pd.NA
                self._validate_columns()
            except (
                OSError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                UnicodeDecodeError,
                ValueError,
            ) as exc:
                messagebox.showerror(
                    "Erro ao carregar planilha",
                    f"Não foi possível carregar a planilha:\n"
                    f"{APP_SETTINGS.dataset_raw_file}\n\n{exc}",
                    parent=self.root,
                )
                self.root.destroy()
                return
        else:
            self.df = dataframe
            try:
                self._validate_columns()
            except ValueError as exc:
                messagebox.showerror(
                    "Erro ao carregar dados clínicos",
                    str(exc),
                    parent=self.root,
                )
                self.root.destroy()
                return

        for column in CLINICAL_COLUMNS.values():
            self.df[column] = self.df[column].astype("object")

        self._configure_style()
        self._build_ui()
        self._populate_patients()

    def _validate_columns(self):
        required = (*PATIENT_COLUMNS, *CLINICAL_COLUMNS.values())
        missing = [column for column in required if column not in self.df.columns]
        if missing:
            raise ValueError(
                "A planilha não contém as colunas necessárias: "
                + ", ".join(missing)
            )

    @staticmethod
    def _display_value(value):
        if pd.isna(value):
            return ""
        return str(value).strip()

    @staticmethod
    def _is_filled(value):
        return (
            not pd.isna(value)
            and str(value).strip().lower() not in ("", "nan", "none")
        )

    def _build_ui(self):
        self.root.configure(bg="#10131a")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="ClinicalHeader.TFrame", padding=(28, 22))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header, text="Dados clínicos", style="ClinicalHeaderTitle.TLabel"
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Selecione um paciente para abrir a evolução e atualizar as informações.",
            style="ClinicalHeaderSubtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(5, 0))

        content = ttk.Frame(self.root, style="ClinicalApp.TFrame", padding=20)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        ttk.Label(
            content, text="Pacientes", style="ClinicalSectionTitle.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        table_frame = ttk.Frame(content, style="ClinicalPanel.TFrame", padding=10)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = ("id", "dia", "hora", "leito")
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
            style="Clinical.Treeview",
        )
        headings = {
            "id": "ID do paciente",
            "dia": "Data",
            "hora": "Hora",
            "leito": "Leito",
        }
        widths = {"id": 180, "dia": 150, "hora": 160, "leito": 120}
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="center")
        self.tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient=tk.VERTICAL,
            command=self.tree.yview,
            style="Clinical.Vertical.TScrollbar",
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<ButtonRelease-1>", self._on_patient_clicked)

        ttk.Label(
            content,
            text="Ao selecionar um paciente, o arquivo de evolução será aberto.",
            style="ClinicalHint.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(12, 0))

    def _configure_style(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("ClinicalApp.TFrame", background="#10131a")
        style.configure("ClinicalPanel.TFrame", background="#191e28")
        style.configure("ClinicalHeader.TFrame", background="#10131a")
        style.configure(
            "Clinical.TLabel",
            background="#10131a",
            foreground="#e7ebf2",
            font=("Segoe UI", 10),
        )
        style.configure(
            "ClinicalHeaderTitle.TLabel",
            background="#10131a",
            foreground="#eef2f8",
            font=("Segoe UI Semibold", 21),
        )
        style.configure(
            "ClinicalHeaderSubtitle.TLabel",
            background="#10131a",
            foreground="#a3adbd",
            font=("Segoe UI", 10),
        )
        style.configure(
            "ClinicalSectionTitle.TLabel",
            background="#10131a",
            foreground="#eef2f8",
            font=("Segoe UI Semibold", 12),
        )
        style.configure(
            "ClinicalHint.TLabel",
            background="#10131a",
            foreground="#a3adbd",
            font=("Segoe UI", 9),
        )
        style.configure(
            "ClinicalPanel.TLabel",
            background="#191e28",
            foreground="#a3adbd",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Clinical.Treeview",
            background="#191e28",
            fieldbackground="#191e28",
            foreground="#e7ebf2",
            font=("Segoe UI", 10),
            rowheight=32,
            borderwidth=0,
        )
        style.configure(
            "Clinical.Treeview.Heading",
            background="#222a37",
            foreground="#cbd4e2",
            font=("Segoe UI Semibold", 9),
            padding=(10, 10),
            relief="flat",
        )
        style.map(
            "Clinical.Treeview",
            background=[("selected", "#394d82")],
            foreground=[("selected", "#ffffff")],
        )
        style.map(
            "Clinical.Treeview.Heading",
            background=[("active", "#303a4b")],
        )
        style.configure(
            "Clinical.TCombobox",
            padding=8,
            fieldbackground="#191e28",
            background="#222a37",
            foreground="#e7ebf2",
            arrowsize=14,
        )
        style.map(
            "Clinical.TCombobox",
            foreground=[
                ("disabled", "#a3adbd"),
                ("readonly", "#e7ebf2"),
                ("!disabled", "#e7ebf2"),
            ],
            fieldbackground=[
                ("disabled", "#222a37"),
                ("readonly", "#191e28"),
                ("focus", "#191e28"),
                ("!disabled", "#191e28"),
            ],
            background=[
                ("disabled", "#222a37"),
                ("readonly", "#222a37"),
                ("active", "#303a4b"),
            ],
        )
        style.configure(
            "Clinical.TEntry",
            padding=8,
            fieldbackground="#191e28",
            foreground="#e7ebf2",
        )
        style.map(
            "Clinical.TEntry",
            foreground=[
                ("disabled", "#a3adbd"),
                ("!disabled", "#e7ebf2"),
            ],
            fieldbackground=[
                ("disabled", "#222a37"),
                ("focus", "#191e28"),
                ("!disabled", "#191e28"),
            ],
        )
        style.configure(
            "Clinical.Vertical.TScrollbar",
            background="#303a4b",
            troughcolor="#191e28",
            bordercolor="#191e28",
            arrowcolor="#cbd4e2",
            relief="flat",
        )
        style.map(
            "Clinical.Vertical.TScrollbar",
            background=[("active", "#394d82")],
        )
        self.root.option_add("*TCombobox*Listbox.background", "#191e28")
        self.root.option_add("*TCombobox*Listbox.foreground", "#e7ebf2")
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#394d82")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        style.configure(
            "Clinical.Success.TButton",
            background="#21875f",
            foreground="#ffffff",
            padding=(12, 10),
            borderwidth=0,
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "Clinical.Success.TButton",
            background=[("active", "#2ba675")],
        )
        style.configure(
            "Clinical.Secondary.TButton",
            background="#222a37",
            foreground="#e7ebf2",
            padding=(12, 10),
            borderwidth=0,
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "Clinical.Secondary.TButton",
            background=[("active", "#303a4b")],
        )

    def _populate_patients(self):
        for index, row in self.df.iterrows():
            self.tree.insert(
                "", tk.END, iid=str(index), values=self._patient_display_values(row)
            )

    @staticmethod
    def _patient_display_values(row):
        return (
            ClinicalDataApp._display_value(row["Id do paciente"]),
            ClinicalDataApp._display_value(row["Dia"]),
            ClinicalDataApp._display_value(row["Hora"]),
            ClinicalDataApp._display_value(row["Leito"]),
        )

    def _on_patient_clicked(self, event):
        index = self.tree.identify_row(event.y)
        if not index:
            return

        if (
            self.last_click is not None
            and self.last_click[0] == index
            and 0 <= event.time - self.last_click[1] < 500
        ):
            return
        self.last_click = (index, event.time)
        self.tree.selection_set(index)
        row = self.df.loc[int(index)]
        try:
            evolution_path = self._resolve_evolution_path(row)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Erro ao abrir evolução", str(exc), parent=self.root)
        else:
            try:
                self._open_file(evolution_path)
            except OSError as exc:
                messagebox.showerror(
                    "Erro ao abrir evolução",
                    f"Não foi possível abrir o arquivo:\n{evolution_path}\n\n{exc}",
                    parent=self.root,
                )

        self._open_editor(int(index))

    @staticmethod
    def _resolve_evolution_path(row):
        registered_path = row["evolucoes_path"]
        if not ClinicalDataApp._is_filled(registered_path):
            raise ValueError("O paciente selecionado não possui evolucoes_path.")

        path = Path(str(registered_path).strip())
        if path.is_absolute():
            found_file = path
        else:
            evolution_root = ClinicalDataApp._evolution_directory(row, 0)
            found_file = evolution_root / path

        if not found_file.is_file():
            raise FileNotFoundError(
                f"Arquivo de evolução registrado não encontrado:\n{found_file}"
            )
        return found_file

    @staticmethod
    def _evolution_directory(row, day_offset):
        day_value = ClinicalDataApp._display_value(row["Dia"])
        date_parts = day_value.split("/")
        if len(date_parts) != 3:
            raise ValueError(
                "A data do paciente não está no formato esperado (DD/MM/AAAA)."
            )
        day, month, year = (int(part.strip()) for part in date_parts)
        day += day_offset
        folder_name = f"{year:04d}{month:02d}{day:02d}"
        return Path(APP_SETTINGS.evolucoes_path) / folder_name

    @staticmethod
    def _resolve_next_evolution_path(row, day_offset):
        registered_path = row["evolucoes_path"]
        if not ClinicalDataApp._is_filled(registered_path):
            raise ValueError("O paciente selecionado não possui evolucoes_path.")

        filename = Path(str(registered_path).strip()).name
        found_file = (
            ClinicalDataApp._evolution_directory(row, day_offset) / filename
        )
        if not found_file.is_file():
            raise FileNotFoundError(
                f"Arquivo de evolução não encontrado:\n{found_file}"
            )
        return found_file

    @staticmethod
    def _open_file(path):
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _open_editor(self, index):
        existing_window = self.editors.get(index)
        if existing_window is not None and existing_window.winfo_exists():
            existing_window.lift()
            existing_window.focus_set()
            return

        row = self.df.loc[index]
        window = tk.Toplevel(self.root)
        self.editors[index] = window
        window.title(f"Dados clínicos - {self._display_value(row['Id do paciente'])}")
        window.geometry("680x640")
        window.resizable(False, False)
        window.transient(self.root)
        window.configure(bg="#10131a")

        frame = ttk.Frame(window, style="ClinicalApp.TFrame", padding=22)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Dados do paciente", style="ClinicalSectionTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 14)
        )
        ttk.Label(frame, text="Diabetes", style="Clinical.TLabel").grid(row=1, column=0, sticky="w", pady=8)
        diabetes = tk.StringVar(value=self._display_value(row["Diabetes"]))
        ttk.Combobox(
            frame,
            textvariable=diabetes,
            values=DIABETES_OPTIONS,
            state="readonly",
            width=35,
            style="Clinical.TCombobox",
        ).grid(row=1, column=1, sticky="ew", pady=8)

        ttk.Label(frame, text="Hipertensão", style="Clinical.TLabel").grid(
            row=2, column=0, sticky="w", pady=8
        )
        hypertension = tk.StringVar(value=self._display_value(row["Hipertensão"]))
        ttk.Combobox(
            frame,
            textvariable=hypertension,
            values=HYPERTENSION_OPTIONS,
            state="readonly",
            width=35,
            style="Clinical.TCombobox",
        ).grid(row=2, column=1, sticky="ew", pady=8)

        ttk.Label(
            frame, text="Condição cardiovascular", style="Clinical.TLabel"
        ).grid(
            row=3, column=0, sticky="w", pady=8
        )
        cardiovascular = tk.StringVar(
            value=self._display_value(row["Problema cardiovascular"])
        )
        self._add_limited_entry(frame, 3, cardiovascular, 100)

        ttk.Label(frame, text="Outras comorbidades", style="Clinical.TLabel").grid(
            row=4, column=0, sticky="w", pady=8
        )
        comorbidities = tk.StringVar(
            value=self._display_value(row["Outras comorbidades"])
        )
        self._add_limited_entry(frame, 4, comorbidities, 250)

        ttk.Label(frame, text="MUC", style="Clinical.TLabel").grid(row=5, column=0, sticky="w", pady=8)
        muc = tk.StringVar(value=self._display_value(row["MUC"]))
        self._add_limited_entry(frame, 5, muc, 250)

        ttk.Label(frame, text="Alergia", style="Clinical.TLabel").grid(row=6, column=0, sticky="w", pady=8)
        allergy = tk.StringVar(value=self._display_value(row["Alergia"]))
        self._add_limited_entry(frame, 6, allergy, 250)

        ttk.Label(frame, text="Ultima prescrição", style="Clinical.TLabel").grid(
            row=7, column=0, sticky="w", pady=8
        )
        prescription = tk.Text(
            frame,
            height=4,
            width=38,
            wrap="word",
            font=("Segoe UI", 10),
            background="#191e28",
            foreground="#e7ebf2",
            insertbackground="#eef2f8",
            selectbackground="#394d82",
            relief=tk.FLAT,
            borderwidth=0,
            padx=8,
            pady=6,
            undo=True,
        )
        prescription.insert("1.0", self._display_value(row["Ultima prescrição"]))
        prescription.grid(row=7, column=1, sticky="ew", pady=6)
        prescription_scrollbar = ttk.Scrollbar(
            frame,
            orient=tk.VERTICAL,
            command=prescription.yview,
            style="Clinical.Vertical.TScrollbar",
        )
        prescription_scrollbar.grid(row=7, column=2, sticky="ns", pady=6)
        prescription.configure(yscrollcommand=prescription_scrollbar.set)
        self._limit_text_length(prescription, 500)

        buttons = ttk.Frame(frame, style="ClinicalApp.TFrame")
        buttons.grid(row=8, column=0, columnspan=2, sticky="e", pady=(22, 0))
        ttk.Button(
            buttons,
            text="Salvar",
            style="Clinical.Success.TButton",
            command=lambda: self._save_patient(
                index,
                window,
                diabetes.get(),
                hypertension.get(),
                cardiovascular.get(),
                comorbidities.get(),
                muc.get(),
                allergy.get(),
                prescription.get("1.0", "end-1c"),
            ),
        ).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(
            buttons,
            text="Cancelar",
            style="Clinical.Secondary.TButton",
            command=lambda: self._close_editor(index, window),
        ).pack(side=tk.RIGHT)
        evolution_status = tk.StringVar(value="")
        ttk.Button(
            buttons,
            text="Abrir próxima evolução",
            style="Clinical.Secondary.TButton",
            command=lambda: self._open_next_evolution(
                index, window, evolution_status
            ),
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(
            frame, textvariable=evolution_status, style="ClinicalHint.TLabel"
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(10, 0))
        window.protocol(
            "WM_DELETE_WINDOW", lambda: self._close_editor(index, window)
        )

    @staticmethod
    def _add_limited_entry(parent, row, variable, max_length):
        validate_command = parent.register(
            lambda value: len(value) <= max_length
        )
        ttk.Entry(
            parent,
            textvariable=variable,
            width=38,
            style="Clinical.TEntry",
            validate="key",
            validatecommand=(validate_command, "%P"),
        ).grid(row=row, column=1, sticky="ew", pady=6)

    @staticmethod
    def _limit_text_length(widget, max_length):
        def enforce_limit(event):
            if event.widget is not widget:
                return
            if not widget.edit_modified():
                return

            content = widget.get("1.0", "end-1c")
            if len(content) > max_length:
                widget.delete(f"1.0 + {max_length} chars", "end-1c")
            widget.edit_modified(False)

        widget.bind("<<Modified>>", enforce_limit)
        widget.edit_modified(False)

    def _save_patient(
        self,
        index,
        window,
        diabetes,
        hypertension,
        cardiovascular,
        comorbidities,
        muc,
        allergy,
        prescription,
    ):
        if diabetes not in DIABETES_OPTIONS:
            messagebox.showerror(
                "Campo obrigatório", "Selecione uma opção para Diabetes.", parent=window
            )
            return
        if hypertension not in HYPERTENSION_OPTIONS:
            messagebox.showerror(
                "Campo obrigatório",
                "Selecione uma opção para Hipertensão.",
                parent=window,
            )
            return
        if (
            len(cardiovascular) > 100
            or len(comorbidities) > 250
            or len(muc) > 250
            or len(allergy) > 250
            or len(prescription) > 500
        ):
            messagebox.showerror(
                "Texto muito longo",
                "Condição cardiovascular aceita até 100 caracteres e "
                "Outras comorbidades, MUC e Alergia até 250 caracteres; "
                "Ultima prescrição até 500 caracteres.",
                parent=window,
            )
            return

        values = {
            "Diabetes": diabetes,
            "Hipertensão": hypertension,
            "Problema cardiovascular": cardiovascular,
            "Outras comorbidades": comorbidities,
            "MUC": muc,
            "Alergia": allergy,
            "Ultima prescrição": prescription,
        }
        previous_values = {column: self.df.at[index, column] for column in values}

        try:
            if self.save_callback is not None:
                self.save_callback(index, values)
            else:
                for column, value in values.items():
                    self.df.at[index, column] = value
                self.df.to_csv(APP_SETTINGS.dataset_raw_file, index=False)
        except (OSError, PermissionError, ValueError) as exc:
            if self.save_callback is None:
                for column, value in previous_values.items():
                    self.df.at[index, column] = value
            messagebox.showerror(
                "Erro ao salvar",
                f"Não foi possível atualizar a planilha:\n{exc}",
                parent=window,
            )
            return

        self.tree.item(
            str(index),
            values=self._patient_display_values(self.df.loc[index]),
        )
        messagebox.showinfo(
            "Dados salvos", "Os dados clínicos foram atualizados na planilha.", parent=window
        )
        self._close_editor(index, window)

    def _open_next_evolution(self, index, window, status):
        day_offset = self.evolution_offsets.get(index, 0) + 1
        self.evolution_offsets[index] = day_offset
        row = self.df.loc[index]
        try:
            evolution_path = self._resolve_next_evolution_path(row, day_offset)
            self._open_file(evolution_path)
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                "Erro ao abrir evolução", str(exc), parent=window
            )
            return

        status.set(f"Evolução aberta: {evolution_path.parent.name}")

    def _close_editor(self, index, window):
        self.editors.pop(index, None)
        self.evolution_offsets.pop(index, None)
        window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ClinicalDataApp(root)
    if root.winfo_exists():
        root.mainloop()
