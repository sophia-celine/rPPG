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
    def __init__(self, root):
        self.root = root
        self.root.title("Dados clínicos dos pacientes")
        self.root.geometry("1280x640")
        self.root.minsize(1000, 460)
        self.editors = {}
        self.evolution_offsets = {}
        self.last_click = None

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
        self.root.configure(bg="#f3f6fb")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(28, 22))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header, text="Dados clínicos", style="HeaderTitle.TLabel"
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Selecione um paciente para abrir a evolução e atualizar as informações.",
            style="HeaderSubtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(5, 0))

        content = ttk.Frame(self.root, style="App.TFrame", padding=20)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        ttk.Label(
            content, text="Pacientes", style="SectionTitle.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        table_frame = ttk.Frame(content, style="Card.TFrame", padding=10)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = ("id", "dia", "hora", "leito")
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
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
            table_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<ButtonRelease-1>", self._on_patient_clicked)

        ttk.Label(
            content,
            text="Ao selecionar um paciente, o arquivo de evolução será aberto.",
            style="Hint.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(12, 0))

    def _configure_style(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("App.TFrame", background="#f3f6fb")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Header.TFrame", background="#19324f")
        style.configure(
            "TLabel",
            background="#f3f6fb",
            foreground="#344b66",
            font=("Segoe UI", 10),
        )
        style.configure(
            "HeaderTitle.TLabel",
            background="#19324f",
            foreground="#ffffff",
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "HeaderSubtitle.TLabel",
            background="#19324f",
            foreground="#d5e0ed",
            font=("Segoe UI", 10),
        )
        style.configure(
            "SectionTitle.TLabel",
            background="#f3f6fb",
            foreground="#19324f",
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "Hint.TLabel",
            background="#f3f6fb",
            foreground="#607086",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview",
            background="#ffffff",
            fieldbackground="#ffffff",
            foreground="#25364a",
            font=("Segoe UI", 10),
            rowheight=36,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background="#eaf0f7",
            foreground="#344b66",
            font=("Segoe UI", 9, "bold"),
            padding=(10, 9),
            relief="flat",
        )
        style.map(
            "Treeview",
            background=[("selected", "#dceafe")],
            foreground=[("selected", "#173b67")],
        )
        style.map(
            "Treeview.Heading",
            background=[("active", "#dce6f1")],
        )
        style.configure(
            "TCombobox",
            padding=8,
            fieldbackground="#ffffff",
            background="#ffffff",
            foreground="#25364a",
            arrowsize=14,
        )
        style.configure(
            "TEntry",
            padding=8,
            fieldbackground="#ffffff",
            foreground="#25364a",
        )
        style.configure(
            "Primary.TButton",
            background="#2874c6",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(16, 9),
            borderwidth=0,
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#1f62aa"), ("pressed", "#194f89")],
            foreground=[("disabled", "#d5e0ed")],
        )
        style.configure(
            "Secondary.TButton",
            background="#eaf0f7",
            foreground="#344b66",
            font=("Segoe UI", 10),
            padding=(14, 9),
            borderwidth=0,
        )
        style.map("Secondary.TButton", background=[("active", "#dce6f1")])

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
        window.geometry("680x600")
        window.resizable(False, False)
        window.transient(self.root)
        window.configure(bg="#f3f6fb")

        frame = ttk.Frame(window, style="App.TFrame", padding=22)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Dados do paciente", style="SectionTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 14)
        )
        ttk.Label(frame, text="Diabetes").grid(row=1, column=0, sticky="w", pady=8)
        diabetes = tk.StringVar(value=self._display_value(row["Diabetes"]))
        ttk.Combobox(
            frame,
            textvariable=diabetes,
            values=DIABETES_OPTIONS,
            state="readonly",
            width=35,
        ).grid(row=1, column=1, sticky="ew", pady=8)

        ttk.Label(frame, text="Hipertensão").grid(
            row=2, column=0, sticky="w", pady=8
        )
        hypertension = tk.StringVar(value=self._display_value(row["Hipertensão"]))
        ttk.Combobox(
            frame,
            textvariable=hypertension,
            values=HYPERTENSION_OPTIONS,
            state="readonly",
            width=35,
        ).grid(row=2, column=1, sticky="ew", pady=8)

        ttk.Label(frame, text="Condição cardiovascular").grid(
            row=3, column=0, sticky="w", pady=8
        )
        cardiovascular = tk.StringVar(
            value=self._display_value(row["Problema cardiovascular"])
        )
        self._add_limited_entry(frame, 3, cardiovascular, 100)

        ttk.Label(frame, text="Outras comorbidades").grid(
            row=4, column=0, sticky="w", pady=8
        )
        comorbidities = tk.StringVar(
            value=self._display_value(row["Outras comorbidades"])
        )
        self._add_limited_entry(frame, 4, comorbidities, 250)

        ttk.Label(frame, text="MUC").grid(row=5, column=0, sticky="w", pady=8)
        muc = tk.StringVar(value=self._display_value(row["MUC"]))
        self._add_limited_entry(frame, 5, muc, 250)

        ttk.Label(frame, text="Alergia").grid(row=6, column=0, sticky="w", pady=8)
        allergy = tk.StringVar(value=self._display_value(row["Alergia"]))
        self._add_limited_entry(frame, 6, allergy, 250)

        ttk.Label(frame, text="Ultima prescrição").grid(
            row=7, column=0, sticky="w", pady=8
        )
        prescription = tk.Text(
            frame,
            height=4,
            width=38,
            wrap="word",
            font=("Segoe UI", 10),
            padx=8,
            pady=6,
            undo=True,
        )
        prescription.insert("1.0", self._display_value(row["Ultima prescrição"]))
        prescription.grid(row=7, column=1, sticky="ew", pady=6)
        prescription_scrollbar = ttk.Scrollbar(
            frame, orient=tk.VERTICAL, command=prescription.yview
        )
        prescription_scrollbar.grid(row=7, column=2, sticky="ns", pady=6)
        prescription.configure(yscrollcommand=prescription_scrollbar.set)
        self._limit_text_length(prescription, 500)

        buttons = ttk.Frame(frame)
        buttons.grid(row=8, column=0, columnspan=2, sticky="e", pady=(22, 0))
        ttk.Button(
            buttons,
            text="Salvar",
            style="Primary.TButton",
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
            style="Secondary.TButton",
            command=lambda: self._close_editor(index, window),
        ).pack(side=tk.RIGHT)
        evolution_status = tk.StringVar(value="")
        ttk.Button(
            buttons,
            text="Abrir próxima evolução",
            style="Secondary.TButton",
            command=lambda: self._open_next_evolution(
                index, window, evolution_status
            ),
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(
            frame, textvariable=evolution_status, style="Hint.TLabel"
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
        for column, value in values.items():
            self.df.at[index, column] = value

        try:
            self.df.to_csv(APP_SETTINGS.dataset_raw_file, index=False)
        except (OSError, PermissionError, ValueError) as exc:
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
