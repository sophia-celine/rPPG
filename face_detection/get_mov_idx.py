import csv
import math
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURAÇÃO DOS MODELOS
# ============================================================

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)

MODEL_DIR = Path("mediapipe_models")

FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"

MOVEMENT_INTERVAL_FRAMES = 25
MOVEMENT_THRESHOLD_EYE_DISTANCE_FACTOR = 0.1


# ============================================================
# DOWNLOAD DOS MODELOS
# ============================================================

def download_model(url, destination):

    if destination.exists():
        print(f"Modelo já existe: {destination}")
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Baixando modelo:")
    print(url)

    try:
        urllib.request.urlretrieve(
            url,
            destination
        )

    except Exception as e:
        print(f"Erro ao baixar o modelo: {e}")
        sys.exit(1)

    print(f"Modelo salvo em: {destination}")


# ============================================================
# CONVERTER LANDMARK NORMALIZADO PARA PIXEL
# ============================================================

def landmark_to_pixel(landmark, width, height):

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    return x, y


def eye_distance_pixels(landmarks, width, height):

    left_eye_center = (
        (landmarks[33].x + landmarks[133].x) * width / 2,
        (landmarks[33].y + landmarks[133].y) * height / 2,
    )
    right_eye_center = (
        (landmarks[362].x + landmarks[263].x) * width / 2,
        (landmarks[362].y + landmarks[263].y) * height / 2,
    )

    return math.hypot(
        right_eye_center[0] - left_eye_center[0],
        right_eye_center[1] - left_eye_center[1],
    )


# ============================================================
# MAIN
# ============================================================

def get_mov_index(video_path, show: bool = True):

    video_path = Path(video_path)

    if not video_path.exists():

        print(f"Vídeo não encontrado: {video_path}")

        sys.exit(1)


    # --------------------------------------------------------
    # Baixar modelos
    # --------------------------------------------------------

    download_model(
        FACE_MODEL_URL,
        FACE_MODEL_PATH
    )


    # --------------------------------------------------------
    # Configurar Face Landmarker
    # --------------------------------------------------------

    face_base_options = python.BaseOptions(
        model_asset_path=str(FACE_MODEL_PATH)
    )

    face_options = vision.FaceLandmarkerOptions(

        base_options=face_base_options,

        running_mode=vision.RunningMode.VIDEO,

        num_faces=1,

        min_face_detection_confidence=0.5,

        min_face_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )


    # --------------------------------------------------------
    # Abrir vídeo
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"Não foi possível abrir o vídeo: {video_path}"
        )

        sys.exit(1)


    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0


    if show:
        print()
        print("Informações do vídeo")
        print("--------------------")
        print(f"Arquivo:   {video_path}")
        print(f"Resolução: {width} x {height}")
        print(f"FPS:       {fps:.2f}")
        print()
        print("Pressione Q ou ESC para sair.")
        print()


    # Número do frame
    frame_number = 0
    movement_frames = []
    reference_distances = []
    movement_percentages = []
    average_displacements = []
    average_displacements_reference_units = []
    movement_indices = []
    accumulated_movement_indices = []
    accumulated_movement_index = 0.0
    MOVEMENT_THRESHOLD_PIXELS = 0.0
    previous_landmarks = None
    previous_measurement_frame = None


    # ========================================================
    # Criar os Landmarkers
    # ========================================================

    with vision.FaceLandmarker.create_from_options(
        face_options
    ) as face_landmarker:

        # ====================================================
        # Loop do vídeo
        # ====================================================
        frames_with_face = 0
        frames_without_face = 0
        last_result = None
        while True:

            ret, frame = cap.read()

            if not ret:
                break


            # ------------------------------------------------
            # Timestamp
            # ------------------------------------------------

            timestamp_ms = int(
                frame_number * 1000 / fps
            )

            frame_number += 1


            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # Criar imagem do MediaPipe
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )


            # ------------------------------------------------
            # Detectar rosto
            # ------------------------------------------------

            face_result = face_landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if frame_number == 1 and not face_result.face_landmarks:
                print("rosto não identificado no primeiro frame")
                cap.release()
                cv2.destroyAllWindows()
                return None

            # ================================================
            # CALCULAR MOVIMENTAÇÃO DOS LANDMARKS
            # ================================================

            if face_result.face_landmarks:
                current_landmarks = face_result.face_landmarks[0]
                current_frame_number = frame_number - 1

                if previous_landmarks is None:
                    previous_landmarks = current_landmarks
                    previous_measurement_frame = current_frame_number
                elif (
                    current_frame_number - previous_measurement_frame
                    >= MOVEMENT_INTERVAL_FRAMES
                ):
                    if len(previous_landmarks) == len(current_landmarks):
                        MOVEMENT_THRESHOLD_PIXELS = (
                            MOVEMENT_THRESHOLD_EYE_DISTANCE_FACTOR
                            * eye_distance_pixels(
                                current_landmarks,
                                width,
                                height
                            )
                        )
                        distances = [
                            math.hypot(
                                (current.x - previous.x) * width,
                                (current.y - previous.y) * height,
                            )
                            for previous, current in zip(
                                previous_landmarks,
                                current_landmarks
                            )
                        ]

                        moved_count = sum(
                            distance > MOVEMENT_THRESHOLD_PIXELS
                            for distance in distances
                        )
                        movement_percentage = (
                            moved_count / len(distances) * 100
                            if distances
                            else 0.0
                        )
                        moved_fraction = (
                            moved_count / len(distances)
                            if distances
                            else 0.0
                        )
                        average_displacement = (
                            sum(distances) / len(distances)
                            if distances
                            else 0.0
                        )
                        average_displacement_reference_units = (
                            average_displacement / MOVEMENT_THRESHOLD_PIXELS
                            if MOVEMENT_THRESHOLD_PIXELS > 0
                            else 0.0
                        )
                        movement_index = (
                            moved_fraction * average_displacement_reference_units
                        )
                        accumulated_movement_index += movement_index

                        movement_frames.append(current_frame_number)
                        reference_distances.append(MOVEMENT_THRESHOLD_PIXELS)
                        movement_percentages.append(movement_percentage)
                        average_displacements.append(average_displacement)
                        average_displacements_reference_units.append(
                            average_displacement_reference_units
                        )
                        movement_indices.append(movement_index)
                        accumulated_movement_indices.append(
                            accumulated_movement_index
                        )

                    previous_landmarks = current_landmarks
                    previous_measurement_frame = current_frame_number


            # ------------------------------------------------
            # Detectar corpo
            # ------------------------------------------------

            # pose_result = pose_landmarker.detect_for_video(
            #     mp_image,
            #     timestamp_ms
            # )


            # =================================================
            # DESENHAR LANDMARKS DO ROSTO
            # =================================================

            if face_result.face_landmarks:
                frames_with_face += 1
                last_result = face_result

                if show:
                    for face_landmarks in face_result.face_landmarks:

                        for landmark in face_landmarks:

                            x, y = landmark_to_pixel(
                                landmark,
                                width,
                                height
                            )

                            cv2.circle(
                                frame,
                                (x, y),
                                1,
                                (0, 255, 0),
                                -1
                            )
            else:
                frames_without_face += 1
                if show and last_result is not None:
                    face_result = last_result
                    for face_landmarks in face_result.face_landmarks:

                        for landmark in face_landmarks:

                            x, y = landmark_to_pixel(
                                landmark,
                                width,
                                height
                            )

                            cv2.circle(
                                frame,
                                (x, y),
                                1,
                                (0, 255, 0),
                                -1
                            )

            # =================================================
            # MOSTRAR
            # =================================================

            if show:
                if movement_percentages:
                    cv2.putText(
                        frame,
                        (
                            "Distancia de referencia: "
                            f"{reference_distances[-1]:.2f} px"
                        ),
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        (
                            "Landmarks movidos (> referencia): "
                            f"{movement_percentages[-1]:.1f}%"
                        ),
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Deslocamento medio: {average_displacements[-1]:.2f} px",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        (
                            "Deslocamento medio: "
                            f"{average_displacements_reference_units[-1]:.2f} "
                            "unidades de referencia"
                        ),
                        (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Indice de movimento: {movement_indices[-1]:.2f}",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        (
                            "Indice de movimento acumulado: "
                            f"{accumulated_movement_indices[-1]:.2f}"
                        ),
                        (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                else:
                    cv2.putText(
                        frame,
                        "Aguardando primeira medicao de movimento",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                cv2.imshow(
                    f"{Path(video_path).parent.name}",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break


    # ========================================================
    # Finalizar
    # ========================================================

    cap.release()

    if show:
        cv2.destroyAllWindows()
    print('frames with face:', frames_with_face)
    print('frames without face:', frames_without_face)
    print(
        "Índice de movimento acumulado:",
        f"{accumulated_movement_index:.2f}"
    )
    total_frames = frames_with_face + frames_without_face
    detection_rate = frames_with_face / total_frames if total_frames else 0.0

    return accumulated_movement_index, detection_rate

def process_dataset(
    csv_path='',
    folder_path=None,
    dataset_csv_path='',
    dataset_cas_path=None,
    show=False
):
    csv_file = Path(csv_path)
    dataset_folder = Path(folder_path) if folder_path is not None else None
    if dataset_csv_path and dataset_cas_path:
        raise ValueError(
            "Informe apenas um dos parâmetros: dataset_csv_path ou dataset_cas_path."
        )
    dataset_source_path = dataset_cas_path or dataset_csv_path
    dataset_csv_file = Path(dataset_source_path) if dataset_source_path else None

    if not csv_file.is_file():
        raise FileNotFoundError(f"Planilha CSV não encontrada: {csv_file}")

    if dataset_csv_file is None or not dataset_csv_file.is_file():
        raise FileNotFoundError(
            f"Planilha de dados do dataset não encontrada: {dataset_source_path}"
        )

    if dataset_folder is None or not dataset_folder.is_dir():
        raise FileNotFoundError(
            f"Pasta do conjunto de dados não encontrada: {folder_path}"
        )

    with dataset_csv_file.open("r", newline="", encoding="utf-8-sig") as source:
        dataset_rows = list(csv.reader(source))

    if not dataset_rows:
        raise ValueError(f"A planilha do dataset está vazia: {dataset_csv_file}")

    dataset_headers = dataset_rows[0]
    if not dataset_headers:
        raise ValueError(
            f"A primeira linha da planilha do dataset não contém cabeçalhos."
        )

    try:
        illumination_column = dataset_headers.index("Iluminação (lux)")
    except ValueError as error:
        raise ValueError(
            "A planilha do dataset não contém a coluna 'Iluminação (lux)'."
        ) from error

    illumination_by_patient = {}
    for row_number, dataset_row in enumerate(dataset_rows[1:], start=2):
        if not dataset_row or not dataset_row[0].strip():
            print('nao achou a linha')
            continue

        patient_name = dataset_row[0].strip().replace('.', '_')
        if patient_name in illumination_by_patient:
            raise ValueError(
                f"Paciente duplicado na planilha do dataset: {patient_name}"
            )
        if illumination_column >= len(dataset_row):
            raise ValueError(
                f"A linha {row_number} da planilha do dataset não possui "
                "a coluna 'Iluminação (lux)'."
            )
        illumination_by_patient[patient_name] = dataset_row[illumination_column]

    batch_folders = sorted(
        folder
        for folder in dataset_folder.iterdir()
        if folder.is_dir()
        and folder.name.lower().startswith("batch")
        and folder.name[5:].isdigit()
    )

    if batch_folders:
        patient_parent_folders = batch_folders
    else:
        patient_parent_folders = [dataset_folder]

    patient_folders = sorted(
        (
            patient_folder
            for parent_folder in patient_parent_folders
            for patient_folder in parent_folder.iterdir()
            if patient_folder.is_dir() #and patient_folder.name.isalnum()
        ),
        key=lambda folder: (folder.name, str(folder)),
    )
    print('patient folders', patient_folders)

    patient_indices = {}
    for patient_folder in patient_folders:
        patient_name = patient_folder.name

        if patient_name in patient_indices:
            raise ValueError(
                f"Nome de paciente duplicado entre pastas: {patient_name}"
            )

        avi_files = sorted(
            path
            for path in patient_folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".avi"
        )

        if len(avi_files) > 1:
            print(
                f"Paciente {patient_name} ignorado: "
                f"mais de um arquivo AVI encontrado."
            )
            continue

        if not avi_files:
            print(f"Paciente {patient_name} ignorado: nenhum arquivo AVI encontrado.")
            continue
        print(f'================= {patient_name} ==================')
        result = get_mov_index(avi_files[0], show=show)
        if result is not None:
            movement_index, detection_rate = result
            patient_indices[patient_name] = (movement_index, detection_rate)

    with csv_file.open("r", newline="", encoding="utf-8-sig") as source:
        rows = list(csv.reader(source))

    if not rows:
        raise ValueError(f"A planilha CSV está vazia: {csv_file}")

    headers = rows[0]
    if not headers:
        raise ValueError(f"A primeira linha da planilha não contém cabeçalhos.")

    patient_columns = {}
    for column_index, patient_name in enumerate(headers[1:], start=1):
        if patient_name in patient_columns:
            raise ValueError(
                f"Paciente duplicado nos cabeçalhos da planilha: {patient_name}"
            )
        if patient_name:
            patient_columns[patient_name] = column_index

    found_indices = {
        patient_name: patient_metrics
        for patient_name, patient_metrics in patient_indices.items()
        if patient_name in patient_columns
    }
    missing_indices = {
        patient_name: patient_metrics
        for patient_name, patient_metrics in patient_indices.items()
        if patient_name not in patient_columns
    }

    movement_row = [""] * len(headers)
    movement_row[0] = "movement_index"
    detection_rate_row = [""] * len(headers)
    detection_rate_row[0] = "detection_rate"
    illumination_row = [""] * len(headers)
    illumination_row[0] = "Iluminação (lux)"
    for patient_name, (movement_index, detection_rate) in found_indices.items():
        movement_row[patient_columns[patient_name]] = movement_index
        detection_rate_row[patient_columns[patient_name]] = detection_rate
        illumination_row[patient_columns[patient_name]] = (
            illumination_by_patient.get(patient_name, "")
        )
    rows.append(movement_row)
    rows.append(detection_rate_row)
    rows.append(illumination_row)

    output_file = csv_file.with_name(f"{csv_file.stem}_mov{csv_file.suffix}")
    with output_file.open("w", newline="", encoding="utf-8") as output:
        csv.writer(output).writerows(rows)

    missing_file = None
    if missing_indices:
        missing_file = csv_file.with_name(
            f"{csv_file.stem}_mov_not_found{csv_file.suffix}"
        )
        with missing_file.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["metric", *missing_indices.keys()])
            writer.writerow([
                "movement_index",
                *(metrics[0] for metrics in missing_indices.values()),
            ])
            writer.writerow([
                "detection_rate",
                *(metrics[1] for metrics in missing_indices.values()),
            ])
            writer.writerow([
                "Iluminação (lux)",
                *(
                    illumination_by_patient.get(patient_name, "")
                    for patient_name in missing_indices
                ),
            ])

    print(f"Planilha com índices de movimento salva em: {output_file}")
    if missing_file is not None:
        print(f"Pacientes ausentes na planilha original salvos em: {missing_file}")

    return output_file, missing_file


if __name__ == "__main__":
    process_dataset(
        csv_path='/home/soph/rppg/rPPG_data/results/results_mp.csv',
        folder_path='/home/soph/ssd/dataset_raw',
        dataset_cas_path='/home/soph/rppg/rPPG_data/ground_truth/dataset_raw.csv',
        show=True
    )