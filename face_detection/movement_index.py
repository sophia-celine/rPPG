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

def main():

    if len(sys.argv) != 2:

        print()
        print("Uso:")
        print("python detect_landmarks.py video.avi")
        print()

        sys.exit(1)

    video_path = Path(sys.argv[1])

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

        min_face_detection_confidence=0.7,

        min_face_presence_confidence=0.7,

        min_tracking_confidence=0.7
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
            elif last_result is not None:
                frames_without_face += 1
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
                "MediaPipe - Face Body Landmarks",
                frame
            )


            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break


    # ========================================================
    # Finalizar
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()
    print('frames with face:', frames_with_face)
    print('frames without face:', frames_without_face)
    print(
        "Índice de movimento acumulado:",
        f"{accumulated_movement_index:.2f}"
    )


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":
    main()