import cv2
import os
import urllib.request
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURAÇÕES
# ============================================================

INPUT_VIDEO = r"C:\Users\Sophia\Videos\Baumer Video Records\VCXU.2-57C\UTI-20-08-2026\L06-20-08-2026-15-58.avi"
OUTPUT_VIDEO = "output_masked.avi"

MODEL_PATH = "face_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)

import cv2
import os
import urllib.request
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# DOWNLOAD DO MODELO
# ============================================================

def download_model():

    if os.path.exists(MODEL_PATH):
        print(f"Modelo encontrado: {MODEL_PATH}")
        return

    print("Modelo não encontrado.")
    print("Baixando Face Landmarker...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Download concluído.")


# ============================================================
# VARIÁVEIS DA SELEÇÃO
# ============================================================

selected_points = []
polygons = []


# ============================================================
# CALLBACK DO MOUSE
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global selected_points

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    landmarks = param

    # --------------------------------------------------------
    # Encontrar landmark mais próximo do clique
    # --------------------------------------------------------

    min_distance = float("inf")
    closest_index = None

    for i, (px, py) in enumerate(landmarks):

        distance = np.sqrt(
            (x - px) ** 2 +
            (y - py) ** 2
        )

        if distance < min_distance:

            min_distance = distance
            closest_index = i

    # --------------------------------------------------------
    # Só aceitar clique próximo de um landmark
    # --------------------------------------------------------

    MAX_CLICK_DISTANCE = 15

    if (
        closest_index is not None
        and min_distance <= MAX_CLICK_DISTANCE
    ):

        if closest_index not in selected_points:

            selected_points.append(
                closest_index
            )

            print(
                f"Landmark selecionado: "
                f"{closest_index}"
            )


# ============================================================
# SELECIONAR REGIÕES NO PRIMEIRO FRAME
# ============================================================

def select_regions(frame, face_landmarks):

    global selected_points
    global polygons

    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # Coordenadas dos landmarks do primeiro frame
    # --------------------------------------------------------

    landmarks = []

    for landmark in face_landmarks:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        landmarks.append((x, y))

    # --------------------------------------------------------
    # Criar janela
    # --------------------------------------------------------

    window_name = (
        "Selecao - "
        "ENTER = fechar regiao | "
        "S = finalizar | "
        "R = limpar"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
        landmarks
    )

    # ========================================================
    # LOOP DE SELEÇÃO
    # ========================================================

    while True:

        display = frame.copy()

        # ----------------------------------------------------
        # Desenhar todos os landmarks
        # ----------------------------------------------------

        for i, (x, y) in enumerate(landmarks):

            cv2.circle(
                display,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                display,
                str(i),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Desenhar seleção atual
        # ----------------------------------------------------

        for i, index in enumerate(selected_points):

            x, y = landmarks[index]

            cv2.circle(
                display,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display,
                str(i + 1),
                (x + 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Linhas da região atual
        # ----------------------------------------------------

        if len(selected_points) >= 2:

            for i in range(
                len(selected_points) - 1
            ):

                p1 = landmarks[
                    selected_points[i]
                ]

                p2 = landmarks[
                    selected_points[i + 1]
                ]

                cv2.line(
                    display,
                    p1,
                    p2,
                    (0, 0, 255),
                    2
                )

        # ----------------------------------------------------
        # Polígonos já finalizados
        # ----------------------------------------------------

        for polygon in polygons:

            pts = np.array(
                [
                    landmarks[index]
                    for index in polygon
                ],
                dtype=np.int32
            )

            cv2.polylines(
                display,
                [pts],
                True,
                (255, 0, 0),
                2
            )

        # ----------------------------------------------------
        # Instruções
        # ----------------------------------------------------

        cv2.putText(
            display,
            "Clique nos landmarks | "
            "ENTER = fechar regiao | "
            "S = finalizar | R = limpar",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        # ====================================================
        # ENTER
        # ====================================================

        if key == 13:

            if len(selected_points) >= 3:

                polygons.append(
                    selected_points.copy()
                )

                print(
                    "Regiao adicionada:",
                    selected_points
                )

                selected_points = []

            else:

                print(
                    "Selecione pelo menos "
                    "3 landmarks."
                )

        # ====================================================
        # R = LIMPAR
        # ====================================================

        elif key == ord("r"):

            selected_points = []
            polygons = []

            print("Selecao limpa.")

        # ====================================================
        # S = FINALIZAR
        # ====================================================

        elif key == ord("s"):

            if len(selected_points) >= 3:

                polygons.append(
                    selected_points.copy()
                )

            break

    cv2.destroyWindow(window_name)

    return polygons


# ============================================================
# CRIAR MÁSCARA DINÂMICA
# ============================================================

def create_dynamic_mask(
    face_landmarks,
    polygons,
    width,
    height
):

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Converter TODOS os landmarks atuais para pixels
    # --------------------------------------------------------

    points = []

    for landmark in face_landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        x = max(
            0,
            min(width - 1, x)
        )

        y = max(
            0,
            min(height - 1, y)
        )

        points.append((x, y))

    # --------------------------------------------------------
    # Para cada polígono selecionado no primeiro frame
    #
    # Os índices permanecem iguais, mas as coordenadas
    # mudam a cada frame.
    # --------------------------------------------------------

    for polygon in polygons:

        polygon_points = []

        for index in polygon:

            if index < len(points):

                polygon_points.append(
                    points[index]
                )

        if len(polygon_points) >= 3:

            polygon_points = np.array(
                polygon_points,
                dtype=np.int32
            )

            cv2.fillPoly(
                mask,
                [polygon_points],
                255
            )

    return mask


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Baixar modelo
    # --------------------------------------------------------

    download_model()

    # --------------------------------------------------------
    # Configurar MediaPipe
    # --------------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,

        running_mode=vision.RunningMode.VIDEO,

        num_faces=1,

        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,

        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    detector = vision.FaceLandmarker.create_from_options(
        options
    )

    # --------------------------------------------------------
    # Abrir vídeo
    # --------------------------------------------------------

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():

        print(
            f"Erro ao abrir: {INPUT_VIDEO}"
        )

        detector.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # ========================================================
    # PRIMEIRO FRAME
    # ========================================================

    ret, first_frame = cap.read()

    if not ret:

        print("Erro ao ler primeiro frame.")

        cap.release()
        detector.close()

        return

    # --------------------------------------------------------
    # Detectar face
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        first_frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(
        mp_image,
        0
    )

    if len(result.face_landmarks) == 0:

        print(
            "Nenhuma face detectada no "
            "primeiro frame."
        )

        cap.release()
        detector.close()

        return

    first_face_landmarks = (
        result.face_landmarks[0]
    )

    # ========================================================
    # SELEÇÃO
    # ========================================================

    print()
    print("======================================")
    print("SELEÇÃO DE REGIÕES")
    print("======================================")
    print()
    print("Clique nos landmarks que delimitam")
    print("as regiões que devem ficar pretas.")
    print()
    print("ENTER = fechar região")
    print("R     = limpar seleção")
    print("S     = finalizar")
    print()

    polygons = select_regions(
        first_frame,
        first_face_landmarks
    )

    if len(polygons) == 0:

        print(
            "Nenhuma região foi selecionada."
        )

        cap.release()
        detector.close()

        return

    print()
    print("Landmarks selecionados:")
    print()

    for i, polygon in enumerate(polygons):

        print(
            f"Região {i + 1}: {polygon}"
        )

    # ========================================================
    # VIDEO WRITER
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        print(
            "Erro ao criar vídeo de saída."
        )

        cap.release()
        detector.close()

        return

    # ========================================================
    # VOLTAR PARA O PRIMEIRO FRAME
    # ========================================================

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    # ========================================================
    # PROCESSAR
    # ========================================================

    frame_number = 0

    print()
    print("Processando vídeo...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ----------------------------------------------------
        # Converter para RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_ms = int(
            frame_number * 1000 / fps
        )

        # ----------------------------------------------------
        # Detectar landmarks ATUAIS
        # ----------------------------------------------------

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ----------------------------------------------------
        # Se encontrou face
        # ----------------------------------------------------

        if len(result.face_landmarks) > 0:

            face_landmarks = (
                result.face_landmarks[0]
            )

            # ------------------------------------------------
            # CRIAR NOVA MÁSCARA
            #
            # usando os mesmos índices selecionados,
            # mas as coordenadas atuais.
            # ------------------------------------------------

            mask = create_dynamic_mask(
                face_landmarks,
                polygons,
                width,
                height
            )

            # ------------------------------------------------
            # Aplicar máscara
            # ------------------------------------------------

            output = frame.copy()

            output[mask == 255] = 0

        else:

            # Se a face não for detectada,
            # mantém o frame original.
            output = frame

        # ----------------------------------------------------
        # Salvar
        # ----------------------------------------------------

        writer.write(output)

        # ----------------------------------------------------
        # Progresso
        # ----------------------------------------------------

        if frame_number % 30 == 0:

            progress = (
                frame_number /
                frame_count
            ) * 100

            print(
                f"\rProcessando: "
                f"{frame_number}/{frame_count} "
                f"({progress:.1f}%)",
                end=""
            )

    # ========================================================
    # FINALIZAÇÃO
    # ========================================================

    cap.release()
    writer.release()
    detector.close()

    print()
    print()
    print("======================================")
    print("Processamento concluído.")
    print(f"Arquivo: {OUTPUT_VIDEO}")
    print("======================================")


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":
    main()