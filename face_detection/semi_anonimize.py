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

# INPUT_VIDEO = r"C:\Users\Sophia\Pictures\Camera Roll\WIN_20260826_19_21_02_Pro.mp4"
INPUT_VIDEO = r"A:\UTI-20-08-2026\L06-20-08-2026-15-54.avi"
OUTPUT_VIDEO = "output_bg_normal.avi"

MODEL_PATH = "face_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)


# ============================================================
# BAIXAR MODELO
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
# CRIAR MÁSCARA DOS OLHOS + SOBRANCELHAS + BOCA
# ============================================================

def create_black_mask(face_landmarks, width, height):

    # Máscara inicialmente preta.
    # Branco = região que será transformada em preto.
    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Converter landmarks normalizados para pixels
    # --------------------------------------------------------

    points = []

    for landmark in face_landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        points.append([x, y])

    points = np.array(
        points,
        dtype=np.int32
    )

    # ========================================================
    # OLHO ESQUERDO + SOBRANCELHA
    # ========================================================

    LEFT_EYE_BROW = [
        # ----------------------------------------------------
        # Olho esquerdo
        # ----------------------------------------------------

        33,
        133,

        160,
        159,
        158,
        157,

        173,

        153,
        144,
        145,

        163,
        7,
        35,
        228,

        # ----------------------------------------------------
        # Sobrancelha esquerda
        # ----------------------------------------------------

        46,
        53,
        52,
        65,
        55,
        70, 63, 105, 66, 107, 24
    ]

    # ========================================================
    # OLHO DIREITO + SOBRANCELHA
    # ========================================================

    RIGHT_EYE_BROW = [
        # ----------------------------------------------------
        # Olho direito
        # ----------------------------------------------------

        362,
        263,

        387,
        386,
        385,
        384,

        398,

        373,
        380,
        381,

        390,
        249,
        383,
        449,
        464,

        # ----------------------------------------------------
        # Sobrancelha direita
        # ----------------------------------------------------

        276,
        283,
        282,
        295,
        285,
        336,
        296,
        334
    ]

    # ========================================================
    # BOCA
    # ========================================================

    MOUTH = [
        61,
        146,
        91,
        181,
        84,

        17,

        314,
        405,
        321,
        375,

        291,

        409,
        270,
        269,
        267,

        0,
        37,
        39,
        40,
        185
    ]

    # ========================================================
    # FUNÇÃO PARA PREENCHER REGIÃO
    # ========================================================

    def fill_region(indices):

        region = []

        for idx in indices:

            if idx < len(points):
                region.append(points[idx])

        if len(region) >= 3:

            region = np.array(
                region,
                dtype=np.int32
            )

            # Convex hull para criar uma região fechada
            hull = cv2.convexHull(region)

            cv2.fillConvexPoly(
                mask,
                hull,
                255
            )

    # --------------------------------------------------------
    # Preencher regiões
    # --------------------------------------------------------

    fill_region(LEFT_EYE_BROW)
    fill_region(RIGHT_EYE_BROW)
    fill_region(MOUTH)

    return mask


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    download_model()

    # --------------------------------------------------------
    # Configurar Face Landmarker
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
            f"Erro: não foi possível abrir "
            f"{INPUT_VIDEO}"
        )

        detector.close()
        return

    # --------------------------------------------------------
    # Informações do vídeo
    # --------------------------------------------------------

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

    print()
    print("Informações do vídeo:")
    print(f"  Resolução: {width} x {height}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {frame_count}")
    print()

    # --------------------------------------------------------
    # Criar vídeo de saída
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"XVID"
        # *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        print("Erro ao criar vídeo de saída.")

        cap.release()
        detector.close()

        return

    # --------------------------------------------------------
    # Processamento
    # --------------------------------------------------------

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ----------------------------------------------------
        # BGR -> RGB
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
        # Detectar face
        # ----------------------------------------------------

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ----------------------------------------------------
        # Se encontrou uma face
        # ----------------------------------------------------

        if len(result.face_landmarks) > 0:

            face_landmarks = result.face_landmarks[0]

            # Criar máscara
            mask = create_black_mask(
                face_landmarks,
                width,
                height
            )

            # ------------------------------------------------
            # Criar uma imagem completamente preta
            # ------------------------------------------------

            black = np.zeros_like(frame)

            # ------------------------------------------------
            # Manter o frame original onde a máscara é preta
            # e usar preto onde a máscara é branca.
            # ------------------------------------------------

            output = np.where(
                mask[:, :, None] == 255,
                black,
                frame
            )

        else:

            # Se não encontrou face,
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

            if frame_count > 0:

                progress = (
                    frame_number / frame_count
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
    print("Processamento concluído.")
    print(
        f"Vídeo salvo em: {OUTPUT_VIDEO}"
    )


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":
    main()