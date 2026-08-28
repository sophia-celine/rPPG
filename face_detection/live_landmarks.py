import cv2
import os
import urllib.request
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)

MODEL_PATH = "face_landmarker.task"

CAMERA_INDEX = 0


# ============================================================
# BAIXAR O MODELO
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
# MAIN
# ============================================================

def main():

    download_model()

    # --------------------------------------------------------
    # Configuração do Face Landmarker
    # --------------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,

        # Número máximo de faces
        num_faces=1,

        # Detecção
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,

        # Blendshapes não são necessários aqui
        output_face_blendshapes=False,

        # Matriz de transformação não é necessária
        output_facial_transformation_matrixes=False
    )

    detector = vision.FaceLandmarker.create_from_options(options)

    # --------------------------------------------------------
    # Abrir câmera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera.")
        return

    # Tente manter uma resolução razoável
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Câmera aberta.")
    print("Pressione Q para sair.")

    frame_timestamp_ms = 0

    # --------------------------------------------------------
    # Loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar frame.")
            break

        # OpenCV -> RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Transformar para MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # O timestamp precisa ser crescente
        frame_timestamp_ms += 33

        # Detectar face
        result = detector.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        # ----------------------------------------------------
        # Desenhar landmarks
        # ----------------------------------------------------

        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                h, w, _ = frame.shape

                # Desenhar cada landmark
                for landmark in face_landmarks:

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    # Verificar se está dentro da imagem
                    if (
                        0 <= x < w
                        and 0 <= y < h
                    ):
                        cv2.circle(
                            frame,
                            (x, y),
                            1,
                            (0, 255, 0),
                            -1
                        )

        # ----------------------------------------------------
        # Mostrar número de faces
        # ----------------------------------------------------

        text = f"Faces: {len(result.face_landmarks)}"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Mostrar imagem
        # ----------------------------------------------------

        cv2.imshow(
            "MediaPipe Face Landmarks",
            frame
        )

        # Q para sair
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # --------------------------------------------------------
    # Finalização
    # --------------------------------------------------------

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()