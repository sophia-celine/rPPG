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

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/"
    "pose_landmarker_lite.task"
)

MODEL_DIR = Path("mediapipe_models")

FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"


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

    download_model(
        POSE_MODEL_URL,
        POSE_MODEL_PATH
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
    # Configurar Pose Landmarker
    # --------------------------------------------------------

    pose_base_options = python.BaseOptions(
        model_asset_path=str(POSE_MODEL_PATH)
    )

    pose_options = vision.PoseLandmarkerOptions(

        base_options=pose_base_options,

        running_mode=vision.RunningMode.VIDEO,

        num_poses=1,

        min_pose_detection_confidence=0.5,

        min_pose_presence_confidence=0.5,

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


    # ========================================================
    # Criar os Landmarkers
    # ========================================================

    with vision.FaceLandmarker.create_from_options(
        face_options
    ) as face_landmarker, \
         vision.PoseLandmarker.create_from_options(
             pose_options
         ) as pose_landmarker:


        # ====================================================
        # Loop do vídeo
        # ====================================================

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
            # DESENHAR LANDMARKS DO CORPO
            # =================================================

            # if pose_result.pose_landmarks:

            #     for pose_landmarks in pose_result.pose_landmarks:

            #         # -----------------------------------------
            #         # Pontos
            #         # -----------------------------------------

            #         for landmark in pose_landmarks:

            #             x, y = landmark_to_pixel(
            #                 landmark,
            #                 width,
            #                 height
            #             )

            #             cv2.circle(
            #                 frame,
            #                 (x, y),
            #                 4,
            #                 (0, 0, 255),
            #                 -1
            #             )


                    # -----------------------------------------
                    # Conexões entre os pontos
                    # -----------------------------------------

                    # for connection in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS:

                    #     start_index = connection.start
                    #     end_index = connection.end

                    #     start_landmark = pose_landmarks[
                    #         start_index
                    #     ]

                    #     end_landmark = pose_landmarks[
                    #         end_index
                    #     ]

                    #     x1, y1 = landmark_to_pixel(
                    #         start_landmark,
                    #         width,
                    #         height
                    #     )

                    #     x2, y2 = landmark_to_pixel(
                    #         end_landmark,
                    #         width,
                    #         height
                    #     )

                    #     cv2.line(
                    #         frame,
                    #         (x1, y1),
                    #         (x2, y2),
                    #         (255, 0, 0),
                    #         2
                    #     )



            # =================================================
            # MOSTRAR
            # =================================================

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


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":
    main()