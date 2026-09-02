import os
import math
import urllib.request
import cv2
import numpy as np
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

INPUT_VIDEO_PATH = r"C:\Users\Sophia\Videos\Baumer Video Records\VCXU.2-57C\UTI-20-08-2026\L06-20-08-2026-15-54.avi"
OUTPUT_VIDEO_PATH = "output_anonimizado.mp4"


# ============================================================
# BAIXAR O MODELO
# ============================================================

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Modelo encontrado: {MODEL_PATH}")
        return

    print("Modelo não encontrado.")
    print("Baixando Face Landmarker...")

    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download concluído.")


def get_indices_from_connections(connections):
    indices = set()
    for conn in connections:
        indices.add(conn.start)
        indices.add(conn.end)
    return list(indices)


# ============================================================
# MAIN
# ============================================================

def main():

    download_model()

    # Mapeamento dos índices dos olhos, sobrancelhas, lábios e contorno facial
    LEFT_EYE_IDXS = get_indices_from_connections(
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
    )
    RIGHT_EYE_IDXS = get_indices_from_connections(
        vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
    )
    LEFT_EYEBROW_IDXS = get_indices_from_connections(
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW
    )
    RIGHT_EYEBROW_IDXS = get_indices_from_connections(
        vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW
    )
    LIPS_IDXS = get_indices_from_connections(
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
    )
    FACE_OVAL_IDXS = get_indices_from_connections(
        vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
    )

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

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

    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    if not cap.isOpened():
        print(f"Erro: não foi possível abrir o vídeo '{INPUT_VIDEO_PATH}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print("Processando e exibindo vídeo... Pressione 'Q' na janela para cancelar.")

    frame_timestamp_ms = 0
    frame_duration_ms = int(1000 / fps)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        frame_timestamp_ms += frame_duration_ms
        result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                points = np.array([
                    [int(lm.x * width), int(lm.y * height)]
                    for lm in face_landmarks
                ])

                # 1. Distância entre os cantos externos dos olhos (pontos 33 e 263)
                p_left_eye = points[33]
                p_right_eye = points[263]
                eye_dist = int(math.hypot(
                    p_left_eye[0] - p_right_eye[0],
                    p_left_eye[1] - p_right_eye[1]
                ))
                eye_dist = max(eye_dist, 1)

                # 2. Máscara do interior do rosto
                face_oval_hull = cv2.convexHull(points[FACE_OVAL_IDXS])
                face_interior = np.zeros((height, width), dtype=np.uint8)
                cv2.fillConvexPoly(face_interior, face_oval_hull, 255)

                # 3. Borda do contorno do rosto (largura 2 * eye_dist)
                thick_border = np.zeros((height, width), dtype=np.uint8)
                cv2.polylines(
                    thick_border,
                    [face_oval_hull],
                    isClosed=True,
                    color=255,
                    thickness=eye_dist * 3
                )

                # 4. Blur SOMENTE fora do rosto
                outside_blur_ring = cv2.bitwise_and(thick_border, cv2.bitwise_not(face_interior))

                # 5. Máscara cobrindo Olhos + Sobrancelhas e Boca
                features_mask = np.zeros((height, width), dtype=np.uint8)

                # Une pontos do olho esquerdo + sobrancelha esquerda
                left_eye_brow_hull = cv2.convexHull(points[LEFT_EYE_IDXS + LEFT_EYEBROW_IDXS])
                cv2.fillConvexPoly(features_mask, left_eye_brow_hull, 255)

                # Une pontos do olho direito + sobrancelha direita
                right_eye_brow_hull = cv2.convexHull(points[RIGHT_EYE_IDXS + RIGHT_EYEBROW_IDXS])
                cv2.fillConvexPoly(features_mask, right_eye_brow_hull, 255)

                # Boca
                cv2.fillConvexPoly(features_mask, cv2.convexHull(points[LIPS_IDXS]), 255)

                # 6. Combina a máscara de feições com o anel do contorno externo
                final_mask = cv2.bitwise_or(features_mask, outside_blur_ring)

                # 7. Suavização leve para transição suave
                mask_blurred = cv2.GaussianBlur(final_mask, (15, 15), 0)
                mask_3d = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR) / 255.0

                # 8. Criação e aplicação do desfoque
                k_size = (eye_dist // 2) * 2 + 1
                k_size = max(k_size, 3)
                blurred_frame = cv2.GaussianBlur(
                    frame,
                    (k_size * 2 + 1, k_size * 2 + 1),
                    0
                )

                frame = (frame * (1 - mask_3d) + blurred_frame * mask_3d).astype(np.uint8)

        out.write(frame)

        cv2.imshow("Processando Vídeo (Pressione 'Q' para interromper)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processamento cancelado pelo usuário.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"Processamento concluído! Vídeo salvo em: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()