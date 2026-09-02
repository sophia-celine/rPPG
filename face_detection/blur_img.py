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

INPUT_IMAGE_PATH = r"C:\Users\Sophia\Downloads\Gemini_Generated_Image_rjd6qrjd6qrjd6qr.png"
OUTPUT_IMAGE_PATH = "output_anonimizado14.jpg"


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

    # Mapeamento dos índices
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
    print(LEFT_EYE_IDXS)

    # --------------------------------------------------------
    # Configuração para modo IMAGEM
    # --------------------------------------------------------

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    detector = vision.FaceLandmarker.create_from_options(options)

    # --------------------------------------------------------
    # Carregar imagem
    # --------------------------------------------------------

    frame = cv2.imread(INPUT_IMAGE_PATH)

    if frame is None:
        print(f"Erro: não foi possível carregar a imagem '{INPUT_IMAGE_PATH}'.")
        return

    height, width, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Detecção na imagem
    result = detector.detect(mp_image)

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
                thickness=eye_dist * 2
            )

            # 4. Blur SOMENTE fora do rosto
            outside_blur_ring = cv2.bitwise_and(thick_border, cv2.bitwise_not(face_interior))

            # 5. Máscara cobrindo Olhos + Sobrancelhas e Boca
            features_mask = np.zeros((height, width), dtype=np.uint8)

            outer_left_eye = [22, 23, 24, 25, 26, 110, 112, 226]
            outer_right_eye = [252, 253, 254, 255, 341, 446]
            left_eye_brow_hull = cv2.convexHull(points[LEFT_EYE_IDXS + LEFT_EYEBROW_IDXS + outer_right_eye])
            cv2.fillConvexPoly(features_mask, left_eye_brow_hull, 255)

           
            right_eye_brow_hull = cv2.convexHull(points[RIGHT_EYE_IDXS + RIGHT_EYEBROW_IDXS + outer_left_eye])
            cv2.fillConvexPoly(features_mask, right_eye_brow_hull, 255)

            cv2.fillConvexPoly(features_mask, cv2.convexHull(points[LIPS_IDXS]), 255)

            nose_triangle = points[[168, 98, 327]]
            cv2.fillConvexPoly(features_mask, nose_triangle, 255)

            # 6. Combina as máscaras
            final_mask = cv2.bitwise_or(features_mask, outside_blur_ring)

            # 7. Suavização das bordas
            mask_blurred = cv2.GaussianBlur(final_mask, (15, 15), 0)
            mask_3d = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR) / 255.0

            # 8. Aplicação do desfoque
            k_size = (eye_dist // 2) * 2 + 1
            k_size = max(k_size, 3)
            blurred_frame = cv2.GaussianBlur(
                frame,
                (k_size * 2 + 1, k_size * 2 + 1),
                0
            )

            frame = (frame * (1 - mask_3d) + blurred_frame * mask_3d).astype(np.uint8)

    # --------------------------------------------------------
    # Salvar e Exibir Resultado
    # --------------------------------------------------------

    cv2.imwrite(OUTPUT_IMAGE_PATH, frame)
    print(f"Imagem salva em: {OUTPUT_IMAGE_PATH}")

    cv2.imshow("Foto Anonimizada (Pressione qualquer tecla para fechar)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()