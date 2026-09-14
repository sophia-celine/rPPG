import sys
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path("mediapipe_models") / "face_landmarker.task"


def download_model_if_needed(model_path: Path, url: str):
    if model_path.exists():
        print(f"Modelo já existe: {model_path}")
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando modelo: {url}")
    import urllib.request

    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"Modelo salvo em: {model_path}")
    except Exception as exc:
        raise RuntimeError(f"Erro ao baixar o modelo: {exc}") from exc


def landmark_to_pixel(landmark, width, height):
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def main():
    if len(sys.argv) != 2:
        print()
        print("Uso:")
        print("python show_landmarks_image.py imagem.jpg")
        print()
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Imagem não encontrada: {image_path}")
        sys.exit(1)

    model_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/"
        "face_landmarker.task"
    )

    try:
        download_model_if_needed(MODEL_PATH, model_url)
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Não foi possível abrir a imagem: {image_path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        print("Nenhum rosto foi detectado na imagem.")
        cv2.imshow("Imagem", image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    height, width = image_bgr.shape[:2]

    for face_landmarks in result.face_landmarks:
        for landmark in face_landmarks:
            x, y = landmark_to_pixel(landmark, width, height)
            cv2.circle(image_bgr, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Landmarks detectados", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
