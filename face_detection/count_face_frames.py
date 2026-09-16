import sys
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path("mediapipe_models") / "face_landmarker.task"


def download_model_if_needed(model_path: Path, url: str):
    if model_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as exc:
        raise RuntimeError(f"Erro ao baixar o modelo: {exc}") from exc


def main():
    if len(sys.argv) != 2:
        print("Uso: py count_face_frames.py caminho_do_video.mp4")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Vídeo não encontrado: {video_path}")
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
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Não foi possível abrir o vídeo: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        frames_with_face = 0
        frames_without_face = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                frames_with_face += 1
            else:
                frames_without_face += 1

            frame_idx += 1

            if frame_idx % 50 == 0 or frame_idx == total_frames:
                print(f"Frames processados: {frame_idx}/{total_frames} | com rosto: {frames_with_face} | sem rosto: {frames_without_face}", flush=True)

    cap.release()

    print("Resumo do vídeo")
    print("---------------")
    print(f"Arquivo: {video_path}")
    print(f"Frames totais: {total_frames}")
    print(f"Frames com rosto detectado: {frames_with_face}")
    print(f"Frames sem rosto: {frames_without_face}")
    print(f"Taxa de detecção: {frames_with_face / total_frames * 100:.2f}%" if total_frames else "Taxa de detecção: 0%")


if __name__ == "__main__":
    main()
