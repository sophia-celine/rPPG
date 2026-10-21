import math
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().with_name("face_landmarker.task")

MOVEMENT_THRESHOLD = 0.01


def download_model():
    if MODEL_PATH.exists():
        return

    print(f"Baixando Face Landmarker para {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download concluído.")


def main():
    if len(sys.argv) != 2:
        print("Uso: python live_landmarks_movement.py caminho_do_video.mp4")
        return

    video_path = Path(sys.argv[1])
    if not video_path.is_file():
        print(f"Vídeo não encontrado: {video_path}")
        return

    download_model()

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        print(f"Erro: não foi possível abrir o vídeo: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    print(f"Vídeo aberto: {video_path}")
    print("Pressione Q para sair.")

    movement_indices: list[float] = []
    previous_landmarks = None
    last_timestamp_ms = -1
    frame_number = 0

    try:
        with vision.FaceLandmarker.create_from_options(options) as detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Erro ao capturar frame.")
                    break

                height, width, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                timestamp_ms = max(
                    int(frame_number * 1000 / fps),
                    last_timestamp_ms + 1,
                )
                frame_number += 1
                last_timestamp_ms = timestamp_ms
                result = detector.detect_for_video(mp_image, timestamp_ms)

                current_landmarks = (
                    result.face_landmarks[0]
                    if result.face_landmarks
                    else None
                )
                landmark_count = sum(
                    len(face_landmarks)
                    for face_landmarks in result.face_landmarks
                )

                changed_fraction = 0.0
                average_distance_px = 0.0
                movement_index = 0.0

                if current_landmarks is not None:
                    for landmark in current_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                    if (
                        previous_landmarks is not None
                        and len(previous_landmarks) == len(current_landmarks)
                    ):
                        distances_px = []
                        changed_count = 0

                        for previous, current in zip(
                            previous_landmarks,
                            current_landmarks,
                        ):
                            normalized_distance = math.hypot(
                                current.x - previous.x,
                                current.y - previous.y,
                            )
                            distance_px = math.hypot(
                                (current.x - previous.x) * width,
                                (current.y - previous.y) * height,
                            )
                            distances_px.append(distance_px)

                            if normalized_distance > MOVEMENT_THRESHOLD:
                                changed_count += 1

                        if distances_px:
                            changed_fraction = changed_count / len(distances_px)
                            average_distance_px = (
                                sum(distances_px) / len(distances_px)
                            )
                            movement_index = (
                                changed_fraction * average_distance_px
                            )

                    previous_landmarks = current_landmarks
                else:
                    previous_landmarks = None

                movement_indices.append(movement_index)

                overlay = (
                    f"Landmarks: {landmark_count}",
                    f"Movidos (> {MOVEMENT_THRESHOLD:.3f} normalizado): "
                    f"{changed_fraction * 100:.1f}%",
                    f"Distancia media: {average_distance_px:.2f} px",
                    f"Indice de movimento: {movement_index:.2f}",
                )
                for line_number, text in enumerate(overlay):
                    cv2.putText(
                        frame,
                        text,
                        (20, 35 + line_number * 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Landmarks e indice de movimento", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
