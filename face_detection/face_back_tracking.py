"""Utilities for recovering faces missed in the initial video frames."""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


def find_first_face_and_backtrack(
    cap,
    face_landmarker=None,
    fps=25.0,
    face_options=None,
):
    """Find the first frame containing a face, then rewind the capture.

    Args:
        cap: An open OpenCV video capture.
        face_landmarker: MediaPipe face landmarker configured for VIDEO mode.
        fps: Video frame rate, used to generate the detection timestamps.
        face_options: Optional FaceLandmarker options. When provided, the
            scan uses an independent landmarker and does not affect the
            landmarker used by the normal video loop.

    Returns:
        A tuple ``(face_result, frame_index)`` for the first detected face.
        ``(None, -1)`` is returned if the video contains no detected face.
        The capture is rewound to frame zero before returning.
    """
    if fps <= 0:
        fps = 25.0

    def scan(landmarker):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return None, -1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            timestamp_ms = int(frame_index * 1000 / fps)
            face_result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            if face_result.face_landmarks:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return face_result, frame_index

            frame_index += 1

    if face_options is not None:
        with vision.FaceLandmarker.create_from_options(face_options) as landmarker:
            return scan(landmarker)
    if face_landmarker is None:
        raise ValueError(
            "Provide face_landmarker or face_options to scan for a face."
        )
    return scan(face_landmarker)