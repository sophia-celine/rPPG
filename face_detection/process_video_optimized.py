"""Calculate movement indices and anonymize videos using MediaPipe landmarks.

The script processes a single video or AVI videos organized in patient folders.
Movement results are written to a CSV spreadsheet, and each anonymized video
is saved next to the original using the standardized ID from ids.csv.
"""

import csv
import math
import sys
import urllib.request
from pathlib import Path

import cv2
import h5py
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MODEL CONFIGURATION
# ============================================================

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)

MODEL_DIR = Path("mediapipe_models")

FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"

IDS_CSV_PATH = Path(r"C:/Users/Sophia/Documents/rPPG_data/dataset/ids.csv")

MOVEMENT_INTERVAL_FRAMES = 1
MOVEMENT_THRESHOLD_EYE_DISTANCE_FACTOR = 0.1


def load_standardized_ids(csv_path):
    """Load the ID mapping from the spreadsheet for naming output videos.

    Args:
        csv_path: Path to the CSV containing the ``Data Id`` and
            ``Id do paciente`` columns.

    Returns:
        A dictionary mapping patient IDs, with periods converted to
        underscores, to their standardized IDs.

    Raises:
        FileNotFoundError: If the spreadsheet does not exist.
        ValueError: If columns are missing, IDs are empty, or patient IDs are
            duplicated.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Planilha de IDs não encontrada: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required_columns = {"Data Id", "Id do paciente"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "A planilha de IDs precisa conter as colunas "
                "'Data Id' e 'Id do paciente'."
            )

        standardized_ids = {}
        for row_number, row in enumerate(reader, start=2):
            standardized_id = (row["Data Id"] or "").strip()
            patient_id = (row["Id do paciente"] or "").strip().replace(".", "_")
            if not standardized_id or not patient_id:
                raise ValueError(
                    f"ID vazio na linha {row_number} da planilha: {csv_path}"
                )
            if patient_id in standardized_ids:
                raise ValueError(
                    f"ID de paciente duplicado na planilha: {patient_id}"
                )
            standardized_ids[patient_id] = standardized_id

    return standardized_ids


def get_face_blur_indices():
    """Get landmark indices used to construct the blur mask.

    Returns:
        A dictionary of landmark indices for the eyes, eyebrows, lips, and
        face outline, grouped by region.
    """
    def indices_from_connections(connections):
        """Extract unique landmark indices from MediaPipe connections.

        Args:
            connections: Connections between landmarks in a facial region.

        Returns:
            A list of landmark indices included in the connections.
        """
        indices = set()
        for connection in connections:
            indices.add(connection.start)
            indices.add(connection.end)
        return list(indices)

    return {
        "left_eye": indices_from_connections(
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
        ),
        "right_eye": indices_from_connections(
            vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
        ),
        "left_eyebrow": indices_from_connections(
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW
        ),
        "right_eyebrow": indices_from_connections(
            vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW
        ),
        "lips": indices_from_connections(
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
        ),
        "face_oval": indices_from_connections(
            vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
        ),
    }


def anonymize_frame(frame, face_landmarks, blur_indices):
    """Apply anonymization blur to facial regions in a frame.

    Args:
        frame: OpenCV BGR image to process.
        face_landmarks: MediaPipe landmarks for one or more faces in the frame.
        blur_indices: Landmark indices grouped by region, as returned by
            ``get_face_blur_indices``.

    Returns:
        The BGR frame with blur applied to the regions defined by the mask.
    """
    height, width = frame.shape[:2]

    for landmarks in face_landmarks:
        points = np.array(
            [
                [int(landmark.x * width), int(landmark.y * height)]
                for landmark in landmarks
            ],
            dtype=np.int32,
        )

        eye_distance = int(
            math.hypot(
                points[33][0] - points[263][0],
                points[33][1] - points[263][1],
            )
        )
        eye_distance = max(eye_distance, 1)

        face_oval_hull = cv2.convexHull(points[blur_indices["face_oval"]])
        blur_kernel_size = max((eye_distance // 2) * 2 + 1, 3)
        frame_blur_radius = blur_kernel_size
        mask_blur_radius = 7
        border_thickness = eye_distance * 3
        # Include the outline, mask feather, and image-blur support in the crop.
        margin = (
            (border_thickness + 1) // 2
            + mask_blur_radius
            + frame_blur_radius
            + 2
        )

        hull_x, hull_y, hull_width, hull_height = cv2.boundingRect(face_oval_hull)
        x0 = max(0, hull_x - margin)
        y0 = max(0, hull_y - margin)
        x1 = min(width, hull_x + hull_width + margin)
        y1 = min(height, hull_y + hull_height + margin)
        if x0 >= x1 or y0 >= y1:
            continue

        roi_width = x1 - x0
        roi_height = y1 - y0
        local_points = points - np.array([x0, y0], dtype=np.int32)
        local_face_oval_hull = cv2.convexHull(
            local_points[blur_indices["face_oval"]]
        )
        face_interior = np.zeros((roi_height, roi_width), dtype=np.uint8)
        cv2.fillConvexPoly(face_interior, local_face_oval_hull, 255)

        thick_border = np.zeros((roi_height, roi_width), dtype=np.uint8)
        cv2.polylines(
            thick_border,
            [local_face_oval_hull],
            isClosed=True,
            color=255,
            thickness=border_thickness,
        )
        outside_blur_ring = cv2.bitwise_and(
            thick_border,
            cv2.bitwise_not(face_interior),
        )

        features_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
        left_eye_brow_indices = (
            blur_indices["left_eye"] + blur_indices["left_eyebrow"]
        )
        right_eye_brow_indices = (
            blur_indices["right_eye"] + blur_indices["right_eyebrow"]
        )
        left_eye_brow_hull = cv2.convexHull(local_points[left_eye_brow_indices])
        right_eye_brow_hull = cv2.convexHull(local_points[right_eye_brow_indices])
        cv2.fillConvexPoly(features_mask, left_eye_brow_hull, 255)
        cv2.fillConvexPoly(features_mask, right_eye_brow_hull, 255)
        cv2.fillConvexPoly(
            features_mask,
            cv2.convexHull(local_points[blur_indices["lips"]]),
            255,
        )

        mouth_left = local_points[291]
        mouth_right = local_points[61]
        lower_face_trapezoid = np.array(
            [
                mouth_left,
                mouth_right,
                [width - 1 - x0, height - 1 - y0],
                [-x0, height - 1 - y0],
            ],
            dtype=np.int32,
        )
        keep_lower_face_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
        cv2.fillPoly(keep_lower_face_mask, [lower_face_trapezoid], 255)

        final_mask = cv2.bitwise_or(features_mask, outside_blur_ring)
        final_mask = cv2.bitwise_and(
            final_mask,
            cv2.bitwise_not(keep_lower_face_mask),
        )
        mask_blurred = cv2.GaussianBlur(final_mask, (15, 15), 0)
        alpha = mask_blurred.astype(np.float32) * (1.0 / 255.0)

        frame_roi = frame[y0:y1, x0:x1]
        blurred_frame = cv2.GaussianBlur(
            frame_roi,
            (blur_kernel_size * 2 + 1, blur_kernel_size * 2 + 1),
            0,
        )
        blended_roi = (
            frame_roi.astype(np.float32) * (1.0 - alpha[:, :, None])
            + blurred_frame.astype(np.float32) * alpha[:, :, None]
        )
        frame[y0:y1, x0:x1] = np.clip(blended_roi, 0, 255).astype(np.uint8)

    return frame


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def download_model(url, destination):
    """Download the MediaPipe model if it does not already exist.
def download_model(url, destination):
    Args:
        url: URL from which to download the model file.
        destination: Local path where the model should be saved.

    Returns:
        ``None``. The file is available at ``destination``; if the download
        fails, the process exits with an error code.
    """

    if destination.exists():
        print(f"Model already exists: {destination}")
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Downloading model:")
    print(url)

    try:
        urllib.request.urlretrieve(
            url,
            destination
        )

    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

    print(f"Model saved to: {destination}")


# ============================================================
# CONVERT NORMALIZED LANDMARK COORDINATES TO PIXELS
# ============================================================

def landmark_to_pixel(landmark, width, height):
    """Convert a landmark's normalized coordinates to pixels.

    Args:
        landmark: Landmark with normalized ``x`` and ``y`` coordinates.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        An integer ``(x, y)`` tuple clamped to the frame boundaries.
    """

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    return x, y


def eye_distance_pixels(landmarks, width, height):
    """Calculate the distance between the eye centers in pixels.

    Args:
        landmarks: Sequence of MediaPipe facial landmarks.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Euclidean distance between the left and right eye centers, in pixels.
    """

    left_eye_center = (
        (landmarks[33].x + landmarks[133].x) * width / 2,
        (landmarks[33].y + landmarks[133].y) * height / 2,
    )
    right_eye_center = (
        (landmarks[362].x + landmarks[263].x) * width / 2,
        (landmarks[362].y + landmarks[263].y) * height / 2,
    )

    return math.hypot(
        right_eye_center[0] - left_eye_center[0],
        right_eye_center[1] - left_eye_center[1],
    )


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_video(
    video_path,
    show: bool = False,
    standardized_id=None,
    ids_csv_path=IDS_CSV_PATH,
):
    """Calculate the movement index and save an anonymized copy of the video.

    Args:
        video_path: Path to the original video to process.
        show: If ``True``, display landmarks and metrics during processing.
        standardized_id: Standardized ID to use for the output video name. If
            ``None``, it is looked up using the video folder name and
            ``ids_csv_path``.
        ids_csv_path: Path to the CSV mapping patient IDs to standardized IDs.

    Returns:
        A tuple of ``(accumulated_movement_index, detection_rate)``. Returns
        ``None`` if no face is detected in the first frame. The anonymized
        video is saved in the same folder as the original, using the
        standardized ID.

    Raises:
        FileNotFoundError: If the ID CSV does not exist.
        ValueError: If the patient is not mapped or the output would overwrite
            the original video.
        RuntimeError: If the anonymized video cannot be created.
    """

    video_path = Path(video_path)

    if not video_path.exists():

        print(f"Video not found: {video_path}")

        sys.exit(1)

    if standardized_id is None:
        standardized_ids = load_standardized_ids(ids_csv_path)
        patient_id = video_path.parent.name.replace(".", "_")
        standardized_id = standardized_ids.get(patient_id)
        if standardized_id is None:
            raise ValueError(
                f"A pasta do vídeo '{video_path.parent.name}' não foi "
                f"encontrada na planilha de IDs: {ids_csv_path}"
            )

    output_video_path = video_path.with_name(
        f"{standardized_id}{video_path.suffix}"
    )
    output_landmarks_path = output_video_path.with_name(
        f"{output_video_path.stem}_landmarks.h5"
    )
    if output_video_path.resolve() == video_path.resolve():
        raise ValueError(
            "O nome padronizado do vídeo anonimizado não pode sobrescrever "
            f"o vídeo original: {video_path}"
        )

    # --------------------------------------------------------
    # Download models
    # --------------------------------------------------------

    download_model(
        FACE_MODEL_URL,
        FACE_MODEL_PATH
    )


    # --------------------------------------------------------
    # Configure the Face Landmarker
    # --------------------------------------------------------

    face_base_options = python.BaseOptions(
        model_asset_path=str(FACE_MODEL_PATH)
    )

    face_options = vision.FaceLandmarkerOptions(

        base_options=face_base_options,

        running_mode=vision.RunningMode.VIDEO,

        num_faces=1,

        min_face_detection_confidence=0.3,

        min_face_presence_confidence=0.3,

        min_tracking_confidence=0.3
    )

    blur_indices = get_face_blur_indices()

    # --------------------------------------------------------
    # Open the video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"Could not open video: {video_path}"
        )

        sys.exit(1)


    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0


    if show:
        print()
        print("Video information")
        print("--------------------")
        print(f"Arquivo:   {video_path}")
        print(f"Resolution: {width} x {height}")
        print(f"FPS:        {fps:.2f}")
        print()
        print("Press Q or ESC to exit.")
        print()


    # Frame number
    frame_number = 0
    video_writer = None
    landmarks_file = None
    landmark_datasets = None
    last_landmarks_source_frame = -1
    movement_frames = []
    reference_distances = []
    movement_percentages = []
    average_displacements = []
    average_displacements_reference_units = []
    movement_indices = []
    accumulated_movement_indices = []
    accumulated_movement_index = 0.0
    MOVEMENT_THRESHOLD_PIXELS = 0.0
    previous_landmarks = None
    previous_measurement_frame = None


    # ========================================================
    # Create the landmarkers
    # ========================================================

    with vision.FaceLandmarker.create_from_options(
        face_options
    ) as face_landmarker:

        # ====================================================
        # Process the video
        # ====================================================
        frames_with_face = 0
        frames_without_face = 0
        last_result = None
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
            total_frames_label = total_frames if total_frames > 0 else "?"
            print(
                f"\r{frame_number}/{total_frames_label} frames processed",
                end="",
                flush=True,
            )


            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # Create a MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )


            # ------------------------------------------------
            # Detect the face
            # ------------------------------------------------

            face_result = face_landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if frame_number == 1 and not face_result.face_landmarks:
                print("No face detected in the first frame.")
                cap.release()
                cv2.destroyAllWindows()
                return None

            if video_writer is None:
                codec = "I420" if video_path.suffix.lower() == ".avi" else "mp4v"
                print('codec:', codec)
                video_writer = cv2.VideoWriter(
                    str(output_video_path),
                    cv2.VideoWriter_fourcc(*codec),
                    fps,
                    (width, height),
                )
                if not video_writer.isOpened():
                    video_writer.release()
                    cap.release()
                    raise RuntimeError(
                        f"Could not create anonymized video: "
                        f"{output_video_path}"
                    )
                chunk_frames = max(1, min(total_frames or 64, 64))
                landmarks_file = h5py.File(output_landmarks_path, "w")
                landmarks_file.attrs["video_filename"] = output_video_path.name
                landmarks_file.attrs["frame_width"] = width
                landmarks_file.attrs["frame_height"] = height
                landmarks_file.attrs["fps"] = fps
                landmarks_file.attrs["coordinate_system"] = (
                    "MediaPipe normalized face coordinates"
                )
                landmarks_file.attrs["coordinate_order"] = "x, y, z"
                landmarks_file.attrs["z_definition"] = (
                    "Relative depth estimate; not a physical distance."
                )
                landmarks_file.attrs["landmark_count"] = 478
                landmarks_file.attrs["frame_indexing"] = (
                    "Zero-based; dataset row i matches video frame i"
                )
                landmark_datasets = {
                    "landmarks": landmarks_file.create_dataset(
                        "landmarks",
                        shape=(total_frames, 478, 3),
                        maxshape=(None, 478, 3),
                        dtype=np.float32,
                        chunks=(chunk_frames, 478, 3),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    ),
                    "face_detected": landmarks_file.create_dataset(
                        "face_detected",
                        shape=(total_frames,),
                        maxshape=(None,),
                        dtype=np.bool_,
                        chunks=(chunk_frames,),
                        compression="gzip",
                        compression_opts=4,
                    ),
                    "source_frame": landmarks_file.create_dataset(
                        "source_frame",
                        shape=(total_frames,),
                        maxshape=(None,),
                        dtype=np.int64,
                        chunks=(chunk_frames,),
                        compression="gzip",
                        compression_opts=4,
                    ),
                }
                landmark_datasets["landmarks"].attrs["description"] = (
                    "Normalized x, y, z coordinates for the first face in each "
                    "video frame. Values are carried forward when detection fails."
                )
                landmark_datasets["face_detected"].attrs["description"] = (
                    "True when this frame has a new face detection; False when "
                    "the previous valid landmarks are reused."
                )
                landmark_datasets["source_frame"].attrs["description"] = (
                    "Zero-based frame index where the landmarks in this row "
                    "were last detected."
                )

            blur_result = (
                face_result
                if face_result.face_landmarks
                else last_result
            )
            frame_index = frame_number - 1
            if face_result.face_landmarks:
                last_landmarks_source_frame = frame_index

            if blur_result is None or not blur_result.face_landmarks:
                raise RuntimeError(
                    "No current or previous face landmarks are available for "
                    f"frame {frame_index}."
                )

            frame_landmarks = np.asarray(
                [
                    [landmark.x, landmark.y, landmark.z]
                    for landmark in blur_result.face_landmarks[0]
                ],
                dtype=np.float32,
            )
            if frame_landmarks.shape != (478, 3):
                raise ValueError(
                    f"Expected 478 landmarks with x, y, z coordinates, got "
                    f"{frame_landmarks.shape} in frame {frame_index}."
                )

            if frame_index >= landmark_datasets["landmarks"].shape[0]:
                new_capacity = max(
                    frame_index + 1,
                    landmark_datasets["landmarks"].shape[0] * 2,
                    1,
                )
                for dataset in landmark_datasets.values():
                    dataset.resize(new_capacity, axis=0)

            landmark_datasets["landmarks"][frame_index] = frame_landmarks
            landmark_datasets["face_detected"][frame_index] = bool(
                face_result.face_landmarks
            )
            landmark_datasets["source_frame"][frame_index] = (
                last_landmarks_source_frame
            )

            anonymized_frame = frame.copy()
            if blur_result is not None and blur_result.face_landmarks:
                anonymized_frame = anonymize_frame(
                    anonymized_frame,
                    blur_result.face_landmarks,
                    blur_indices,
                )
            video_writer.write(anonymized_frame)

            # ================================================
            # CALCULATE LANDMARK MOVEMENT
            # ================================================

            if face_result.face_landmarks:
                current_landmarks = face_result.face_landmarks[0]
                current_frame_number = frame_number - 1

                if previous_landmarks is None:
                    previous_landmarks = current_landmarks
                    previous_measurement_frame = current_frame_number
                elif (
                    current_frame_number - previous_measurement_frame
                    >= MOVEMENT_INTERVAL_FRAMES
                ):
                    if len(previous_landmarks) == len(current_landmarks):
                        MOVEMENT_THRESHOLD_PIXELS = (
                            MOVEMENT_THRESHOLD_EYE_DISTANCE_FACTOR
                            * eye_distance_pixels(
                                current_landmarks,
                                width,
                                height
                            )
                        )
                        distances = [
                            math.hypot(
                                (current.x - previous.x) * width,
                                (current.y - previous.y) * height,
                            )
                            for previous, current in zip(
                                previous_landmarks,
                                current_landmarks
                            )
                        ]

                        moved_distances = [
                            distance
                            for distance in distances
                            if distance > MOVEMENT_THRESHOLD_PIXELS
                        ]
                        moved_count = len(moved_distances)
                        movement_percentage = (
                            moved_count / len(distances) * 100
                            if distances
                            else 0.0
                        )
                        moved_fraction = (
                            moved_count / len(distances)
                            if distances
                            else 0.0
                        )
                        average_displacement = (
                            sum(moved_distances) / len(moved_distances)
                            if moved_distances
                            else 0.0
                        )
                        average_displacement_reference_units = (
                            average_displacement / MOVEMENT_THRESHOLD_PIXELS
                            if MOVEMENT_THRESHOLD_PIXELS > 0
                            else 0.0
                        )
                        movement_index = (
                            moved_fraction * average_displacement_reference_units
                        )
                        accumulated_movement_index += movement_index

                        movement_frames.append(current_frame_number)
                        reference_distances.append(MOVEMENT_THRESHOLD_PIXELS)
                        movement_percentages.append(movement_percentage)
                        average_displacements.append(average_displacement)
                        average_displacements_reference_units.append(
                            average_displacement_reference_units
                        )
                        movement_indices.append(movement_index)
                        accumulated_movement_indices.append(
                            accumulated_movement_index
                        )

                    previous_landmarks = current_landmarks
                    previous_measurement_frame = current_frame_number


            # =================================================
            # DRAW FACE LANDMARKS
            # =================================================

            if face_result.face_landmarks:
                frames_with_face += 1
                last_result = face_result

                if show:
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
            else:
                frames_without_face += 1
                if show and last_result is not None:
                    face_result = last_result
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
            # DISPLAY
            # =================================================

            if show:
                if movement_percentages:
                    cv2.putText(
                        frame,
                        (
                            "Distancia de referencia: "
                            f"{reference_distances[-1]:.2f} px"
                        ),
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        (
                            "Landmarks movidos (> referencia): "
                            f"{movement_percentages[-1]:.1f}%"
                        ),
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Deslocamento medio: {average_displacements[-1]:.2f} px",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        (
                            "Deslocamento medio: "
                            f"{average_displacements_reference_units[-1]:.2f} "
                            "unidades de referencia"
                        ),
                        (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Indice de movimento: {movement_indices[-1]:.2f}",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        (
                            "Indice de movimento acumulado: "
                            f"{accumulated_movement_indices[-1]:.2f}"
                        ),
                        (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                else:
                    cv2.putText(
                        frame,
                        "Aguardando primeira medicao de movimento",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                cv2.imshow(
                    f"{Path(video_path).parent.name}",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

        if frame_number:
            print()

    # ========================================================
    # Clean up
    # ========================================================

    cap.release()
    if video_writer is not None:
        video_writer.release()
    if landmarks_file is not None:
        for dataset in landmark_datasets.values():
            dataset.resize(frame_number, axis=0)
        landmarks_file.attrs["frame_count"] = frame_number
        landmarks_file.close()

    if show:
        cv2.destroyAllWindows()
    print("Frames with face:", frames_with_face)
    print("Frames without face:", frames_without_face)
    print(
        "Accumulated movement index:",
        f"{accumulated_movement_index:.2f}"
    )
    print(f"Anonymized video saved to: {output_video_path}")
    print(f"Landmark data saved to: {output_landmarks_path}")
    total_frames = frames_with_face + frames_without_face
    detection_rate = frames_with_face / total_frames if total_frames else 0.0

    return accumulated_movement_index, detection_rate

def process_dataset(
    csv_path='',
    folder_path=None,
    dataset_csv_path='',
    show=False,
    ids_csv_path=IDS_CSV_PATH,
):
    """Process videos in patient folders and update the metrics spreadsheets.

    Args:
        csv_path: Path to the results spreadsheet to update.
        folder_path: Root folder containing patient folders, optionally
            organized into ``batchN`` subfolders.
        dataset_csv_path: Path to the dataset spreadsheet, including
            illumination and phototype data.
        show: If ``True``, display video previews during processing.
        ids_csv_path: Path to the CSV mapping patient folders to the
            standardized IDs used for anonymized videos.

    Returns:
        A tuple of ``(updated_spreadsheet_path, missing_patients_path)``. The
        second item is ``None`` when all results match columns in the main
        spreadsheet. Anonymized videos are saved alongside the original videos.

    Raises:
        FileNotFoundError: If a spreadsheet or the dataset folder does not
            exist.
        ValueError: If the spreadsheets, videos, or IDs are inconsistent.
    """
    csv_file = Path(csv_path)
    dataset_folder = Path(folder_path) if folder_path is not None else None
    dataset_csv_file = Path(dataset_csv_path) if dataset_csv_path else None

    if not csv_file.is_file():
        raise FileNotFoundError(f"Planilha CSV não encontrada: {csv_file}")

    if dataset_csv_file is None or not dataset_csv_file.is_file():
        raise FileNotFoundError(
            f"Planilha de dados do dataset não encontrada: {dataset_csv_path}"
        )

    if dataset_folder is None or not dataset_folder.is_dir():
        raise FileNotFoundError(
            f"Pasta do conjunto de dados não encontrada: {folder_path}"
        )

    standardized_ids = load_standardized_ids(ids_csv_path)

    with dataset_csv_file.open("r", newline="", encoding="utf-8-sig") as source:
        dataset_rows = list(csv.reader(source))

    if not dataset_rows:
        raise ValueError(f"A planilha do dataset está vazia: {dataset_csv_file}")

    dataset_headers = dataset_rows[0]
    if not dataset_headers:
        raise ValueError(
            f"A primeira linha da planilha do dataset não contém cabeçalhos."
        )

    try:
        illumination_column = dataset_headers.index("Iluminação (lux)")
    except ValueError as error:
        raise ValueError(
            "A planilha do dataset não contém a coluna 'Iluminação (lux)'."
        ) from error

    try:
        phototype_column = dataset_headers.index("Fototipo")
    except ValueError as error:
        raise ValueError(
            "A planilha do dataset não contém a coluna 'Fototipo'."
        ) from error

    patient_dataset_info = {}
    for row_number, dataset_row in enumerate(dataset_rows[1:], start=2):
        if not dataset_row or not dataset_row[0].strip():
            print("Skipping empty dataset row.")
            continue

        patient_name = dataset_row[0].strip().replace('.', '_')
        if patient_name in patient_dataset_info:
            raise ValueError(
                f"Paciente duplicado na planilha do dataset: {patient_name}"
            )
        if max(illumination_column, phototype_column) >= len(dataset_row):
            raise ValueError(
                f"A linha {row_number} da planilha do dataset não possui "
                "as colunas 'Iluminação (lux)' e 'Fototipo'."
            )
        patient_dataset_info[patient_name] = (
            dataset_row[illumination_column],
            dataset_row[phototype_column],
        )

    batch_folders = sorted(
        folder
        for folder in dataset_folder.iterdir()
        if folder.is_dir()
        and folder.name.lower().startswith("batch")
        and folder.name[5:].isdigit()
    )

    if batch_folders:
        patient_parent_folders = batch_folders
    else:
        patient_parent_folders = [dataset_folder]

    patient_folders = sorted(
        (
            patient_folder
            for parent_folder in patient_parent_folders
            for patient_folder in parent_folder.iterdir()
            if patient_folder.is_dir() #and patient_folder.name.isalnum()
        ),
        key=lambda folder: (folder.name, str(folder)),
    )
    print("Patient folders:", patient_folders)

    patient_indices = {}
    for patient_folder in patient_folders:
        patient_name = patient_folder.name

        if patient_name in patient_indices:
            raise ValueError(
                f"Nome de paciente duplicado entre pastas: {patient_name}"
            )

        avi_files = sorted(
            path
            for path in patient_folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".avi"
        )

        if len(avi_files) > 1:
            print(
                f"Patient {patient_name} skipped: "
                "multiple AVI files found."
            )
            continue

        if not avi_files:
            print(f"Patient {patient_name} skipped: no AVI file found.")
            continue
        standardized_id = standardized_ids.get(patient_name.replace(".", "_"))
        if standardized_id is None:
            raise ValueError(
                f"A pasta do paciente '{patient_name}' não foi encontrada "
                f"na planilha de IDs: {ids_csv_path}"
            )
        print(f"================= {patient_name} ==================")
        result = process_video(
            avi_files[0],
            show=show,
            standardized_id=standardized_id,
            ids_csv_path=ids_csv_path,
        )
        if result is not None:
            movement_index, detection_rate = result
            patient_indices[patient_name] = (movement_index, detection_rate)

    with csv_file.open("r", newline="", encoding="utf-8-sig") as source:
        rows = list(csv.reader(source))

    if not rows:
        raise ValueError(f"A planilha CSV está vazia: {csv_file}")

    headers = rows[0]
    if not headers:
        raise ValueError(f"A primeira linha da planilha não contém cabeçalhos.")

    patient_columns = {}
    for column_index, patient_name in enumerate(headers[1:], start=1):
        if patient_name in patient_columns:
            raise ValueError(
                f"Paciente duplicado nos cabeçalhos da planilha: {patient_name}"
            )
        if patient_name:
            patient_columns[patient_name] = column_index

    found_indices = {
        patient_name: patient_metrics
        for patient_name, patient_metrics in patient_indices.items()
        if patient_name in patient_columns
    }
    missing_indices = {
        patient_name: patient_metrics
        for patient_name, patient_metrics in patient_indices.items()
        if patient_name not in patient_columns
    }

    movement_row = [""] * len(headers)
    movement_row[0] = "movement_index"
    detection_rate_row = [""] * len(headers)
    detection_rate_row[0] = "detection_rate"
    illumination_row = [""] * len(headers)
    illumination_row[0] = "Iluminação (lux)"
    phototype_row = [""] * len(headers)
    phototype_row[0] = "Fototipo"
    for patient_name, (movement_index, detection_rate) in found_indices.items():
        movement_row[patient_columns[patient_name]] = movement_index
        detection_rate_row[patient_columns[patient_name]] = detection_rate
        illumination, phototype = patient_dataset_info.get(patient_name, ("", ""))
        illumination_row[patient_columns[patient_name]] = illumination
        phototype_row[patient_columns[patient_name]] = phototype
    rows.append(movement_row)
    rows.append(detection_rate_row)
    rows.append(illumination_row)
    rows.append(phototype_row)

    output_file = csv_file.with_name(f"{csv_file.stem}_mov{csv_file.suffix}")
    with output_file.open("w", newline="", encoding="utf-8") as output:
        csv.writer(output).writerows(rows)

    missing_file = None
    if missing_indices:
        missing_file = csv_file.with_name(
            f"{csv_file.stem}_mov_not_found{csv_file.suffix}"
        )
        with missing_file.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["metric", *missing_indices.keys()])
            writer.writerow([
                "movement_index",
                *(metrics[0] for metrics in missing_indices.values()),
            ])
            writer.writerow([
                "detection_rate",
                *(metrics[1] for metrics in missing_indices.values()),
            ])
            writer.writerow([
                "Iluminação (lux)",
                *(
                    patient_dataset_info.get(patient_name, ("", ""))[0]
                    for patient_name in missing_indices
                ),
            ])
            writer.writerow([
                "Fototipo",
                *(
                    patient_dataset_info.get(patient_name, ("", ""))[1]
                    for patient_name in missing_indices
                ),
            ])

    print(f"Movement metrics spreadsheet saved to: {output_file}")
    if missing_file is not None:
        print(f"Patients missing from the original spreadsheet saved to: {missing_file}")

    return output_file, missing_file


if __name__ == "__main__":
    (accumulated_movement_index, detection_rate) = process_video(
    r"A:\dataset_raw\367781_0\L05-22-10-2026-16-28.avi")
    print(f"Accumulated movement index: {accumulated_movement_index:.2f}")
    print(f"Detection rate: {detection_rate:.2f}")