"""Read saved face landmarks and visualize them on an anonymized video.

Run this script with a video path. By default, it looks for a matching
``<video_stem>_landmarks.h5`` file in the same folder and displays the
landmarks frame by frame. Use ``--output`` to save an annotated copy.
Inspect an HDF5 file with 
``python read_landmarks.py --inspect-h5 "path/to/2_1_landmarks.h5"`` 
to see its metadata, dataset shapes, and example field access.

Example:

    python read_landmarks.py "path/to/2_1.avi"

The HDF5 file contains:

* ``landmarks``: float32 array shaped ``(frames, 478, 3)``. Each row is a video
  frame; each point stores normalized ``x``, ``y``, and relative ``z``.
* ``face_detected``: boolean array marking frames with a fresh detection.
* ``source_frame``: zero-based frame where that row's landmarks were last
  detected. Missing detections reuse the previous valid landmark positions.
* File attributes with video name, dimensions, FPS, coordinate convention,
  landmark count, and total frame count.
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def print_h5_contents(landmarks_path, sample_frames=3, sample_landmarks=3):
    """Print HDF5 metadata, dataset shapes, and examples of reading each field."""
    landmarks_path = Path(landmarks_path)
    if not landmarks_path.is_file():
        raise FileNotFoundError(f"Landmark file not found: {landmarks_path}")

    with h5py.File(landmarks_path, "r") as landmark_data:
        print(f"HDF5 file: {landmarks_path}")
        print("\nFile attributes (metadata):")
        if landmark_data.attrs:
            for name, value in landmark_data.attrs.items():
                print(f"  {name} = {value}")
        else:
            print("  (no attributes)")

        print("\nDatasets:")
        for name, dataset in landmark_data.items():
            if not isinstance(dataset, h5py.Dataset):
                print(f"  {name}: group")
                continue
            print(
                f"  {name}: shape={dataset.shape}, dtype={dataset.dtype}"
            )
            if dataset.attrs:
                for attr_name, attr_value in dataset.attrs.items():
                    print(f"    attribute {attr_name} = {attr_value}")

        if "landmarks" not in landmark_data:
            raise ValueError("The file does not contain the 'landmarks' dataset.")

        landmarks = landmark_data["landmarks"]
        if landmarks.ndim != 3 or landmarks.shape[1:] != (478, 3):
            raise ValueError(
                "Expected 'landmarks' to have shape "
                f"(frames, 478, 3); got {landmarks.shape}."
            )

        frame_count = min(sample_frames, len(landmarks))
        landmark_count = min(sample_landmarks, landmarks.shape[1])
        print(
            "\nExample: x, y, z coordinates for the first "
            f"{landmark_count} landmarks in each of the first "
            f"{frame_count} frames:"
        )
        for frame_index in range(frame_count):
            print(
                f"  landmarks[{frame_index}, :{landmark_count}, :] = "
                f"{landmarks[frame_index, :landmark_count, :].tolist()}"
            )

        for name in ("face_detected", "source_frame"):
            if name in landmark_data:
                values = landmark_data[name][:sample_frames]
                print(f"\nExample: {name}[:{len(values)}] = {values.tolist()}")

        print(
            "\nAccess in Python:\n"
            "  with h5py.File(caminho_h5, 'r') as h5:\n"
            "      coordinates = h5['landmarks'][frame, landmark, :]\n"
            "      detected = h5['face_detected'][frame]\n"
            "      source_frame = h5['source_frame'][frame]\n"
        )


def draw_landmarks_on_video(
    video_path,
    landmarks_path=None,
    output_path=None,
    show=True,
    point_radius=1,
):
    """Draw the saved face landmarks over the corresponding video frames.

    Args:
        video_path: Path to the anonymized video.
        landmarks_path: Path to the matching HDF5 file. If omitted, the
            function looks for ``<video_stem>_landmarks.h5`` beside the video.
        output_path: Optional path where an annotated video will be written.
        show: If ``True``, show the annotated frames in an OpenCV window.
        point_radius: Radius in pixels for each landmark dot.

    Returns:
        The number of video frames processed.

    Raises:
        FileNotFoundError: If the video or landmarks file does not exist.
        RuntimeError: If the video or annotated output cannot be opened.
        ValueError: If the HDF5 data is invalid or does not match the video.
    """
    video_path = Path(video_path)
    if landmarks_path is None:
        landmarks_path = video_path.with_name(
            f"{video_path.stem}_landmarks.h5"
        )
    else:
        landmarks_path = Path(landmarks_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not landmarks_path.is_file():
        raise FileNotFoundError(f"Landmark file not found: {landmarks_path}")
    if output_path is not None:
        output_path = Path(output_path)

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    output_video = None
    processed_frames = 0
    stopped_early = False

    try:
        with h5py.File(landmarks_path, "r") as landmark_data:
            # HDF5 datasets are read by name. Row i in each dataset corresponds
            # to zero-based frame i of the anonymized video.
            landmarks = landmark_data["landmarks"]
            face_detected = landmark_data["face_detected"]
            source_frame = landmark_data["source_frame"]

            if landmarks.ndim != 3 or landmarks.shape[1:] != (478, 3):
                raise ValueError(
                    "Expected the 'landmarks' dataset to have shape "
                    f"(frames, 478, 3); got {landmarks.shape}."
                )
            if (
                len(face_detected) != len(landmarks)
                or len(source_frame) != len(landmarks)
            ):
                raise ValueError(
                    "The 'landmarks', 'face_detected', and 'source_frame' "
                    "datasets must have the same number of frames."
                )

            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = float(landmark_data.attrs.get("fps", 25.0))

            saved_width = int(landmark_data.attrs["frame_width"])
            saved_height = int(landmark_data.attrs["frame_height"])
            if (width, height) != (saved_width, saved_height):
                raise ValueError(
                    "Video dimensions do not match the HDF5 metadata: "
                    f"video is {width}x{height}, data is "
                    f"{saved_width}x{saved_height}."
                )

            if output_path is not None:
                codec = (
                    "XVID"
                    if output_path.suffix.lower() == ".avi"
                    else "mp4v"
                )
                output_video = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*codec),
                    fps,
                    (width, height),
                )
                if not output_video.isOpened():
                    raise RuntimeError(
                        f"Could not create annotated video: {output_path}"
                    )

            while True:
                success, frame = video.read()
                if not success:
                    break

                if processed_frames >= len(landmarks):
                    raise ValueError(
                        "The video contains more frames than the HDF5 "
                        "landmark data."
                    )

                # x and y are normalized to the frame size; z is relative
                # depth and is not needed to draw 2D points.
                frame_landmarks = landmarks[processed_frames]
                points = np.rint(
                    frame_landmarks[:, :2]
                    * np.array([width, height], dtype=np.float32)
                ).astype(np.int32)
                points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, height - 1)

                # Green means a fresh detection; orange means the last valid
                # detection was carried forward for this frame.
                color = (
                    (0, 255, 0)
                    if face_detected[processed_frames]
                    else (0, 165, 255)
                )
                for x, y in points:
                    cv2.circle(
                        frame,
                        (int(x), int(y)),
                        point_radius,
                        color,
                        -1,
                    )

                status = (
                    f"Frame {processed_frames + 1}/{len(landmarks)} | "
                )
                cv2.putText(
                    frame,
                    status,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )

                if output_video is not None:
                    output_video.write(frame)

                if show:
                    cv2.imshow("Saved face landmarks", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        processed_frames += 1
                        stopped_early = True
                        break

                processed_frames += 1

            if not stopped_early and processed_frames != len(landmarks):
                raise ValueError(
                    "The video and HDF5 landmark data have different frame "
                    f"counts: processed {processed_frames}, "
                    f"landmarks contain {len(landmarks)}."
                )
    finally:
        video.release()
        if output_video is not None:
            output_video.release()
        if show:
            cv2.destroyAllWindows()

    if output_path is not None:
        print(f"Annotated video saved to: {output_path}")
    return processed_frames


def main():
    """Inspect an HDF5 file or run the landmark video visualization."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect saved HDF5 face landmarks or draw them on a video."
        )
    )
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        help="Path to the anonymized video.",
    )
    parser.add_argument(
        "--inspect-h5",
        type=Path,
        help="Print HDF5 metadata, dataset shapes, and example field access.",
    )
    parser.add_argument(
        "--landmarks",
        type=Path,
        help=(
            "Path to the HDF5 landmarks file. Defaults to "
            "<video_stem>_landmarks.h5 beside the video."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save a copy of the video with landmarks drawn.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process the video without opening a display window.",
    )
    args = parser.parse_args()

    if args.inspect_h5 is not None:
        print_h5_contents(args.inspect_h5)
        return

    if args.video is None:
        parser.error("Provide a video path or use --inspect-h5.")
    if args.no_display and args.output is None:
        parser.error("--no-display requires --output so the result is saved.")

    frame_count = draw_landmarks_on_video(
        args.video,
        landmarks_path=args.landmarks,
        output_path=args.output,
        show=not args.no_display,
    )
    print(f"Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
