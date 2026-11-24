"""Read saved face landmarks and visualize them on an anonymized video.

Run this script with a video path. By default, it looks for a matching
``<video_stem>_landmarks.h5`` file or, if absent, a unique ``*landmarks.h5``
file in the same folder. It displays the landmarks frame by frame. Use
``--output`` to save an annotated copy.
Inspect an HDF5 file with 
``python read_landmarks.py --inspect-h5 "path/to/2_1_landmarks.h5"`` 
to see its metadata, dataset shapes, and example field access.
Use ``--show-blur-regions`` to highlight the pixels covered by saved blur
polygons while playing the anonymized video.

Example:

    python read_landmarks.py "path/to/2_1.avi"

The HDF5 file contains:

* ``landmarks``: float32 array shaped ``(frames, 478, 3)``. Each row is a video
  frame; each point stores normalized ``x``, ``y``, and relative ``z``.
* ``face_detected``: boolean array marking frames with a fresh detection.
* ``source_frame``: zero-based frame where that row's landmarks were last
  detected. Missing detections reuse the previous valid landmark positions.
* ``blur_regions``: manual polygon vertices and inclusive frame ranges, plus
  ``face_blur`` with the standard per-frame face-blur polygons.
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
                if isinstance(dataset, h5py.Group):
                    for attr_name, attr_value in dataset.attrs.items():
                        print(f"    attribute {attr_name} = {attr_value}")
                    for child_name, child in dataset.items():
                        if isinstance(child, h5py.Dataset):
                            print(
                                f"    {child_name}: shape={child.shape}, "
                                f"dtype={child.dtype}"
                            )
                            if child_name == "frame_ranges":
                                print(
                                    "      example values = "
                                    f"{child[:3].tolist()}"
                                )
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


def _find_blur_regions_file(video_path, landmarks_path=None):
    """Find the HDF5 metadata file containing blur regions for a video."""
    video_path = Path(video_path)
    if landmarks_path is not None:
        path = Path(landmarks_path)
        if not path.is_file():
            raise FileNotFoundError(f"Landmark file not found: {path}")
        return path

    expected_path = video_path.with_name(f"{video_path.stem}_landmarks.h5")
    if expected_path.is_file():
        with h5py.File(expected_path, "r") as landmark_data:
            if (
                "blur_regions" in landmark_data
                or "process_video_blur_regions" in landmark_data
            ):
                return expected_path

    candidates = []
    for path in sorted(video_path.parent.glob("*_landmarks.h5")):
        with h5py.File(path, "r") as landmark_data:
            if (
                "blur_regions" in landmark_data
                or "process_video_blur_regions" in landmark_data
            ):
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "No landmark HDF5 file containing manual or standard blur polygons "
            "was found beside "
            f"the video: {video_path.parent}"
        )
    if len(candidates) > 1:
        raise ValueError(
            "Multiple landmark HDF5 files containing blur polygons were found; "
            f"specify one with --landmarks: {', '.join(str(path) for path in candidates)}"
        )
    return candidates[0]


def _find_face_blur_group(landmark_data):
    if "blur_regions" in landmark_data:
        blur_regions = landmark_data["blur_regions"]
        if isinstance(blur_regions, h5py.Group):
            if "face_blur" in blur_regions:
                return blur_regions["face_blur"]
            if "process_video" in blur_regions:
                return blur_regions["process_video"]
    if "process_video_blur_regions" in landmark_data:
        return landmark_data["process_video_blur_regions"]
    return None


def _decode_h5_text(value):
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


def show_blur_regions_on_video(
    video_path,
    landmarks_path=None,
    overlay_color=(0, 0, 255),
    overlay_alpha=0.4,
):
    """Play an anonymized video and tint every pixel covered by saved blur polygons.

    Manual polygons and the standard per-frame polygons are rasterized from the
    HDF5 metadata and highlighted while the video plays.

    Args:
        video_path: Path to the anonymized video to display.
        landmarks_path: Optional HDF5 metadata path. If omitted, the function
            looks for a matching ``<video_stem>_landmarks.h5`` file and then
            for a unique ``*_landmarks.h5`` file containing blur polygons.
        overlay_color: BGR color used to highlight blurred pixels.
        overlay_alpha: Overlay strength between 0 (transparent) and 1 (opaque).

    Returns:
        The number of video frames displayed.

    Raises:
        FileNotFoundError: If the video or matching HDF5 file does not exist.
        RuntimeError: If the video cannot be opened.
        ValueError: If the HDF5 blur-region data is missing or malformed.

    Press ``q`` or ``Esc`` in the video window to stop playback.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not 0 <= overlay_alpha <= 1:
        raise ValueError("overlay_alpha must be between 0 and 1.")
    if len(overlay_color) != 3 or any(
        not isinstance(channel, int) or not 0 <= channel <= 255
        for channel in overlay_color
    ):
        raise ValueError("overlay_color must contain three BGR values from 0 to 255.")

    metadata_path = _find_blur_regions_file(video_path, landmarks_path)
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        video.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    displayed_frames = 0
    stopped_early = False
    try:
        with h5py.File(metadata_path, "r") as landmark_data:
            regions = landmark_data.get("blur_regions")
            if regions is not None and not isinstance(regions, h5py.Group):
                raise ValueError("The HDF5 'blur_regions' entry must be a group.")
            face_blur = _find_face_blur_group(landmark_data)
            if face_blur is not None and not isinstance(face_blur, h5py.Group):
                raise ValueError("The standard blur-region entry must be a group.")
            if regions is None and face_blur is None:
                raise ValueError(
                    "The HDF5 file does not contain manual or standard blur polygons."
                )

            manual_names = ("frame_ranges", "polygon_offsets", "polygon_points")
            has_manual_fields = regions is not None and any(
                name in regions for name in manual_names
            )
            if has_manual_fields and any(name not in regions for name in manual_names):
                raise ValueError(
                    "The 'blur_regions' group must contain frame_ranges, "
                    "polygon_offsets, and polygon_points together."
                )
            if has_manual_fields:
                frame_ranges = regions["frame_ranges"][:]
                polygon_offsets = regions["polygon_offsets"][:]
                polygon_points = regions["polygon_points"][:]
                if (
                    frame_ranges.ndim != 2
                    or frame_ranges.shape[1] != 2
                    or not np.issubdtype(frame_ranges.dtype, np.integer)
                    or polygon_offsets.shape != (len(frame_ranges) + 1,)
                    or not np.issubdtype(polygon_offsets.dtype, np.integer)
                    or polygon_points.ndim != 2
                    or polygon_points.shape[1] != 2
                    or not np.issubdtype(polygon_points.dtype, np.integer)
                    or not len(polygon_offsets)
                    or polygon_offsets[0] != 0
                    or polygon_offsets[-1] != len(polygon_points)
                    or np.any(polygon_offsets[1:] < polygon_offsets[:-1])
                ):
                    raise ValueError(
                        "The HDF5 manual blur-region arrays have invalid shapes or offsets."
                    )
            else:
                frame_ranges = np.empty((0, 2), dtype=np.int64)
                polygon_offsets = np.asarray([0], dtype=np.uint64)
                polygon_points = np.empty((0, 2), dtype=np.int32)

            standard_frame_indices = np.empty((0,), dtype=np.int64)
            standard_region_names = np.empty((0,), dtype="S32")
            standard_region_roles = np.empty((0,), dtype="S8")
            standard_stroke_widths = np.empty((0,), dtype=np.int32)
            standard_polygon_offsets = np.asarray([0], dtype=np.uint64)
            standard_polygon_points = np.empty((0, 2), dtype=np.int32)
            standard_regions_by_frame = {}
            standard_count = 0
            if face_blur is not None:
                required_datasets = (
                    "frame_indices",
                    "region_names",
                    "region_roles",
                    "stroke_width_px",
                    "polygon_offsets",
                    "polygon_points",
                )
                if any(name not in face_blur for name in required_datasets):
                    raise ValueError(
                        "The standard blur group must contain frame_indices, "
                        "region_names, region_roles, stroke_width_px, "
                        "polygon_offsets, and polygon_points."
                    )
                standard_frame_indices = face_blur["frame_indices"][:]
                standard_region_names = face_blur["region_names"][:]
                standard_region_roles = face_blur["region_roles"][:]
                standard_stroke_widths = face_blur["stroke_width_px"][:]
                standard_polygon_offsets = face_blur["polygon_offsets"][:]
                standard_polygon_points = face_blur["polygon_points"][:]
                standard_count = len(standard_frame_indices)
                if (
                    standard_frame_indices.ndim != 1
                    or not np.issubdtype(standard_frame_indices.dtype, np.integer)
                    or standard_region_names.shape != (standard_count,)
                    or standard_region_roles.shape != (standard_count,)
                    or standard_stroke_widths.shape != (standard_count,)
                    or not np.issubdtype(standard_stroke_widths.dtype, np.integer)
                    or standard_polygon_offsets.shape != (standard_count + 1,)
                    or not np.issubdtype(standard_polygon_offsets.dtype, np.integer)
                    or standard_polygon_points.ndim != 2
                    or standard_polygon_points.shape[1] != 2
                    or not np.issubdtype(standard_polygon_points.dtype, np.integer)
                    or not len(standard_polygon_offsets)
                    or standard_polygon_offsets[0] != 0
                    or standard_polygon_offsets[-1] != len(standard_polygon_points)
                    or np.any(
                        standard_polygon_offsets[1:]
                        < standard_polygon_offsets[:-1]
                    )
                    or np.any(standard_stroke_widths < 0)
                ):
                    raise ValueError(
                        "The standard blur-region arrays have invalid shapes or offsets."
                    )
                for region_index, frame_index in enumerate(standard_frame_indices):
                    region_name = _decode_h5_text(standard_region_names[region_index])
                    region_role = _decode_h5_text(standard_region_roles[region_index])
                    if region_role not in ("include", "exclude"):
                        raise ValueError(
                            f"Standard blur polygon {region_index} has an invalid role."
                        )
                    start = int(standard_polygon_offsets[region_index])
                    end = int(standard_polygon_offsets[region_index + 1])
                    if end - start < 3:
                        raise ValueError(
                            f"Standard blur polygon {region_index} has fewer than 3 points."
                        )
                    standard_regions_by_frame.setdefault(
                        int(frame_index), []
                    ).append((region_index, region_name, region_role))
            if not len(frame_ranges) and not standard_count:
                raise ValueError("The HDF5 file contains no blur polygons to display.")

            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(video.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = float(landmark_data.attrs.get("fps", 25.0))
            if not np.isfinite(fps) or fps <= 0:
                fps = 25.0
            default_width = int(landmark_data.attrs.get("frame_width", -1))
            default_height = int(landmark_data.attrs.get("frame_height", -1))
            saved_width = (
                int(regions.attrs.get("frame_width", default_width))
                if regions is not None
                else default_width
            )
            saved_height = (
                int(regions.attrs.get("frame_height", default_height))
                if regions is not None
                else default_height
            )
            if (
                frame_count <= 0
                or saved_width <= 0
                or saved_height <= 0
                or video_width < saved_width
                or video_height < saved_height
                or video_width - saved_width > 1
                or video_height - saved_height > 1
            ):
                raise ValueError(
                    "The video dimensions do not match the blur-region metadata."
                )
            if len(frame_ranges) and (
                np.any(frame_ranges[:, 0] < 0)
                or np.any(frame_ranges[:, 1] < frame_ranges[:, 0])
                or np.any(frame_ranges[:, 1] >= frame_count)
            ):
                raise ValueError(
                    "A saved blur-region frame range is outside the video."
                )
            for region_index in range(len(frame_ranges)):
                start = int(polygon_offsets[region_index])
                end = int(polygon_offsets[region_index + 1])
                polygon = polygon_points[start:end]
                if (
                    len(polygon) < 3
                    or np.any(polygon[:, 0] < 0)
                    or np.any(polygon[:, 0] >= saved_width)
                    or np.any(polygon[:, 1] < 0)
                    or np.any(polygon[:, 1] >= saved_height)
                ):
                    raise ValueError(
                        f"Blur region {region_index} contains invalid polygon points."
                    )

            if len(standard_frame_indices) and (
                np.any(standard_frame_indices < 0)
                or np.any(standard_frame_indices >= frame_count)
            ):
                raise ValueError(
                    "A standard blur polygon frame index is outside the video."
                )

            regions_starting = {}
            regions_ending = {}
            for region_index, (start_frame, end_frame) in enumerate(frame_ranges):
                regions_starting.setdefault(int(start_frame), []).append(region_index)
                regions_ending.setdefault(int(end_frame) + 1, []).append(region_index)
            active_regions = set()

            while True:
                success, frame = video.read()
                if not success:
                    break
                pixel_mask = np.zeros((video_height, video_width), dtype=np.uint8)

                # The saved frame ranges are inclusive and use zero-based frame numbers.
                active_regions.update(regions_starting.get(displayed_frames, ()))
                active_regions.difference_update(regions_ending.get(displayed_frames, ()))
                for region_index in active_regions:
                    start = int(polygon_offsets[region_index])
                    end = int(polygon_offsets[region_index + 1])
                    polygon = np.round(polygon_points[start:end]).astype(np.int32)
                    cv2.fillPoly(pixel_mask, [polygon], 255)

                frame_standard_regions = standard_regions_by_frame.get(
                    displayed_frames,
                    (),
                )
                if frame_standard_regions:
                    feature_mask = np.zeros_like(pixel_mask)
                    face_border_mask = np.zeros_like(pixel_mask)
                    face_interior_mask = np.zeros_like(pixel_mask)
                    exclusion_mask = np.zeros_like(pixel_mask)
                    for region_index, region_name, region_role in frame_standard_regions:
                        start = int(standard_polygon_offsets[region_index])
                        end = int(standard_polygon_offsets[region_index + 1])
                        polygon = np.round(
                            standard_polygon_points[start:end]
                        ).astype(np.int32)
                        if region_role == "exclude":
                            cv2.fillPoly(exclusion_mask, [polygon], 255)
                        elif region_name == "outside_face_ring":
                            cv2.polylines(
                                face_border_mask,
                                [polygon],
                                True,
                                255,
                                max(
                                    1,
                                    int(standard_stroke_widths[region_index]),
                                ),
                            )
                            cv2.fillPoly(face_interior_mask, [polygon], 255)
                        else:
                            cv2.fillPoly(feature_mask, [polygon], 255)

                    face_ring_mask = cv2.bitwise_and(
                        face_border_mask,
                        cv2.bitwise_not(face_interior_mask),
                    )
                    standard_mask = cv2.bitwise_or(feature_mask, face_ring_mask)
                    standard_mask = cv2.bitwise_and(
                        standard_mask,
                        cv2.bitwise_not(exclusion_mask),
                    )
                    pixel_mask = cv2.bitwise_or(pixel_mask, standard_mask)

                selected_pixels = pixel_mask != 0
                if np.any(selected_pixels):
                    color = np.asarray(overlay_color, dtype=np.float32)
                    original = frame[selected_pixels].astype(np.float32)
                    frame[selected_pixels] = np.clip(
                        original * (1.0 - overlay_alpha) + color * overlay_alpha,
                        0,
                        255,
                    ).astype(np.uint8)

                cv2.putText(
                    frame,
                    f"Frame {displayed_frames + 1}/{frame_count} | "
                    f"Manual regions: {len(active_regions)} | "
                    f"Standard polygons: {len(frame_standard_regions)} | "
                    "Press Q to quit",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Blurred pixel regions", frame)
                key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
                displayed_frames += 1
                if key == ord("q") or key == 27:
                    stopped_early = True
                    break

            if not stopped_early and displayed_frames != frame_count:
                raise ValueError(
                    "The video and HDF5 blur-region metadata have different "
                    f"frame counts: displayed {displayed_frames}, expected {frame_count}."
                )
    finally:
        video.release()
        cv2.destroyAllWindows()

    return displayed_frames


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
            function looks for ``<video_stem>_landmarks.h5`` first, then a
            unique ``*landmarks.h5`` file beside the video.
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
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if landmarks_path is None:
        expected_path = video_path.with_name(f"{video_path.stem}_landmarks.h5")
        if expected_path.is_file():
            landmarks_path = expected_path
        else:
            candidates = sorted(video_path.parent.glob("*landmarks.h5"))
            if not candidates:
                raise FileNotFoundError(
                    "No landmark file ending in 'landmarks.h5' was found beside "
                    f"the video: {video_path.parent}"
                )
            if len(candidates) > 1:
                raise ValueError(
                    "Multiple landmark files ending in 'landmarks.h5' were found; "
                    f"specify one with --landmarks: {', '.join(map(str, candidates))}"
                )
            landmarks_path = candidates[0]
    else:
        landmarks_path = Path(landmarks_path)

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
    """Inspect HDF5 data or visualize saved landmarks or blurred regions."""
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
        "--show-blur-regions",
        action="store_true",
        help=(
            "Play the video and highlight manual and standard saved blur regions."
        ),
    )
    parser.add_argument(
        "--landmarks",
        type=Path,
        help=(
            "Path to the HDF5 landmarks file. Defaults to the matching "
            "<video_stem>_landmarks.h5 or a unique *landmarks.h5 file beside "
            "the video."
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
        if args.show_blur_regions:
            parser.error("--inspect-h5 cannot be combined with --show-blur-regions.")
        print_h5_contents(args.inspect_h5)
        return

    if args.video is None:
        parser.error("Provide a video path or use --inspect-h5.")
    if args.show_blur_regions:
        if args.output is not None or args.no_display:
            parser.error(
                "--show-blur-regions displays the video and cannot be combined "
                "with --output or --no-display."
            )
        frame_count = show_blur_regions_on_video(
            args.video,
            landmarks_path=args.landmarks,
        )
        print(f"Displayed {frame_count} frames.")
        return
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
