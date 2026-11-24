from __future__ import annotations

import argparse
import importlib
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}
_PROCESSOR = None


def _load_processor():
    global _PROCESSOR
    if _PROCESSOR is None:
        project_root = str(PROJECT_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        try:
            _PROCESSOR = importlib.import_module(
                "face_detection.process_video_optimized"
            )
        except ModuleNotFoundError as error:
            if error.name == "mediapipe" or (
                error.name and error.name.startswith("mediapipe.")
            ):
                raise RuntimeError(
                    "MediaPipe is required to reconstruct the standard blur "
                    "polygons for HDF5 files that do not already contain them."
                ) from error
            raise
    return _PROCESSOR


def _find_process_regions_group(landmark_data: h5py.File):
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


def _has_valid_process_regions(
    landmark_data: h5py.File,
    frame_count: int,
) -> bool:
    group = _find_process_regions_group(landmark_data)
    if not isinstance(group, h5py.Group):
        return False

    required = (
        "frame_indices",
        "region_names",
        "region_roles",
        "stroke_width_px",
        "polygon_offsets",
        "polygon_points",
    )
    if any(name not in group for name in required):
        return False

    frame_indices = group["frame_indices"]
    if frame_indices.ndim != 1:
        return False
    polygon_offsets = group["polygon_offsets"][:]
    polygon_points = group["polygon_points"]
    region_count = frame_indices.shape[0]
    if (
        frame_indices.ndim != 1
        or region_count == 0
        or group["region_names"].shape != (region_count,)
        or group["region_roles"].shape != (region_count,)
        or group["stroke_width_px"].shape != (region_count,)
        or polygon_offsets.shape != (region_count + 1,)
        or polygon_points.ndim != 2
        or polygon_points.shape[1] != 2
        or polygon_offsets[0] != 0
        or polygon_offsets[-1] != len(polygon_points)
        or np.any(polygon_offsets[1:] < polygon_offsets[:-1])
    ):
        return False

    saved_frames = frame_indices[:]
    return bool(
        np.all(saved_frames >= 0)
        and np.all(saved_frames < frame_count)
    )


def _find_landmarks_file(
    video_path: Path,
    video_count_in_folder: int,
) -> Path:
    candidates = sorted(video_path.parent.glob("*_landmarks.h5"))
    if not candidates:
        raise FileNotFoundError(
            f"No *_landmarks.h5 file was found beside the video: {video_path.parent}"
        )
    metadata_matches = []
    unidentified_candidates = []
    for candidate in candidates:
        with h5py.File(candidate, "r") as landmark_data:
            h5_video_name = landmark_data.attrs.get("video_filename")
            if isinstance(h5_video_name, bytes):
                h5_video_name = h5_video_name.decode("utf-8")
            if h5_video_name == video_path.name:
                metadata_matches.append(candidate)
            elif h5_video_name is None:
                unidentified_candidates.append(candidate)
    if len(metadata_matches) == 1:
        return metadata_matches[0]
    if len(metadata_matches) > 1:
        raise ValueError(
            f"Multiple HDF5 files identify {video_path.name}: "
            f"{', '.join(path.name for path in metadata_matches)}"
        )
    if (
        len(candidates) == 1
        and len(unidentified_candidates) == 1
        and video_count_in_folder == 1
    ):
        return unidentified_candidates[0]
    raise ValueError(
        "Could not safely match this video to an HDF5 landmarks file. "
        f"Candidates matching '*_landmarks.h5': "
        f"{', '.join(path.name for path in candidates)}"
    )


def _save_standard_regions(landmarks_path: Path) -> int:
    with h5py.File(landmarks_path, "r+") as landmark_data:
        if "landmarks" not in landmark_data:
            raise ValueError(
                f"The HDF5 file does not contain a 'landmarks' dataset: {landmarks_path}"
            )
        landmarks = landmark_data["landmarks"]
        if landmarks.ndim != 3 or landmarks.shape[1:] != (478, 3):
            raise ValueError(
                f"Expected landmarks with shape (frames, 478, 3): {landmarks_path}"
            )
        frame_count = len(landmarks)
        if (
            "frame_count" in landmark_data.attrs
            and int(landmark_data.attrs["frame_count"]) != frame_count
        ):
            raise ValueError(
                f"The HDF5 frame_count attribute does not match the landmarks "
                f"dataset: {landmarks_path}"
            )
        try:
            width = int(landmark_data.attrs["frame_width"])
            height = int(landmark_data.attrs["frame_height"])
        except KeyError as error:
            raise ValueError(
                f"The HDF5 file is missing frame dimensions: {landmarks_path}"
            ) from error
        if width <= 0 or height <= 0 or frame_count == 0:
            raise ValueError(
                f"The HDF5 file has invalid frame dimensions or no landmarks: "
                f"{landmarks_path}"
            )
        blur_regions = landmark_data.get("blur_regions")
        if blur_regions is not None and not isinstance(blur_regions, h5py.Group):
            raise ValueError(
                f"The HDF5 'blur_regions' entry is not a group: {landmarks_path}"
            )
        if _has_valid_process_regions(landmark_data, frame_count):
            process_regions = _find_process_regions_group(landmark_data)
            if process_regions.name != "/blur_regions/face_blur":
                region_count = len(process_regions["frame_indices"])
                created_blur_regions = blur_regions is None
                if created_blur_regions:
                    blur_regions = landmark_data.create_group("blur_regions")
                try:
                    if process_regions.name == "/process_video_blur_regions":
                        landmark_data.move(
                            "process_video_blur_regions",
                            "blur_regions/face_blur",
                        )
                    else:
                        blur_regions.move("process_video", "face_blur")
                    landmark_data.flush()
                except Exception:
                    if (
                        created_blur_regions
                        and "blur_regions" in landmark_data
                        and not len(landmark_data["blur_regions"])
                    ):
                        del landmark_data["blur_regions"]
                    raise
                return region_count
            return 0

        processor = _load_processor()
        blur_indices = processor.get_face_blur_indices()
        created_blur_regions = blur_regions is None
        if created_blur_regions:
            blur_regions = landmark_data.create_group("blur_regions")
        parent = blur_regions
        target_name = "face_blur"

        operation_id = uuid.uuid4().hex
        pending_name = f"_std_blur_pending_{operation_id}"
        backup_name = f"_std_blur_backup_{operation_id}"
        buffered_regions = []
        saved_region_count = 0
        try:
            pending = processor._create_process_blur_regions_group(
                parent,
                group_name=pending_name,
            )
            pending.attrs["reconstructed_from"] = "HDF5 landmarks dataset"
            pending.attrs["source_script"] = "save_std_blur_regions.py"
            for frame_index in range(frame_count):
                frame_landmarks = np.asarray(landmarks[frame_index], dtype=np.float64)
                xy = frame_landmarks[:, :2]
                if not np.all(np.isfinite(xy)):
                    raise ValueError(
                        f"Non-finite landmark coordinates in frame {frame_index}: "
                        f"{landmarks_path}"
                    )

                pixel_points = np.trunc(
                    xy * np.asarray([width, height], dtype=np.float64)
                ).astype(np.int32)
                geometry = processor.get_process_blur_region_geometry(
                    pixel_points,
                    width,
                    height,
                    blur_indices,
                )
                if geometry is not None:
                    buffered_regions.extend(
                        (
                            frame_index,
                            name,
                            polygon,
                            role,
                            stroke_width,
                        )
                        for name, role, polygon, stroke_width in geometry["regions"]
                    )
                if frame_index % 128 == 127:
                    processor._append_process_blur_regions(
                        pending,
                        buffered_regions,
                    )
                    saved_region_count += len(buffered_regions)
                    buffered_regions.clear()

            processor._append_process_blur_regions(pending, buffered_regions)
            saved_region_count += len(buffered_regions)
            if saved_region_count == 0:
                raise ValueError(
                    f"No standard blur polygons could be derived from the "
                    f"landmarks in {landmarks_path}"
                )
            landmark_data.flush()

            if target_name in parent:
                parent.move(target_name, backup_name)
            parent.move(pending_name, target_name)
            if backup_name in parent:
                del parent[backup_name]
            if "process_video" in parent:
                del parent["process_video"]
            if "process_video_blur_regions" in landmark_data:
                del landmark_data["process_video_blur_regions"]
            landmark_data.flush()
        except Exception:
            if target_name not in parent and backup_name in parent:
                parent.move(backup_name, target_name)
            if pending_name in parent:
                del parent[pending_name]
            if created_blur_regions and not len(parent):
                del landmark_data["blur_regions"]
            landmark_data.flush()
            raise

    return saved_region_count


def _video_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def _process_video(video_path: Path, video_count_in_folder: int) -> bool:
    if not video_path.suffix:
        raise ValueError(f"Expected a video file path, got: {video_path}")
    landmarks_path = _find_landmarks_file(video_path, video_count_in_folder)
    saved_count = _save_standard_regions(landmarks_path)
    if saved_count:
        print(
            f"Polígonos padrão gravados/organizados ({saved_count}) em "
            f"{landmarks_path}, no grupo blur_regions/face_blur."
        )
        return True
    print(
        f"Polígonos padrão já existem em {landmarks_path}, "
        "no grupo blur_regions/face_blur; sem alterações."
    )
    return False


def _process_dataset(dataset_path: Path) -> int:
    if not dataset_path.is_dir():
        print(f"A pasta do dataset não existe: {dataset_path}", file=sys.stderr)
        return 1

    videos = _video_files(dataset_path)
    if not videos:
        print(f"Nenhum vídeo encontrado em {dataset_path}", file=sys.stderr)
        return 1

    videos_by_folder = defaultdict(list)
    for video_path in videos:
        videos_by_folder[video_path.parent].append(video_path)

    errors = 0
    for video_path in videos:
        try:
            _process_video(
                video_path,
                len(videos_by_folder[video_path.parent]),
            )
        except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
            errors += 1
            print(f"Falha em {video_path}: {error}", file=sys.stderr)

    print(f"Processados {len(videos) - errors}/{len(videos)} vídeos.")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Save the standard process_video_optimized blur polygons in the "
            "matching HDF5 file, using only its stored landmarks (no video read)."
        )
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        help="Path to one video whose matching *_landmarks.h5 file should be updated.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Dataset folder; recursively process videos in its subfolders.",
    )
    args = parser.parse_args(argv)
    if (args.video is None) == (args.dataset is None):
        parser.error("Provide either a video path or --dataset, but not both.")

    if args.dataset is not None:
        return _process_dataset(args.dataset)

    try:
        _process_video(args.video, max(1, len(_video_files(args.video.parent))))
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"Falha: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())