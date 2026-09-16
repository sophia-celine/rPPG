import os
import cv2
import json
import random
import argparse
import itertools
import numpy as np
import pandas as pd

from deepface import DeepFace


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = [
    "ArcFace",
    "Facenet512",
    "SFace",
]

DISTANCE_METRIC = "cosine"
DETECTOR_BACKEND = "retinaface"

NUM_FRAMES = 10


# ============================================================
# VIDEO FUNCTIONS
# ============================================================

def get_video_info(video_path):
    """
    Return basic information about a video.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def extract_frames(video_path, frame_indices):
    """
    Extract selected frames from a video.

    Returns:
        dict:
            frame_index -> frame (BGR numpy array)
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = {}

    for frame_index in sorted(frame_indices):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = cap.read()

        if success:
            frames[frame_index] = frame
        else:
            print(
                f"WARNING: Could not read frame "
                f"{frame_index} from {video_path}"
            )

    cap.release()

    return frames


def save_frames(frames, output_dir, prefix):
    """
    Save extracted frames as PNG files.
    """

    os.makedirs(output_dir, exist_ok=True)

    for frame_index, frame in frames.items():

        filename = os.path.join(
            output_dir,
            f"{prefix}_frame_{frame_index:06d}.png"
        )

        cv2.imwrite(filename, frame)


# ============================================================
# FACE VERIFICATION
# ============================================================

def verify_faces(
    img1,
    img2,
    model_name,
    detector_backend=DETECTOR_BACKEND,
    distance_metric=DISTANCE_METRIC,
):
    """
    Compare two images using DeepFace.

    Returns a dictionary containing:
        result
        distance
        threshold
        confidence
        error
    """

    try:

        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=True,
            align=True,
        )

        distance = result.get("distance", np.nan)
        threshold = result.get("threshold", np.nan)
        confidence = result.get("confidence", np.nan)

        verified = result.get("verified", None)

        if verified is True:
            classification = "SAME_PERSON"

        elif verified is False:
            classification = "NOT_SAME_PERSON"

        else:
            classification = "UNKNOWN"

        return {
            "result": classification,
            "distance": distance,
            "threshold": threshold,
            "confidence": confidence,
            "error": "",
        }

    except Exception as e:

        return {
            "result": "FACE_NOT_DETECTED_OR_ERROR",
            "distance": np.nan,
            "threshold": np.nan,
            "confidence": np.nan,
            "error": str(e),
        }


# ============================================================
# TEMPORARY IMAGE MANAGEMENT
# ============================================================

def create_frame_files(frames, directory, prefix):
    """
    Save frames to disk because DeepFace.verify accepts
    image paths.
    """

    os.makedirs(directory, exist_ok=True)

    paths = {}

    for frame_index, frame in frames.items():

        path = os.path.join(
            directory,
            f"{prefix}_{frame_index:06d}.png"
        )

        cv2.imwrite(path, frame)

        paths[frame_index] = path

    return paths


# ============================================================
# EXPERIMENT 1
# SAME PATIENT - ORIGINAL VS ORIGINAL
# ============================================================

def experiment_1(original_paths, frame_indices, model_name):
    """
    Compare every pair of the 10 original frames.

    Number of comparisons:
        C(10, 2) = 45
    """

    results = []

    for frame_a, frame_b in itertools.combinations(
        frame_indices, 2
    ):

        verification = verify_faces(
            original_paths[frame_a],
            original_paths[frame_b],
            model_name,
        )

        results.append({
            "experiment": 1,
            "comparison_type": "same_patient_original_original",
            "model": model_name,
            "frame_a": frame_a,
            "frame_b": frame_b,
            "result": verification["result"],
            "distance": verification["distance"],
            "threshold": verification["threshold"],
            "confidence": verification["confidence"],
            "error": verification["error"],
        })

    return results


# ============================================================
# EXPERIMENT 2
# SAME FRAME - ORIGINAL VS ANONYMIZED
# ============================================================

def experiment_2(
    original_paths,
    anonymized_paths,
    frame_indices,
    model_name,
):
    """
    Compare original frame i with anonymized frame i.

    Number of comparisons:
        10
    """

    results = []

    for frame_index in frame_indices:

        verification = verify_faces(
            original_paths[frame_index],
            anonymized_paths[frame_index],
            model_name,
        )

        results.append({
            "experiment": 2,
            "comparison_type": "same_frame_original_anonymized",
            "model": model_name,
            "frame_a": frame_index,
            "frame_b": frame_index,
            "result": verification["result"],
            "distance": verification["distance"],
            "threshold": verification["threshold"],
            "confidence": verification["confidence"],
            "error": verification["error"],
        })

    return results


# ============================================================
# EXPERIMENT 3
# ORIGINAL VS ALL ANONYMIZED FRAMES
# ============================================================

def experiment_3(
    original_paths,
    anonymized_paths,
    frame_indices,
    model_name,
):
    """
    Compare every original frame against every anonymized frame.

    Number of comparisons:
        10 x 10 = 100
    """

    results = []

    for frame_a in frame_indices:

        for frame_b in frame_indices:

            verification = verify_faces(
                original_paths[frame_a],
                anonymized_paths[frame_b],
                model_name,
            )

            results.append({
                "experiment": 3,
                "comparison_type": "all_original_vs_all_anonymized",
                "model": model_name,
                "frame_a": frame_a,
                "frame_b": frame_b,
                "result": verification["result"],
                "distance": verification["distance"],
                "threshold": verification["threshold"],
                "confidence": verification["confidence"],
                "error": verification["error"],
            })

    return results


# ============================================================
# EXPERIMENT 4
# DIFFERENT PATIENTS - ORIGINAL VS ORIGINAL
# ============================================================

def experiment_4(
    patient_a_paths,
    patient_b_paths,
    patient_a_indices,
    patient_b_indices,
    model_name,
):
    """
    Compare frames from two DIFFERENT patients.

    Every frame from patient A is compared against every
    frame from patient B.

    Number of comparisons:
        10 x 10 = 100

    This is the negative-control experiment.
    """

    results = []

    for frame_a in patient_a_indices:

        for frame_b in patient_b_indices:

            verification = verify_faces(
                patient_a_paths[frame_a],
                patient_b_paths[frame_b],
                model_name,
            )

            results.append({
                "experiment": 4,
                "comparison_type": "different_patient_original_original",
                "model": model_name,
                "frame_a": frame_a,
                "frame_b": frame_b,
                "result": verification["result"],
                "distance": verification["distance"],
                "threshold": verification["threshold"],
                "confidence": verification["confidence"],
                "error": verification["error"],
            })

    return results


# ============================================================
# SUMMARY
# ============================================================

def summarize_results(results):
    """
    Generate summary statistics for each experiment/model.
    """

    df = pd.DataFrame(results)

    summary_rows = []

    for (experiment, model), group in df.groupby(
        ["experiment", "model"]
    ):

        total = len(group)

        valid = group[
            group["result"].isin(
                ["SAME_PERSON", "NOT_SAME_PERSON"]
            )
        ]

        failed = group[
            group["result"] == "FACE_NOT_DETECTED_OR_ERROR"
        ]

        same = group[
            group["result"] == "SAME_PERSON"
        ]

        different = group[
            group["result"] == "NOT_SAME_PERSON"
        ]

        distances = valid["distance"].dropna()

        summary_rows.append({

            "experiment": experiment,

            "model": model,

            "total_comparisons": total,

            "valid_comparisons": len(valid),

            "failed_or_no_face": len(failed),

            "same_person": len(same),

            "not_same_person": len(different),

            "same_person_percentage_valid":
                (
                    len(same) / len(valid) * 100
                    if len(valid) > 0
                    else np.nan
                ),

            "not_same_person_percentage_valid":
                (
                    len(different) / len(valid) * 100
                    if len(valid) > 0
                    else np.nan
                ),

            "mean_distance":
                distances.mean()
                if len(distances) > 0
                else np.nan,

            "median_distance":
                distances.median()
                if len(distances) > 0
                else np.nan,

            "min_distance":
                distances.min()
                if len(distances) > 0
                else np.nan,

            "max_distance":
                distances.max()
                if len(distances) > 0
                else np.nan,
        })

    return pd.DataFrame(summary_rows)


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(summary_df):

    print("\n")
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for _, row in summary_df.iterrows():

        print(
            f"\nExperiment {int(row['experiment'])} | "
            f"{row['model']}"
        )

        print(
            f"  Total comparisons:       "
            f"{int(row['total_comparisons'])}"
        )

        print(
            f"  Valid comparisons:       "
            f"{int(row['valid_comparisons'])}"
        )

        print(
            f"  Face/error failures:     "
            f"{int(row['failed_or_no_face'])}"
        )

        print(
            f"  SAME_PERSON:             "
            f"{int(row['same_person'])}"
        )

        print(
            f"  NOT_SAME_PERSON:         "
            f"{int(row['not_same_person'])}"
        )

        if not np.isnan(row["same_person_percentage_valid"]):

            print(
                f"  Same person (% valid):   "
                f"{row['same_person_percentage_valid']:.2f}%"
            )

            print(
                f"  Different (% valid):     "
                f"{row['not_same_person_percentage_valid']:.2f}%"
            )

        if not np.isnan(row["mean_distance"]):

            print(
                f"  Mean distance:           "
                f"{row['mean_distance']:.4f}"
            )

            print(
                f"  Median distance:         "
                f"{row['median_distance']:.4f}"
            )

            print(
                f"  Min distance:            "
                f"{row['min_distance']:.4f}"
            )

            print(
                f"  Max distance:            "
                f"{row['max_distance']:.4f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Face-recognition evaluation of video "
            "anonymization."
        )
    )

    parser.add_argument(
        "--original",
        required=True,
        help="Original video of patient A",
    )

    parser.add_argument(
        "--anonymized",
        required=True,
        help="Anonymized video of patient A",
    )

    parser.add_argument(
        "--different-patient",
        required=True,
        help=(
            "Original video of a DIFFERENT patient "
            "(patient B)"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for frame selection",
    )

    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save sampled frames to output directory",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(args.output, exist_ok=True)

    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    print("\nReading video information...")

    original_info = get_video_info(args.original)
    anonymized_info = get_video_info(args.anonymized)
    patient_b_info = get_video_info(args.different_patient)

    print("\nPatient A - Original:")
    print(original_info)

    print("\nPatient A - Anonymized:")
    print(anonymized_info)

    print("\nPatient B - Different patient:")
    print(patient_b_info)

    # --------------------------------------------------------
    # CHECK ORIGINAL / ANONYMIZED VIDEO
    # --------------------------------------------------------

    if (
        original_info["frame_count"]
        != anonymized_info["frame_count"]
    ):

        raise ValueError(
            "Original and anonymized videos do not have "
            "the same number of frames."
        )

    # --------------------------------------------------------
    # RANDOM FRAME SELECTION
    # --------------------------------------------------------

    random.seed(args.seed)

    original_frame_count = original_info["frame_count"]
    patient_b_frame_count = patient_b_info["frame_count"]

    if original_frame_count < NUM_FRAMES:
        raise ValueError(
            f"Patient A video has fewer than "
            f"{NUM_FRAMES} frames."
        )

    if patient_b_frame_count < NUM_FRAMES:
        raise ValueError(
            f"Patient B video has fewer than "
            f"{NUM_FRAMES} frames."
        )

    # Same 10 indices for original and anonymized Patient A
    patient_a_indices = sorted(
        random.sample(
            range(original_frame_count),
            NUM_FRAMES,
        )
    )

    # Independent random 10 indices for Patient B
    patient_b_indices = sorted(
        random.sample(
            range(patient_b_frame_count),
            NUM_FRAMES,
        )
    )

    print("\n")
    print("=" * 100)
    print("SELECTED FRAMES")
    print("=" * 100)

    print(
        f"\nPatient A frame indices:\n"
        f"{patient_a_indices}"
    )

    print(
        f"\nPatient B frame indices:\n"
        f"{patient_b_indices}"
    )

    # --------------------------------------------------------
    # EXTRACT FRAMES
    # --------------------------------------------------------

    print("\nExtracting Patient A original frames...")

    original_frames = extract_frames(
        args.original,
        patient_a_indices,
    )

    print("Extracting Patient A anonymized frames...")

    anonymized_frames = extract_frames(
        args.anonymized,
        patient_a_indices,
    )

    print("Extracting Patient B original frames...")

    patient_b_frames = extract_frames(
        args.different_patient,
        patient_b_indices,
    )

    # Check whether all frames were successfully extracted
    if len(original_frames) != NUM_FRAMES:
        raise RuntimeError(
            "Not all Patient A original frames "
            "could be extracted."
        )

    if len(anonymized_frames) != NUM_FRAMES:
        raise RuntimeError(
            "Not all Patient A anonymized frames "
            "could be extracted."
        )

    if len(patient_b_frames) != NUM_FRAMES:
        raise RuntimeError(
            "Not all Patient B frames "
            "could be extracted."
        )

    # --------------------------------------------------------
    # SAVE OPTIONAL FRAME COPIES
    # --------------------------------------------------------

    if args.save_frames:

        frames_dir = os.path.join(
            args.output,
            "sampled_frames",
        )

        save_frames(
            original_frames,
            frames_dir,
            "patient_A_original",
        )

        save_frames(
            anonymized_frames,
            frames_dir,
            "patient_A_anonymized",
        )

        save_frames(
            patient_b_frames,
            frames_dir,
            "patient_B_original",
        )

    # --------------------------------------------------------
    # CREATE TEMPORARY IMAGE FILES FOR DEEPFACE
    # --------------------------------------------------------

    temp_dir = os.path.join(
        args.output,
        "temporary_frames",
    )

    original_paths = create_frame_files(
        original_frames,
        temp_dir,
        "patient_A_original",
    )

    anonymized_paths = create_frame_files(
        anonymized_frames,
        temp_dir,
        "patient_A_anonymized",
    )

    patient_b_paths = create_frame_files(
        patient_b_frames,
        temp_dir,
        "patient_B_original",
    )

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    all_results = []

    for model_name in MODELS:

        print("\n")
        print("=" * 100)
        print(f"MODEL: {model_name}")
        print("=" * 100)

        # ----------------------------------------------------
        # EXPERIMENT 1
        # ----------------------------------------------------

        print(
            "\nExperiment 1: "
            "same patient - original vs original"
        )

        results_1 = experiment_1(
            original_paths,
            patient_a_indices,
            model_name,
        )

        all_results.extend(results_1)

        # ----------------------------------------------------
        # EXPERIMENT 2
        # ----------------------------------------------------

        print(
            "\nExperiment 2: "
            "original vs anonymized - same frame"
        )

        results_2 = experiment_2(
            original_paths,
            anonymized_paths,
            patient_a_indices,
            model_name,
        )

        all_results.extend(results_2)

        # ----------------------------------------------------
        # EXPERIMENT 3
        # ----------------------------------------------------

        print(
            "\nExperiment 3: "
            "all original vs all anonymized"
        )

        results_3 = experiment_3(
            original_paths,
            anonymized_paths,
            patient_a_indices,
            model_name,
        )

        all_results.extend(results_3)

        # ----------------------------------------------------
        # EXPERIMENT 4
        # ----------------------------------------------------

        print(
            "\nExperiment 4: "
            "different patient - original vs original"
        )

        results_4 = experiment_4(
            original_paths,
            patient_b_paths,
            patient_a_indices,
            patient_b_indices,
            model_name,
        )

        all_results.extend(results_4)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    detailed_csv = os.path.join(
        args.output,
        "detailed_results.csv",
    )

    results_df.to_csv(
        detailed_csv,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_df = summarize_results(
        all_results
    )

    summary_csv = os.path.join(
        args.output,
        "summary_results.csv",
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    json_path = os.path.join(
        args.output,
        "results.json",
    )

    json_data = {
        "configuration": {
            "models": MODELS,
            "distance_metric": DISTANCE_METRIC,
            "detector_backend": DETECTOR_BACKEND,
            "num_frames": NUM_FRAMES,
            "seed": args.seed,
        },

        "videos": {
            "patient_A_original": args.original,
            "patient_A_anonymized": args.anonymized,
            "patient_B_original": args.different_patient,
        },

        "video_information": {
            "patient_A_original": original_info,
            "patient_A_anonymized": anonymized_info,
            "patient_B_original": patient_b_info,
        },

        "selected_frames": {
            "patient_A": patient_a_indices,
            "patient_B": patient_b_indices,
        },

        "detailed_results": all_results,

        "summary": summary_df.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_data,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print_summary(summary_df)

    # --------------------------------------------------------
    # FILE LOCATIONS
    # --------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("OUTPUT FILES")
    print("=" * 100)

    print(
        f"\nDetailed results:\n"
        f"{detailed_csv}"
    )

    print(
        f"\nSummary:\n"
        f"{summary_csv}"
    )

    print(
        f"\nComplete JSON:\n"
        f"{json_path}"
    )

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()