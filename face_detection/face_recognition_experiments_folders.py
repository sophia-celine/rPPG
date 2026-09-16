import argparse
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from deepface import DeepFace


MODELS = [
    "ArcFace",
    "Facenet512",
    "SFace",
]

DISTANCE_METRIC = "cosine"
DETECTOR_BACKEND = "retinaface"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_files(folder_path):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = [
        str(path)
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise ValueError(f"No image files were found in folder: {folder_path}")

    return files


def verify_faces(img1, img2, model_name):
    try:
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model_name,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
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
    except Exception as exc:
        return {
            "result": "FACE_NOT_DETECTED_OR_ERROR",
            "distance": np.nan,
            "threshold": np.nan,
            "confidence": np.nan,
            "error": str(exc),
        }


def experiment_same_folder(folder_name, image_paths, model_name):
    results = []

    for img_a, img_b in combinations(image_paths, 2):
        verification = verify_faces(img_a, img_b, model_name)

        results.append(
            {
                "experiment": 1,
                "experiment_label": "same_folder_comparisons",
                "folder": folder_name,
                "model": model_name,
                "image_a": os.path.basename(img_a),
                "image_b": os.path.basename(img_b),
                "image_a_path": img_a,
                "image_b_path": img_b,
                "result": verification["result"],
                "distance": verification["distance"],
                "threshold": verification["threshold"],
                "confidence": verification["confidence"],
                "error": verification["error"],
            }
        )

    return results


def experiment_cross_folder(folder_a_paths, folder_b_paths, model_name):
    results = []

    for idx_a, img_a in enumerate(folder_a_paths):
        for idx_b, img_b in enumerate(folder_b_paths):
            verification = verify_faces(img_a, img_b, model_name)

            results.append(
                {
                    "experiment": 2,
                    "experiment_label": "cross_folder_comparisons",
                    "folder_a": os.path.basename(os.path.dirname(img_a)),
                    "folder_b": os.path.basename(os.path.dirname(img_b)),
                    "model": model_name,
                    "image_a_index": idx_a,
                    "image_b_index": idx_b,
                    "image_a": os.path.basename(img_a),
                    "image_b": os.path.basename(img_b),
                    "image_a_path": img_a,
                    "image_b_path": img_b,
                    "result": verification["result"],
                    "distance": verification["distance"],
                    "threshold": verification["threshold"],
                    "confidence": verification["confidence"],
                    "error": verification["error"],
                }
            )

    return results


def summarize_results(results):
    df = pd.DataFrame(results)
    summary_rows = []

    for (experiment, model), group in df.groupby(["experiment", "model"]):
        total = len(group)
        valid = group[group["result"].isin(["SAME_PERSON", "NOT_SAME_PERSON"])]
        failed = group[group["result"] == "FACE_NOT_DETECTED_OR_ERROR"]
        same = group[group["result"] == "SAME_PERSON"]
        different = group[group["result"] == "NOT_SAME_PERSON"]
        distances = valid["distance"].dropna()

        summary_rows.append(
            {
                "experiment": experiment,
                "model": model,
                "total_comparisons": total,
                "valid_comparisons": len(valid),
                "failed_or_no_face": len(failed),
                "same_person": len(same),
                "not_same_person": len(different),
                "same_person_percentage_valid": len(same) / len(valid) * 100 if len(valid) > 0 else np.nan,
                "not_same_person_percentage_valid": len(different) / len(valid) * 100 if len(valid) > 0 else np.nan,
                "mean_distance": distances.mean() if len(distances) > 0 else np.nan,
                "median_distance": distances.median() if len(distances) > 0 else np.nan,
                "min_distance": distances.min() if len(distances) > 0 else np.nan,
                "max_distance": distances.max() if len(distances) > 0 else np.nan,
            }
        )

    return pd.DataFrame(summary_rows)


def print_summary(summary_df):
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for _, row in summary_df.iterrows():
        print(f"\nExperiment {int(row['experiment'])} | {row['model']}")
        print(f"  Total comparisons:       {int(row['total_comparisons'])}")
        print(f"  Valid comparisons:       {int(row['valid_comparisons'])}")
        print(f"  Face/error failures:     {int(row['failed_or_no_face'])}")
        print(f"  SAME_PERSON:             {int(row['same_person'])}")
        print(f"  NOT_SAME_PERSON:         {int(row['not_same_person'])}")

        if not np.isnan(row["same_person_percentage_valid"]):
            print(f"  Same person (% valid):   {row['same_person_percentage_valid']:.2f}%")
            print(f"  Different (% valid):     {row['not_same_person_percentage_valid']:.2f}%")

        if not np.isnan(row["mean_distance"]):
            print(f"  Mean distance:           {row['mean_distance']:.4f}")
            print(f"  Median distance:         {row['median_distance']:.4f}")
            print(f"  Min distance:            {row['min_distance']:.4f}")
            print(f"  Max distance:            {row['max_distance']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare images from two folders using DeepFace models."
    )
    parser.add_argument("--folder1", required=True, help="First image folder")
    parser.add_argument("--folder2", required=True, help="Second image folder")
    parser.add_argument("--output", required=True, help="Directory for CSV results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    folder1_files = list_image_files(args.folder1)
    folder2_files = list_image_files(args.folder2)

    if len(folder1_files) != len(folder2_files):
        raise ValueError(
            f"The folders must contain the same number of images. "
            f"Folder 1: {len(folder1_files)}; Folder 2: {len(folder2_files)}"
        )

    print("\nReading image files...")
    print(f"Folder 1: {args.folder1} ({len(folder1_files)} images)")
    print(f"Folder 2: {args.folder2} ({len(folder2_files)} images)")

    all_results = []

    for model_name in MODELS:
        print("\n" + "=" * 100)
        print(f"MODEL: {model_name}")
        print("=" * 100)

        print("\nExperiment 1: images within same folder")
        results_1a = experiment_same_folder("folder_1", folder1_files, model_name)
        results_1b = experiment_same_folder("folder_2", folder2_files, model_name)
        all_results.extend(results_1a)
        all_results.extend(results_1b)

        print("\nExperiment 2: images across folders")
        results_2 = experiment_cross_folder(folder1_files, folder2_files, model_name)
        all_results.extend(results_2)

    results_df = pd.DataFrame(all_results)
    detailed_csv = output_dir / "detailed_results.csv"
    results_df.to_csv(detailed_csv, index=False)

    summary_df = summarize_results(all_results)
    summary_csv = output_dir / "summary_results.csv"
    summary_df.to_csv(summary_csv, index=False)

    print_summary(summary_df)

    print("\n" + "=" * 100)
    print("OUTPUT FILES")
    print("=" * 100)
    print(f"\nDetailed results: {detailed_csv}")
    print(f"Summary: {summary_csv}")
    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
