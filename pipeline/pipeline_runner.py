"""Run feature extraction, pose training, and video inference in order."""

import os
import json
import subprocess
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def required_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f"Required environment variable {variable} is not set")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{variable} does not exist: {path}")
    return path


def required_file(variable: str, hint: str) -> Path:
    path = required_path(variable)
    if not path.is_file():
        raise FileNotFoundError(f"{variable} must be a file: {path}. {hint}")
    return path


def ensure_annotations(dataset_path: Path, annotations_path: Path) -> None:
    if annotations_path.is_file():
        with annotations_path.open("r", encoding="utf-8-sig") as annotation_file:
            annotation_data = json.load(annotation_file)
        if isinstance(annotation_data, dict) and annotation_data.get("_template"):
            raise ValueError(
                f"Annotation template exists at {annotations_path}, but its keypoints "
                "are empty. Run the Blender dataset generator to create real joint "
                "labels (or fill in the template), then run the container again."
            )
        return

    image_paths = sorted(
        path.relative_to(dataset_path).as_posix()
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"Cannot create an annotation template: no supported images found in "
            f"{dataset_path}."
        )

    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = annotations_path.with_suffix(annotations_path.suffix + ".tmp")
    template = {
        "_template": True,
        "_instructions": (
            "This is only a template; keypoints are intentionally empty. Generate "
            "real joint annotations with the Blender dataset generator before training."
        ),
        "annotations": [
            {"image": image_path, "keypoints": {}} for image_path in image_paths
        ],
    }
    with temporary_path.open("w", encoding="utf-8") as annotation_file:
        json.dump(template, annotation_file, indent=2)
        annotation_file.write("\n")
    os.replace(temporary_path, annotations_path)
    raise ValueError(
        f"Created an annotation template at {annotations_path} for "
        f"{len(image_paths)} images, but keypoints cannot be inferred from images. "
        "Run the Blender dataset generator to replace this template with real joint "
        "annotations, then run the container again."
    )


def main() -> None:
    dataset_path = required_path("DATASET_PATH")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"DATASET_PATH must be a directory: {dataset_path}")
    annotations_path_value = os.environ.get("ANNOTATIONS_PATH")
    if not annotations_path_value:
        raise ValueError("Required environment variable ANNOTATIONS_PATH is not set")
    annotations_path = Path(annotations_path_value)
    ensure_annotations(dataset_path, annotations_path)
    video_path = required_file(
        "VIDEO_PATH",
        "Place the video in INPUTS_DIR or set VIDEO_FILE in the Compose environment.",
    )

    output_dir = Path(os.environ.get("OUTPUT_DIR", "/outputs"))
    features_dir = output_dir / "features"
    checkpoint_path = output_dir / "pose_estimator.pt"
    prediction_path = output_dir / "tracked_poses.json"
    model_name = os.environ.get("FEATURE_MODEL", "dinov2_vits14")
    device = os.environ.get("DEVICE", "auto")

    steps = [
        (
            "Extracting image features",
            [
                sys.executable,
                "extract_features.py",
                "--dataset_path",
                str(dataset_path),
                "--output_dir",
                str(features_dir),
                "--model",
                model_name,
                "--device",
                device,
            ],
        ),
        (
            "Training pose estimator",
            [
                sys.executable,
                "train_pose_estimator.py",
                "--features_dir",
                str(features_dir),
                "--annotations_path",
                str(annotations_path),
                "--epochs",
                os.environ.get("EPOCHS", "50"),
                "--batch_size",
                os.environ.get("BATCH_SIZE", "16"),
                "--device",
                device,
                "--output_path",
                str(checkpoint_path),
            ],
        ),
        (
            "Tracking video poses",
            [
                sys.executable,
                "run_inference.py",
                "--video_path",
                str(video_path),
                "--pose_model",
                str(checkpoint_path),
                "--output_json",
                str(prediction_path),
                "--model",
                model_name,
                "--frame_stride",
                os.environ.get("FRAME_STRIDE", "1"),
                "--device",
                device,
            ],
        ),
    ]

    for step_name, command in steps:
        print(f"\n=== {step_name} ===", flush=True)
        subprocess.run(command, check=True)

    print(f"\nPipeline complete. Tracked poses: {prediction_path}", flush=True)


if __name__ == "__main__":
    main()
