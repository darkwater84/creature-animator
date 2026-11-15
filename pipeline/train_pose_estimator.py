"""Train a pose/keypoint regressor from DINOv2 features.

Expected annotation JSON:
{
  "annotations": [
    {"image": "pose_blur_0000.png", "keypoints": [[x1, y1], [x2, y2]]}
  ]
}
Keypoints may have any fixed coordinate dimension, such as (x, y) or (x, y, z).
"""

import argparse
import json
import math
import random
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PoseRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a regression model on features.pt and annotations JSON. "
            "The JSON must contain an 'annotations' list with image paths and "
            "fixed-shape numeric 'keypoints' for each labeled image."
        ),
        epilog=(
            'Example entry: {"image": "pose_blur_0000.png", '
            '"keypoints": [[0.1, 0.2], [0.5, 0.8]]}.'
        ),
    )
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--annotations_path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device; auto uses a supported CUDA GPU, otherwise CPU.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Checkpoint path (default: <features_dir>/pose_estimator.pt).",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    automatic = requested == "auto"
    if automatic:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        requested = "cuda"

    device = torch.device(requested)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available in this PyTorch install")

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    architecture = f"sm_{capability[0]}{capability[1]}"
    if architecture not in torch.cuda.get_arch_list():
        if automatic:
            print(
                f"Warning: PyTorch does not support this GPU ({architecture}); falling back to CPU."
            )
            return torch.device("cpu")
        raise RuntimeError(
            f"PyTorch does not support this GPU ({architecture}); install a compatible PyTorch build."
        )
    return device


def normalize_image_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Each annotation must have a non-empty string 'image' path")
    return PurePosixPath(value.replace("\\", "/")).as_posix()


def read_training_data(
    features_dir: Path, annotations_path: Path
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], list[str], list[str]]:
    feature_path = features_dir / "features.pt"
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotations_path}")

    feature_data = torch.load(feature_path, map_location="cpu", weights_only=True)
    if not isinstance(feature_data, dict) or "features" not in feature_data or "image_paths" not in feature_data:
        raise ValueError(
            f"{feature_path} must contain 'features' and 'image_paths' entries"
        )
    features = torch.as_tensor(feature_data["features"], dtype=torch.float32)
    image_paths = feature_data["image_paths"]
    if features.ndim != 2 or len(image_paths) != features.shape[0]:
        raise ValueError("Feature matrix must be 2D and have one image path per row")
    feature_indices = {normalize_image_path(path): index for index, path in enumerate(image_paths)}
    if len(feature_indices) != len(image_paths):
        raise ValueError("Feature file contains duplicate image paths")

    with annotations_path.open("r", encoding="utf-8-sig") as annotation_file:
        annotation_data = json.load(annotation_file)
    if isinstance(annotation_data, dict):
        annotation_records = annotation_data.get("annotations", annotation_data)
        if isinstance(annotation_records, dict):
            records = [
                {
                    "image": image_path,
                    "keypoints": (
                        value.get("keypoints", value.get("joints", value.get("points")))
                        if isinstance(value, dict)
                        and any(key in value for key in ("keypoints", "joints", "points"))
                        else value
                    ),
                }
                for image_path, value in annotation_records.items()
            ]
        else:
            records = annotation_records
    else:
        records = annotation_data
    if not isinstance(records, list) or not records:
        raise ValueError("Annotations JSON must be a non-empty list or an object with an 'annotations' list")

    # Expand v2 grouped pose records ({"pose_id": i, "renders": {"front": {...}, "side": {...}}})
    expanded_records: list[dict[str, Any]] = []
    for record_number, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Annotation {record_number} must be a JSON object")
        grouped_renders = record.get("renders", record.get("views"))
        if isinstance(grouped_renders, dict) and grouped_renders:
            for view_name, view_entry in grouped_renders.items():
                if not isinstance(view_entry, dict):
                    raise ValueError(
                        f"Annotation {record_number} render view {view_name!r} must be a JSON object"
                    )
                expanded_records.append(
                    {
                        "pose_id": record.get("pose_id", record_number),
                        "view": view_name,
                        "image": view_entry.get("image", view_entry.get("image_path")),
                        "keypoints": view_entry.get(
                            "keypoints", view_entry.get("joints", view_entry.get("points"))
                        ),
                    }
                )
        elif isinstance(grouped_renders, list) and grouped_renders:
            for view_entry in grouped_renders:
                if not isinstance(view_entry, dict):
                    raise ValueError(f"Annotation {record_number} render entry must be a JSON object")
                expanded_records.append(
                    {
                        "pose_id": record.get("pose_id", record_number),
                        "view": view_entry.get("view", "default"),
                        "image": view_entry.get("image", view_entry.get("image_path")),
                        "keypoints": view_entry.get(
                            "keypoints", view_entry.get("joints", view_entry.get("points"))
                        ),
                    }
                )
        else:
            expanded_records.append(record)
    records = expanded_records

    matched_indices: list[int] = []
    targets: list[torch.Tensor] = []
    names: list[str] | None = None
    expected_shape: tuple[int, ...] | None = None
    seen_paths: set[str] = set()
    unmatched: list[str] = []

    for record_number, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Annotation {record_number} must be a JSON object")
        path_value = record.get("image", record.get("image_path"))
        relative_path = normalize_image_path(path_value)
        if relative_path in seen_paths:
            raise ValueError(f"Duplicate annotation for image {relative_path!r}")
        seen_paths.add(relative_path)

        if relative_path not in feature_indices:
            unmatched.append(relative_path)
            continue

        raw_target = record.get("keypoints", record.get("joints", record.get("points")))
        if raw_target is None:
            raise ValueError(
                f"Annotation for {relative_path!r} is missing 'keypoints'/'joints'"
            )
        if isinstance(raw_target, dict):
            if names is None:
                names = list(raw_target)
                if not names:
                    raise ValueError(f"Annotation for {relative_path!r} has no keypoints")
            if set(raw_target) != set(names):
                raise ValueError(f"Named keypoints differ for image {relative_path!r}")
            raw_target = [raw_target[name] for name in names]
            raw_target = [
                [point[axis] for axis in ("x", "y", "z") if axis in point]
                if isinstance(point, dict)
                and "x" in point
                and "y" in point
                else point
                for point in raw_target
            ]

        try:
            target = torch.as_tensor(raw_target, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Keypoints for {relative_path!r} must be a numeric array"
            ) from exc
        if target.ndim == 0 or target.numel() == 0 or not torch.isfinite(target).all():
            raise ValueError(f"Keypoints for {relative_path!r} must contain finite numeric values")
        if expected_shape is None:
            expected_shape = tuple(target.shape)
        elif tuple(target.shape) != expected_shape:
            raise ValueError(
                f"Keypoint shape for {relative_path!r} is {tuple(target.shape)}, "
                f"expected {expected_shape}"
            )

        matched_indices.append(feature_indices[relative_path])
        targets.append(target.reshape(-1))

    if unmatched:
        examples = ", ".join(repr(path) for path in unmatched[:5])
        raise ValueError(
            f"{len(unmatched)} annotated image(s) were not found in features.pt: {examples}"
        )
    if len(matched_indices) < 2:
        raise ValueError("At least two annotated images matching the feature file are required")

    selected_features = features[matched_indices]
    if not torch.isfinite(selected_features).all():
        raise ValueError("Feature file contains non-finite values for annotated images")
    return (
        selected_features,
        torch.stack(targets),
        expected_shape or (),
        [normalize_image_path(image_paths[index]) for index in matched_indices],
        names or [],
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative")
    if not 0.0 < args.validation_split < 1.0:
        raise ValueError("--validation_split must be between 0 and 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    features, targets, target_shape, image_paths, keypoint_names = read_training_data(
        args.features_dir, args.annotations_path
    )
    sample_count = len(features)
    validation_count = max(1, math.ceil(sample_count * args.validation_split))
    validation_count = min(validation_count, sample_count - 1)
    permutation = torch.randperm(sample_count)
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]

    feature_mean = features[training_indices].mean(dim=0)
    feature_std = features[training_indices].std(dim=0, unbiased=False).clamp_min(1e-6)
    target_mean = targets[training_indices].mean(dim=0)
    target_std = targets[training_indices].std(dim=0, unbiased=False).clamp_min(1e-6)

    normalized_features = (features - feature_mean) / feature_std
    normalized_targets = (targets - target_mean) / target_std
    training_loader = DataLoader(
        TensorDataset(
            normalized_features[training_indices],
            normalized_targets[training_indices],
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    device = resolve_device(args.device)
    model = PoseRegressor(features.shape[1], targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()
    validation_x = normalized_features[validation_indices].to(device)
    validation_y = normalized_targets[validation_indices].to(device)
    best_validation_loss = math.inf
    output_path = args.output_path or args.features_dir / "pose_estimator.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Training on {len(training_indices)} images; validating on {validation_count}; "
        f"target shape {target_shape}; device {device}."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_features, batch_targets in training_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_features)
            seen += len(batch_features)

        model.eval()
        with torch.inference_mode():
            validation_predictions = model(validation_x)
            validation_loss = loss_function(validation_predictions, validation_y).item()
            predictions_in_units = validation_predictions * target_std.to(device) + target_mean.to(device)
            targets_in_units = validation_y * target_std.to(device) + target_mean.to(device)
            validation_rmse = torch.mean((predictions_in_units - targets_in_units) ** 2).sqrt().item()

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_mse={total_loss / seen:.6f} "
            f"val_mse={validation_loss:.6f} "
            f"val_rmse={validation_rmse:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "input_dim": features.shape[1],
                    "output_dim": targets.shape[1],
                    "target_shape": target_shape,
                    "keypoint_names": keypoint_names,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "image_paths": image_paths,
                    "validation_loss": validation_loss,
                    "epoch": epoch,
                },
                output_path,
            )

    print(f"Saved best validation checkpoint to {output_path}")


if __name__ == "__main__":
    main()
