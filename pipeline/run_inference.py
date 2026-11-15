"""Run DINOv2 pose-regressor inference on a video."""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import torch
from PIL import Image
from torchvision import transforms

from train_pose_estimator import PoseRegressor, resolve_device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict annotated keypoints for sampled video frames."
    )
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--pose_model", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="dinov2_vits14",
        help="DINOv2 backbone used to extract the pose model's input features.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: process every frame).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device; auto uses a supported CUDA GPU, otherwise CPU.",
    )
    return parser.parse_args(argv)


def load_pose_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[PoseRegressor, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pose model checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required_fields = {
        "model_state_dict",
        "input_dim",
        "output_dim",
        "target_shape",
        "feature_mean",
        "feature_std",
        "target_mean",
        "target_std",
    }
    if not isinstance(checkpoint, dict) or not required_fields.issubset(checkpoint):
        missing = sorted(required_fields - set(checkpoint)) if isinstance(checkpoint, dict) else sorted(required_fields)
        raise ValueError(
            "Unsupported pose checkpoint. Expected a checkpoint created by "
            f"train_pose_estimator.py; missing fields: {', '.join(missing)}"
        )

    model = PoseRegressor(int(checkpoint["input_dim"]), int(checkpoint["output_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to(device)
    for field in ("feature_mean", "feature_std", "target_mean", "target_std"):
        checkpoint[field] = torch.as_tensor(checkpoint[field], dtype=torch.float32).to(device)
    return model, checkpoint


def decode_predictions(
    frames: list[tuple[int, Any]],
    feature_model: torch.nn.Module,
    pose_model: PoseRegressor,
    checkpoint: dict[str, Any],
    preprocess: transforms.Compose,
    device: torch.device,
    feature_dim: int,
    keypoint_names: list[str],
    target_shape: tuple[int, ...],
    fps: float,
) -> list[dict[str, Any]]:
    images = []
    for _, frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(preprocess(Image.fromarray(rgb_frame)))

    batch = torch.stack(images).to(device)
    with torch.inference_mode():
        features = feature_model.forward_features(batch)["x_norm_clstoken"]
        if features.shape[1] != feature_dim:
            raise ValueError(
                f"DINOv2 produced {features.shape[1]} features, but the pose checkpoint "
                f"expects {feature_dim}; check the --model argument."
            )
        normalized_features = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
        normalized_predictions = pose_model(normalized_features)
        predictions = normalized_predictions * checkpoint["target_std"] + checkpoint["target_mean"]

    results = []
    for (frame_index, _), prediction in zip(frames, predictions.cpu()):
        values = prediction.reshape(target_shape).tolist()
        if keypoint_names:
            if len(target_shape) != 2 or len(keypoint_names) != target_shape[0]:
                raise ValueError("Checkpoint keypoint names do not match its target shape")
            keypoints: Any = dict(zip(keypoint_names, values))
        else:
            keypoints = values
        results.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": frame_index / fps if fps > 0 else None,
                "keypoints": keypoints,
            }
        )
    return results


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be at least 1")
    if not args.video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {args.video_path}")

    device = resolve_device(args.device)
    pose_model, checkpoint = load_pose_model(args.pose_model, device)
    target_shape = tuple(int(size) for size in checkpoint["target_shape"])
    if not target_shape or any(size < 1 for size in target_shape):
        raise ValueError("Pose checkpoint contains an invalid target_shape")
    if torch.tensor(target_shape).prod().item() != int(checkpoint["output_dim"]):
        raise ValueError("Pose checkpoint target_shape does not match output_dim")

    dinov2_root = Path(__file__).resolve().parent / "dinov2"
    if not (dinov2_root / "hubconf.py").is_file():
        raise FileNotFoundError(f"Expected the DINOv2 repository at {dinov2_root}")
    print(f"Loading {args.model} and pose estimator on {device}...")
    feature_model = torch.hub.load(str(dinov2_root), args.model, source="local")
    feature_model.eval().to(device)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    feature_dim = int(checkpoint["input_dim"])
    keypoint_names = checkpoint.get("keypoint_names", [])
    capture = cv2.VideoCapture(str(args.video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {args.video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    results: list[dict[str, Any]] = []
    pending_frames: list[tuple[int, Any]] = []
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % args.frame_stride == 0:
                pending_frames.append((frame_index, frame))
                if len(pending_frames) >= args.batch_size:
                    results.extend(
                        decode_predictions(
                            pending_frames,
                            feature_model,
                            pose_model,
                            checkpoint,
                            preprocess,
                            device,
                            feature_dim,
                            keypoint_names,
                            target_shape,
                            fps,
                        )
                    )
                    pending_frames.clear()
                    print(
                        f"Processed frame {frame_index + 1}"
                        + (f"/{frame_count}" if frame_count > 0 else ""),
                        end="\r",
                        flush=True,
                    )
            frame_index += 1

        if pending_frames:
            results.extend(
                decode_predictions(
                    pending_frames,
                    feature_model,
                    pose_model,
                    checkpoint,
                    preprocess,
                    device,
                    feature_dim,
                    keypoint_names,
                    target_shape,
                    fps,
                )
            )
    finally:
        capture.release()

    if not results:
        raise ValueError(f"No video frames were processed from {args.video_path}")

    output_data = {
        "video_path": str(args.video_path),
        "pose_model": str(args.pose_model),
        "feature_model": args.model,
        "fps": fps if fps > 0 else None,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frame_stride": args.frame_stride,
        "processed_frame_count": len(results),
        "target_shape": target_shape,
        "keypoint_names": keypoint_names,
        "frames": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as output_file:
        json.dump(output_data, output_file, indent=2)
        output_file.write("\n")
    print(f"\nSaved predictions for {len(results)} frames to {args.output_json}")


if __name__ == "__main__":
    main()
