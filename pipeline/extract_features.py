"""Extract DINOv2 class-token features from a directory of images."""

import argparse
from pathlib import Path
import sys
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No supported images found in {root}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.relative_to(self.root).as_posix()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 normalized class-token features from images."
    )
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model", default="dinov2_vits14")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device; auto uses a supported CUDA GPU, otherwise CPU.",
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
        message = (
            f"PyTorch does not support this GPU ({architecture}). "
            "Install a compatible PyTorch build to use CUDA."
        )
        if automatic:
            print(f"Warning: {message} Falling back to CPU.", file=sys.stderr)
            return torch.device("cpu")
        raise RuntimeError(message)
    return device


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset_path}")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative")

    dataset = ImageDataset(args.dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = resolve_device(args.device)
    repo_path = Path(__file__).resolve().parent / "dinov2"
    if not (repo_path / "hubconf.py").is_file():
        raise FileNotFoundError(f"Expected the DINOv2 repository at {repo_path}")

    print(f"Loading {args.model} on {device}...")
    model = torch.hub.load(str(repo_path), args.model, source="local")
    model.eval().to(device)

    feature_batches = []
    image_paths = []
    with torch.inference_mode():
        for images, paths in loader:
            output = model.forward_features(images.to(device))
            feature_batches.append(output["x_norm_clstoken"].cpu())
            image_paths.extend(paths)
            print(f"Processed {len(image_paths)}/{len(dataset)} images", end="\r", flush=True)

    features = torch.cat(feature_batches)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "features.pt"
    torch.save(
        {
            "features": features,
            "image_paths": image_paths,
            "model": args.model,
        },
        output_path,
    )
    print(f"\nSaved {len(image_paths)} feature vectors ({features.shape[1]} dimensions) to {output_path}")


if __name__ == "__main__":
    main()
