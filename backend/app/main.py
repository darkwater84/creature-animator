"""FastAPI Orchestrator for the 5-Step Creature & Monster Animation Studio."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import httpx
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
PROJECTS_DIR = Path(os.environ.get("PROJECTS_DIR", str(WORKSPACE_DIR / "projects")))
PIPELINE_DIR = Path("/app/pipeline")
RIGGING_SERVICE_URL = os.environ.get("RIGGING_SERVICE_URL", "http://rigging-worker:8001")
BLENDER_SERVICE_URL = os.environ.get("BLENDER_SERVICE_URL", "http://blender-worker:8002")

mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")
mimetypes.add_type("text/plain", ".obj")
mimetypes.add_type("text/plain", ".mtl")

app = FastAPI(
    title="Monster Animation Studio API",
    version="1.0.0",
    description="Orchestrates SkinTokens, UniRig, Seedance Reference Setup, Meta DINOv2 Pose Training, and Headless Blender Retargeting.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for ws in list(self.active):
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


ws_manager = ConnectionManager()
JOBS: dict[str, dict[str, Any]] = {}
EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+train_mse=([\d.eE+-]+)\s+val_mse=([\d.eE+-]+)\s+val_rmse=([\d.eE+-]+)"
)


# ============================================================================
# Request Models
# ============================================================================
class StepUpdateRequest(BaseModel):
    current_step: int = Field(ge=1, le=5)


class CleanupRequest(BaseModel):
    input_file: str = "01-models/raw/giant_spider_rider_raw.glb"
    remove_gltf_not_exported: bool = True
    repair_non_manifold: bool = True


class RiggingRequest(BaseModel):
    engine: str = "skintokens"  # "skintokens" | "unirig"
    input_file: str = "01-models/clean/giant_spider_rider_clean.glb"
    num_bones: int = 24
    resolution: int = 256
    cleanup_loose_bones: bool = True
    use_transfer: bool = True
    use_postprocess: bool = False
    top_k: int = 5
    top_p: float = 0.95
    temperature: float = 1.0
    num_beams: int = 10


class OrthoRenderRequest(BaseModel):
    rigged_file: str = "01-models/rigged/spider_rider_rigged_v2.glb"
    bg_color: str = "#FF00FF"
    resolution_x: int = 1920
    resolution_y: int = 1080


class DatasetGenRequest(BaseModel):
    rigged_file: str = "01-models/rigged/spider_rider_rigged_v2.glb"
    num_renders: int = 50
    max_rotation_deg: float = 45.0
    motion_blur_shutter: float = 0.65


class ExtractFeaturesRequest(BaseModel):
    model: str = "dinov2_vits14"
    batch_size: int = 32
    device: str = "auto"


class TrainPoseRequest(BaseModel):
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    validation_split: float = 0.2
    device: str = "auto"


class ValidationRequest(BaseModel):
    video_file: str = "02-seedance/videos/seedance_video.mp4"
    model: str = "dinov2_vits14"
    max_frames: int = 30
    device: str = "auto"


class VideoInferenceRequest(BaseModel):
    video_file: str = "02-seedance/videos/seedance_video.mp4"
    model: str = "dinov2_vits14"
    batch_size: int = 16
    frame_stride: int = 1
    device: str = "auto"


class RetargetExportRequest(BaseModel):
    rigged_file: str = "01-models/rigged/spider_rider_rigged_v2.glb"
    tracks_file: str = "04-outputs/tracks/tracked_poses.json"
    retarget_space: str = "ARMATURE_BASE"  # "CAMERA_PLANE" | "ARMATURE_BASE"
    plane_depth: float = 10.0
    rotation_influence: float = 0.15
    start_frame: int = 1
    frame_step: float = 1.0
    export_format: str = "glb"


# ============================================================================
# Helper Functions
# ============================================================================
def ensure_project_dirs(project_dir: Path) -> None:
    for sub in (
        "01-models/raw",
        "01-models/clean",
        "01-models/rigged",
        "02-seedance/references",
        "02-seedance/videos",
        "03-dataset/dinov2",
        "04-outputs/features",
        "04-outputs/checkpoints",
        "04-outputs/tracks",
        "04-outputs/validation",
        "04-outputs/exports",
    ):
        (project_dir / sub).mkdir(parents=True, exist_ok=True)


def list_relative_files(project_dir: Path, subdir: str, exts: set[str] | None = None) -> list[dict[str, Any]]:
    target = project_dir / subdir
    if not target.is_dir():
        return []
    items = []
    for p in sorted(target.rglob("*")):
        if p.is_file():
            if exts and p.suffix.lower() not in exts:
                continue
            rel = p.relative_to(project_dir).as_posix()
            items.append(
                {
                    "name": p.name,
                    "path": rel,
                    "size_bytes": p.stat().st_size,
                    "modified_at": p.stat().st_mtime,
                }
            )
    return items


def inspect_video_metadata(video_path: Path) -> dict[str, Any] | None:
    if not video_path.is_file():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return {
            "fps": round(fps, 2),
            "frame_count": frames,
            "width": width,
            "height": height,
            "duration_sec": round(frames / fps, 2) if fps > 0 else 0,
        }
    finally:
        cap.release()


def get_project_state(project_id: str) -> dict[str, Any]:
    project_dir = PROJECTS_DIR / project_id
    if not project_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    ensure_project_dirs(project_dir)

    state_file = project_dir / "project.json"
    if state_file.is_file():
        try:
            saved = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            saved = {}
    else:
        saved = {}

    raw_models = list_relative_files(project_dir, "01-models/raw", {".glb", ".obj", ".fbx", ".gltf"})
    clean_models = list_relative_files(project_dir, "01-models/clean", {".glb", ".obj", ".fbx", ".gltf"})
    rigged_models = list_relative_files(project_dir, "01-models/rigged", {".glb", ".obj", ".fbx", ".gltf"})
    references = list_relative_files(project_dir, "02-seedance/references", {".png", ".jpg", ".jpeg", ".webp"})
    videos = list_relative_files(project_dir, "02-seedance/videos", {".mp4", ".webm", ".mov"})
    exports = list_relative_files(project_dir, "04-outputs/exports", {".glb", ".gltf", ".fbx"})

    dataset_dir = project_dir / "03-dataset/dinov2"
    png_count = sum(1 for _ in dataset_dir.glob("*.png")) if dataset_dir.is_dir() else 0
    annot_file = dataset_dir / "annotations.json"
    has_annotations = annot_file.is_file() and annot_file.stat().st_size > 200_000

    features_file = project_dir / "04-outputs/features/features.pt"
    checkpoint_file = project_dir / "04-outputs/checkpoints/pose_estimator.pt"
    tracks_file = project_dir / "04-outputs/tracks/tracked_poses.json"
    val_tracks_file = project_dir / "04-outputs/validation/val_tracked_poses.json"

    video_meta = None
    if videos:
        video_meta = inspect_video_metadata(project_dir / videos[0]["path"])

    checkpoint_meta = None
    if checkpoint_file.is_file():
        try:
            ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
            checkpoint_meta = {
                "input_dim": int(ckpt.get("input_dim", 384)),
                "output_dim": int(ckpt.get("output_dim", 0)),
                "target_shape": list(ckpt.get("target_shape", [])),
                "keypoint_count": len(ckpt.get("keypoint_names", [])),
                "bone_count": len(ckpt.get("keypoint_names", [])) // 2,
                "best_epoch": int(ckpt.get("epoch", 0)),
                "validation_loss": round(float(ckpt.get("validation_loss", 0.0)), 6),
            }
        except Exception:
            checkpoint_meta = {"exists": True}

    steps_completed = {
        "1": len(rigged_models) > 0,
        "2": len(references) > 0 and len(videos) > 0,
        "3": features_file.is_file() and checkpoint_file.is_file(),
        "4": tracks_file.is_file() or val_tracks_file.is_file(),
        "5": tracks_file.is_file(),
    }

    state = {
        "project_id": project_id,
        "name": saved.get("name", project_id.replace("-", " ").title()),
        "current_step": saved.get("current_step", 1),
        "steps_completed": steps_completed,
        "assets": {
            "raw_models": raw_models,
            "clean_models": clean_models,
            "rigged_models": rigged_models,
            "references": references,
            "videos": videos,
            "exports": exports,
        },
        "video_metadata": video_meta,
        "dataset": {
            "render_count": png_count,
            "has_valid_annotations": has_annotations,
            "annotations_size_mb": round(annot_file.stat().st_size / (1024 * 1024), 2) if annot_file.is_file() else 0,
        },
        "pipeline_artifacts": {
            "features_ready": features_file.is_file(),
            "features_size_mb": round(features_file.stat().st_size / (1024 * 1024), 2) if features_file.is_file() else 0,
            "checkpoint_ready": checkpoint_file.is_file(),
            "checkpoint_meta": checkpoint_meta,
            "tracks_ready": tracks_file.is_file(),
            "validation_ready": val_tracks_file.is_file() or tracks_file.is_file(),
        },
        "training_history": saved.get("training_history", []),
    }
    return state


def save_project_patch(project_id: str, patch: dict[str, Any]) -> None:
    project_dir = PROJECTS_DIR / project_id
    state_file = project_dir / "project.json"
    data = {}
    if state_file.is_file():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.update(patch)
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


async def run_subprocess_job(
    job_id: str,
    project_id: str,
    step: int,
    title: str,
    cmd: list[str],
    cwd: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> None:
    JOBS[job_id]["status"] = "running"
    await ws_manager.broadcast({"type": "job_update", "job": JOBS[job_id]})

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    training_epochs: list[dict[str, Any]] = []
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        buffer = ""
        while True:
            chunk = await proc.stdout.read(256)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            buffer += text
            while "\n" in buffer or "\r" in buffer:
                sep_idx = min(
                    idx for idx in (buffer.find("\n"), buffer.find("\r")) if idx != -1
                )
                line = buffer[:sep_idx].strip()
                buffer = buffer[sep_idx + 1 :]
                if not line:
                    continue
                JOBS[job_id]["logs"].append(line)
                if len(JOBS[job_id]["logs"]) > 300:
                    JOBS[job_id]["logs"] = JOBS[job_id]["logs"][-300:]

                match = EPOCH_PATTERN.search(line)
                if match:
                    metric = {
                        "epoch": int(match.group(1)),
                        "total_epochs": int(match.group(2)),
                        "train_mse": float(match.group(3)),
                        "val_mse": float(match.group(4)),
                        "val_rmse": float(match.group(5)),
                    }
                    training_epochs.append(metric)
                    JOBS[job_id]["metrics"] = training_epochs
                    JOBS[job_id]["progress"] = round(100.0 * metric["epoch"] / max(1, metric["total_epochs"]), 1)

                await ws_manager.broadcast(
                    {
                        "type": "job_log",
                        "job_id": job_id,
                        "project_id": project_id,
                        "step": step,
                        "line": line,
                        "progress": JOBS[job_id].get("progress"),
                        "metrics": JOBS[job_id].get("metrics"),
                    }
                )

        rc = await proc.wait()
        if training_epochs:
            save_project_patch(project_id, {"training_history": training_epochs})

        if rc == 0:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["progress"] = 100.0
        else:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = f"Process exited with code {rc}"
    except Exception as exc:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(exc)
        JOBS[job_id]["logs"].append(f"ERROR: {exc}")

    JOBS[job_id]["finished_at"] = time.time()
    await ws_manager.broadcast({"type": "job_update", "job": JOBS[job_id]})


def create_job_record(project_id: str, step: int, title: str) -> dict[str, Any]:
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    job = {
        "job_id": job_id,
        "project_id": project_id,
        "step": step,
        "title": title,
        "status": "queued",
        "progress": 0.0,
        "logs": [],
        "metrics": [],
        "created_at": time.time(),
        "finished_at": None,
        "error": None,
    }
    JOBS[job_id] = job
    return job


# ============================================================================
# Core Routes
# ============================================================================
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "api-orchestrator",
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws_manager.connect(ws)
    try:
        await ws.send_json({"type": "connected", "jobs": list(JOBS.values())[-10:]})
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


@app.get("/api/system/status")
async def system_status() -> dict[str, Any]:
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if cuda_ok
        else 0
    )

    rigging_status: dict[str, Any] = {"status": "unreachable"}
    blender_status: dict[str, Any] = {"status": "unreachable"}

    async with httpx.AsyncClient(timeout=4.0) as client:
        try:
            r = await client.get(f"{RIGGING_SERVICE_URL}/health")
            rigging_status = r.json()
        except Exception as exc:
            rigging_status = {"status": "offline", "detail": str(exc)}
        try:
            b = await client.get(f"{BLENDER_SERVICE_URL}/health")
            blender_status = b.json()
        except Exception as exc:
            blender_status = {"status": "offline", "detail": str(exc)}

    return {
        "orchestrator": {
            "status": "online",
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "cuda_available": cuda_ok,
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "dinov2_mounted": Path("/app/pipeline/dinov2/hubconf.py").is_file(),
        },
        "rigging_worker": rigging_status,
        "blender_worker": blender_status,
    }


@app.get("/api/projects")
async def list_projects() -> list[dict[str, Any]]:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(PROJECTS_DIR.iterdir()):
        if p.is_dir():
            result.append(get_project_state(p.name))
    return result


@app.get("/api/projects/{project_id}")
async def read_project(project_id: str) -> dict[str, Any]:
    return get_project_state(project_id)


@app.patch("/api/projects/{project_id}/step")
async def update_project_step(project_id: str, req: StepUpdateRequest) -> dict[str, Any]:
    save_project_patch(project_id, {"current_step": req.current_step})
    return get_project_state(project_id)


@app.get("/api/projects/{project_id}/files/{file_path:path}")
async def serve_project_file(project_id: str, file_path: str) -> FileResponse:
    project_dir = (PROJECTS_DIR / project_id).resolve()
    target = (project_dir / file_path).resolve()
    if not str(target).startswith(str(project_dir)) or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    media_type, _ = mimetypes.guess_type(str(target))
    return FileResponse(target, media_type=media_type or "application/octet-stream")


@app.post("/api/projects/{project_id}/upload/{category}")
async def upload_project_asset(
    project_id: str, category: str, file: UploadFile = File(...)
) -> dict[str, Any]:
    project_dir = PROJECTS_DIR / project_id
    ensure_project_dirs(project_dir)
    category_map = {
        "raw_model": "01-models/raw",
        "clean_model": "01-models/clean",
        "rigged_model": "01-models/rigged",
        "reference": "02-seedance/references",
        "video": "02-seedance/videos",
    }
    if category not in category_map:
        raise HTTPException(status_code=400, detail=f"Unsupported upload category: {category}")

    safe_name = Path(file.filename or "uploaded_file").name
    dest = project_dir / category_map[category] / safe_name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {
        "uploaded": dest.relative_to(project_dir).as_posix(),
        "project": get_project_state(project_id),
    }


@app.get("/api/projects/{project_id}/dataset-summary")
async def get_dataset_summary(project_id: str) -> dict[str, Any]:
    project_dir = PROJECTS_DIR / project_id
    dataset_dir = project_dir / "03-dataset/dinov2"
    annot_path = dataset_dir / "annotations.json"
    png_files = sorted(p.name for p in dataset_dir.glob("*.png")) if dataset_dir.is_dir() else []

    if not annot_path.is_file():
        return {"render_count": len(png_files), "has_annotations": False, "samples": []}

    with annot_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    records = data.get("annotations", []) if isinstance(data, dict) else []
    raw_samples = records[:12]
    normalized_samples = []
    keypoint_names: list[str] = []

    for idx, rec in enumerate(raw_samples):
        if not isinstance(rec, dict):
            continue
        pose_id = rec.get("pose_id", idx)
        grouped = rec.get("renders", rec.get("views"))
        if isinstance(grouped, dict) and grouped:
            renders_dict = {}
            for v_name, v_obj in grouped.items():
                if isinstance(v_obj, dict):
                    kp = v_obj.get("keypoints", {})
                    if not keypoint_names and isinstance(kp, dict):
                        keypoint_names = list(kp.keys())
                    renders_dict[v_name] = {
                        "view": v_name,
                        "image": v_obj.get("image", ""),
                        "camera": v_obj.get("camera", v_name),
                        "keypoints": kp,
                    }
            front_entry = renders_dict.get("front") or next(iter(renders_dict.values()))
            normalized_samples.append(
                {
                    "pose_id": pose_id,
                    "image": front_entry["image"],
                    "keypoints": front_entry["keypoints"],
                    "renders": renders_dict,
                }
            )
        else:
            kp = rec.get("keypoints", {})
            if not keypoint_names and isinstance(kp, dict):
                keypoint_names = list(kp.keys())
            img_name = rec.get("image", f"pose_blur_{idx:04d}.png")
            normalized_samples.append(
                {
                    "pose_id": pose_id,
                    "image": img_name,
                    "keypoints": kp,
                    "renders": {
                        "front": {
                            "view": "front",
                            "image": img_name,
                            "camera": "Camera_Front",
                            "keypoints": kp,
                        }
                    },
                }
            )

    bone_names = sorted({re.sub(r"_(head|tail)$", "", k) for k in keypoint_names})
    views = data.get("views", ["front", "side"]) if isinstance(data, dict) else ["front"]

    return {
        "schema_version": data.get("schema_version", "2.0") if isinstance(data, dict) else "1.0",
        "views": views,
        "render_count": len(png_files),
        "pose_count": len(records),
        "has_annotations": bool(records),
        "annotated_count": len(records),
        "coordinate_system": data.get("coordinate_system", "normalized_image_xy_origin_top_left"),
        "image_resolution": data.get("image_resolution", [1920, 1080]),
        "keypoint_count": len(keypoint_names),
        "bone_count": len(bone_names),
        "bone_names": bone_names,
        "samples": normalized_samples,
    }


@app.get("/api/projects/{project_id}/tracks")
async def get_project_tracks(project_id: str, source: str = "main") -> dict[str, Any]:
    project_dir = PROJECTS_DIR / project_id
    tracks_path = (
        project_dir / "04-outputs/validation/val_tracked_poses.json"
        if source == "validation" and (project_dir / "04-outputs/validation/val_tracked_poses.json").is_file()
        else project_dir / "04-outputs/tracks/tracked_poses.json"
    )
    if not tracks_path.is_file():
        raise HTTPException(status_code=404, detail="Tracked poses JSON not found")
    with tracks_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


# ============================================================================
# Step 1–5 Pipeline Action Endpoints
# ============================================================================
@app.post("/api/projects/{project_id}/step1/cleanup")
async def step1_cleanup_mesh(project_id: str, req: CleanupRequest) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{BLENDER_SERVICE_URL}/mesh/cleanup",
            json={
                "project_id": project_id,
                "input_file": req.input_file,
                "remove_gltf_not_exported": req.remove_gltf_not_exported,
                "repair_non_manifold": req.repair_non_manifold,
            },
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return {"result": resp.json(), "project": get_project_state(project_id)}


@app.post("/api/projects/{project_id}/step1/rig")
async def step1_auto_rig(project_id: str, req: RiggingRequest) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{RIGGING_SERVICE_URL}/rig",
            json={"project_id": project_id, **req.model_dump()},
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return {"result": resp.json(), "project": get_project_state(project_id)}


@app.post("/api/projects/{project_id}/step2/render-ortho")
async def step2_render_ortho_reference(project_id: str, req: OrthoRenderRequest) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{BLENDER_SERVICE_URL}/render/ortho-reference",
            json={"project_id": project_id, **req.model_dump()},
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return {"result": resp.json(), "project": get_project_state(project_id)}


@app.post("/api/projects/{project_id}/step3/generate-dataset")
async def step3_generate_dataset(project_id: str, req: DatasetGenRequest) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{BLENDER_SERVICE_URL}/dataset/generate",
            json={"project_id": project_id, **req.model_dump()},
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return {"result": resp.json(), "project": get_project_state(project_id)}


@app.post("/api/projects/{project_id}/step3/extract-features")
async def step3_extract_features(project_id: str, req: ExtractFeaturesRequest) -> dict[str, Any]:
    project_dir = PROJECTS_DIR / project_id
    dataset_path = project_dir / "03-dataset/dinov2"
    output_dir = project_dir / "04-outputs/features"
    job = create_job_record(project_id, 3, f"Extract DINOv2 Features ({req.model})")

    cmd = [
        sys.executable,
        str(PIPELINE_DIR / "extract_features.py"),
        "--dataset_path",
        str(dataset_path),
        "--output_dir",
        str(output_dir),
        "--model",
        req.model,
        "--batch_size",
        str(req.batch_size),
        "--device",
        req.device,
    ]
    asyncio.create_task(run_subprocess_job(job["job_id"], project_id, 3, job["title"], cmd))
    return job


@app.post("/api/projects/{project_id}/step3/train")
async def step3_train_pose_estimator(project_id: str, req: TrainPoseRequest) -> dict[str, Any]:
    project_dir = PROJECTS_DIR / project_id
    features_dir = project_dir / "04-outputs/features"
    annotations_path = project_dir / "03-dataset/dinov2/annotations.json"
    checkpoint_path = project_dir / "04-outputs/checkpoints/pose_estimator.pt"
    job = create_job_record(project_id, 3, f"Train Pose Regressor ({req.epochs} epochs)")

    cmd = [
        sys.executable,
        str(PIPELINE_DIR / "train_pose_estimator.py"),
        "--features_dir",
        str(features_dir),
        "--annotations_path",
        str(annotations_path),
        "--epochs",
        str(req.epochs),
        "--batch_size",
        str(req.batch_size),
        "--learning_rate",
        str(req.learning_rate),
        "--validation_split",
        str(req.validation_split),
        "--device",
        req.device,
        "--output_path",
        str(checkpoint_path),
    ]
    asyncio.create_task(run_subprocess_job(job["job_id"], project_id, 3, job["title"], cmd))
    return job


@app.post("/api/projects/{project_id}/step4/validate")
async def step4_run_validation(project_id: str, req: ValidationRequest) -> dict[str, Any]:
    project_dir = PROJECTS_DIR / project_id
    video_path = project_dir / req.video_file
    checkpoint_path = project_dir / "04-outputs/checkpoints/pose_estimator.pt"
    output_json = project_dir / "04-outputs/validation/val_tracked_poses.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    job = create_job_record(project_id, 4, "Validation Test ('La Prueba de Examen')")

    cmd = [
        sys.executable,
        str(PIPELINE_DIR / "run_inference.py"),
        "--video_path",
        str(video_path),
        "--pose_model",
        str(checkpoint_path),
        "--output_json",
        str(output_json),
        "--model",
        req.model,
        "--frame_stride",
        "2",
        "--device",
        req.device,
    ]
    asyncio.create_task(run_subprocess_job(job["job_id"], project_id, 4, job["title"], cmd))
    return job


@app.post("/api/projects/{project_id}/step5/infer-video")
async def step5_infer_video(project_id: str, req: VideoInferenceRequest) -> dict[str, Any]:
    project_dir = PROJECTS_DIR / project_id
    video_path = project_dir / req.video_file
    checkpoint_path = project_dir / "04-outputs/checkpoints/pose_estimator.pt"
    output_json = project_dir / "04-outputs/tracks/tracked_poses.json"
    job = create_job_record(project_id, 5, f"Track Video Poses ({Path(req.video_file).name})")

    cmd = [
        sys.executable,
        str(PIPELINE_DIR / "run_inference.py"),
        "--video_path",
        str(video_path),
        "--pose_model",
        str(checkpoint_path),
        "--output_json",
        str(output_json),
        "--model",
        req.model,
        "--batch_size",
        str(req.batch_size),
        "--frame_stride",
        str(req.frame_stride),
        "--device",
        req.device,
    ]
    asyncio.create_task(run_subprocess_job(job["job_id"], project_id, 5, job["title"], cmd))
    return job


@app.post("/api/projects/{project_id}/step5/retarget")
async def step5_retarget_and_export(project_id: str, req: RetargetExportRequest) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{BLENDER_SERVICE_URL}/retarget/export",
            json={"project_id": project_id, **req.model_dump()},
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return {"result": resp.json(), "project": get_project_state(project_id)}


@app.get("/api/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    return sorted(JOBS.values(), key=lambda j: j["created_at"], reverse=True)[:25]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]
