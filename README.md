# Monster Animation Studio 

End-to-end 3D creature and monster animation pipeline powered by **SkinTokens / UniRig** (auto-rigging & skinning), **Seedance** (dual-orthographic reference video motion generation), **Meta DINOv2** (self-supervised visual feature extraction & pose regression), and **Headless Blender 4.2** (synthetic motion-blurred dataset generation, 2D-to-3D bone retargeting, keyframe baking, and `.glb` export).

---

## 📁 Directory Structure

```text
D:\clean-implementation\
├── docker-compose.yml                        # Multi-container Web App & CLI Pipeline orchestration
├── docker-compose.gpu.yml                    # NVIDIA GPU device reservation overlay
├── .env.example                              # Environment variables & hyperparameter defaults
├── .dockerignore                             # Docker build context exclusions
│
├── docs/                                     # 📚 Pipeline Documentation
│   └── pipeline-guide-es.md                  # Detailed Spanish pipeline guide (Steps 1–5)
│
├── scripts/                                  # 🛠️ Canonical Installers & Environment Verifiers
│   ├── installers/
│   │   ├── install_dinov2.bat                # Meta DINOv2 setup script
│   │   ├── install_skintokens.bat            # SkinTokens setup (Astral uv + Python 3.11 + PyTorch 2.7 cu128)
│   │   └── install_unirig.bat                # UniRig setup (Astral uv + Python 3.11 + bpy 4.2 + spconv-cu120)
│   └── verify/
│       ├── verify_skintokens.py              # Verifies Python 3.11, CUDA, HuggingFace, and scientific deps
│       └── verify_unirig.py                  # Verifies Python 3.11, CUDA, bpy 4.2, and spconv
│
├── blender-addons/                           # 🎨 Blender Add-ons
│   ├── ai_anim_plugin.py                     # Step 3A: DINOv2 synthetic motion-blur dataset & annotations generator
│   └── pose_retarget_plugin.py               # Step 5B: 2D keypoint track to 3D Armature quaternion retargeter
│
├── pipeline/                                 # 🧠 Core DINOv2 Pose Estimation Pipeline
│   ├── Dockerfile                            # PyTorch 2.7.1 CUDA 12.8 runtime container
│   ├── requirements.txt                      # Pipeline Python dependencies
│   ├── pipeline_runner.py                    # Sequential orchestrator (Extract -> Train -> Infer)
│   ├── extract_features.py                   # Step 3B: Extracts normalized DINOv2 CLS-token features
│   ├── train_pose_estimator.py               # Step 3C: Trains MLP PoseRegressor on features.pt + annotations.json
│   └── run_inference.py                      # Step 4/5: Runs frame-by-frame keypoint inference on video
│
├── engines/                                  # ⚙️ External AI Engines (Clean source + Pretrained Weights)
│   ├── dinov2/                               # Meta DINOv2 repository (hubconf.py + dinov2 package)
│   ├── SkinTokens/                           # VAST-AI SkinTokens (TokenRig + FSQ-CVAE + Qwen3-0.6B checkpoints)
│   └── UniRig/                               # VAST-AI UniRig (Articulation-XL2.0 skeleton & skinning framework)
│
└── workspace/                                # 📦 Project Data, 3D Assets, Datasets & Outputs
    ├── cache/
    │   └── torch/                            # Pre-cached DINOv2 backbone weights (dinov2_vits14_pretrain.pth)
    └── projects/
        └── spider-rider/                     # Default Creature Project organized by Stepper stage
            ├── 01-models/
            │   ├── raw/                      # Original uncleaned 3D meshes (.glb)
            │   ├── clean/                    # Manifold-repaired meshes (.glb, .obj, .mtl)
            │   └── rigged/                   # Auto-rigged meshes & skeletons (.glb, .fbx)
            ├── 02-seedance/
            │   ├── references/               # Front, Side, and Dual-Orthographic #FF00FF chroma renders
            │   └── videos/                   # Generated Seedance motion video (seedance_video.mp4)
            ├── 03-dataset/
            │   └── dinov2/                   # 3,000 synthetic motion-blurred renders + 44.7 MB annotations.json
            └── 04-outputs/
                ├── features/                 # Extracted DINOv2 feature tensor (features.pt)
                ├── checkpoints/              # Trained pose regressor weights (pose_estimator.pt)
                └── tracks/                   # Inferred frame-by-frame 2D bone tracks (tracked_poses.json)
```

---

## 🚀 Quick Start with Docker Compose

### 1. Run the Full Web Application Stack (with NVIDIA GPU)
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
- **Web UI (5-Step Stepper & 3D Viewer):** `http://localhost:3000`
- **Backend API & WebSocket Orchestrator:** `http://localhost:8000`

### 2. Run the Standalone Batch DINOv2 Pipeline (CLI Profile)
```bash
# CPU / Auto device
docker compose --profile cli run --build --rm monster-animation

# NVIDIA GPU accelerated
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile cli run --build --rm monster-animation
```
This reads from `workspace/projects/spider-rider/03-dataset/dinov2/` and `workspace/projects/spider-rider/02-seedance/videos/seedance_video.mp4`, and writes outputs to `workspace/projects/spider-rider/04-outputs/`.
