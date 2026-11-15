import React, { useEffect, useState } from "react";
import {
  DatasetSummary,
  ProjectState,
  TrackedPosesData,
} from "./types";
import { Viewport3D } from "./components/Viewport3D";
import { VideoKeypointPlayer } from "./components/VideoKeypointPlayer";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws";
const PROJECT_ID = "spider-rider";

const SEEDANCE_PROMPTS = [
  {
    name: "Walk Cycle (Ciclo de Caminata)",
    prompt:
      "A low-poly 3D creature walking forward continuously, heavy and menacing steps, maintaining the exact same flat magenta background, static locked camera, no environment, stable silhouette.",
  },
  {
    name: "Quick Attack (Ataque Rápido)",
    prompt:
      "The creature performs a quick horizontal sword slash, aggressive dynamic motion, dark fantasy 80s anime aesthetic, flat #FF00FF background, static camera angle, no background details.",
  },
  {
    name: "Idle Breathing (Animación de Reposo)",
    prompt:
      "The monster standing still and breathing heavily, idle pose, slight shoulder and chest movement, flat magenta background, static locked camera.",
  },
];

const STEPS = [
  {
    num: 1,
    title: "1. Model & Auto-Rigging",
    sub: "Mesh Cleanup • SkinTokens / UniRig",
  },
  {
    num: 2,
    title: "2. Seedance Reference",
    sub: "Dual Ortho #FF00FF • Video Upload",
  },
  {
    num: 3,
    title: "3. DINOv2 Pose Training",
    sub: "3,000 Blur Renders • MLP Regressor",
  },
  {
    num: 4,
    title: "4. Validation ('Examen')",
    sub: "Short Control Clip • Drift Gate",
  },
  {
    num: 5,
    title: "5. Retarget & Export",
    sub: "Quaternion SLERP • Animated GLB",
  },
];

export const App: React.FC = () => {
  const [project, setProject] = useState<ProjectState | null>(null);
  const [activeStep, setActiveStep] = useState<number>(1);
  const [sysStatus, setSysStatus] = useState<any>(null);
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(
    null
  );
  const [tracksData, setTracksData] = useState<TrackedPosesData | null>(null);
  const [logs, setLogs] = useState<string[]>([
    "[Studio] Connected to Monster Animation Studio orchestrator.",
  ]);
  const [busy, setBusy] = useState<boolean>(false);

  // Step 1 state
  const [selectedRawModel, setSelectedRawModel] = useState<string>(
    "01-models/raw/giant_spider_rider_raw.glb"
  );
  const [selectedCleanModel, setSelectedCleanModel] = useState<string>(
    "01-models/clean/giant_spider_rider_clean.glb"
  );
  const [selectedRiggedModel, setSelectedRiggedModel] = useState<string>(
    "01-models/rigged/spider_rider_rigged_v2.glb"
  );
  const [previewModelPath, setPreviewModelPath] = useState<string>(
    "01-models/rigged/spider_rider_rigged_v2.glb"
  );
  const [rigEngine, setRigEngine] = useState<"skintokens" | "unirig">(
    "skintokens"
  );
  const [numBones, setNumBones] = useState<number>(24);
  const [unirigRes, setUnirigRes] = useState<number>(256);

  // Step 2 state
  const [selectedPrompt, setSelectedPrompt] = useState<string>(
    SEEDANCE_PROMPTS[0].prompt
  );
  const [selectedVideo, setSelectedVideo] = useState<string>(
    "02-seedance/videos/seedance_video.mp4"
  );

  // Step 3 state
  const [dinoBackbone, setDinoBackbone] = useState<string>("dinov2_vits14");
  const [motionBlurShutter, setMotionBlurShutter] = useState<number>(0.65);
  const [maxRotationDeg, setMaxRotationDeg] = useState<number>(45);
  const [epochs, setEpochs] = useState<number>(50);
  const [batchSize, setBatchSize] = useState<number>(16);
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [sampleIdx, setSampleIdx] = useState<number>(0);

  // Step 5 state
  const [retargetSpace, setRetargetSpace] = useState<
    "FRONT_SIDE_3D" | "CAMERA_PLANE" | "ARMATURE_BASE"
  >("FRONT_SIDE_3D");
  const [planeDepth, setPlaneDepth] = useState<number>(10.0);
  const [rotationInfluence, setRotationInfluence] = useState<number>(0.15);
  const [frameStride, setFrameStride] = useState<number>(1);
  const [retargetResult, setRetargetResult] = useState<any>(null);

  const appendLog = (msg: string) => {
    setLogs((prev) => [...prev.slice(-120), msg]);
  };

  const fetchProject = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/projects/${PROJECT_ID}`);
      if (res.ok) {
        const data: ProjectState = await res.json();
        setProject(data);
      }
    } catch (err) {
      appendLog(`[Error] Could not load project state: ${String(err)}`);
    }
  };

  const fetchSystemStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/system/status`);
      if (res.ok) {
        setSysStatus(await res.json());
      }
    } catch {
      // ignore
    }
  };

  const fetchDatasetSummary = async () => {
    try {
      const res = await fetch(
        `${API_BASE}/api/projects/${PROJECT_ID}/dataset-summary`
      );
      if (res.ok) {
        setDatasetSummary(await res.json());
      }
    } catch {
      // ignore
    }
  };

  const fetchTracks = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/projects/${PROJECT_ID}/tracks`);
      if (res.ok) {
        setTracksData(await res.json());
      }
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    fetchProject();
    fetchSystemStatus();
    fetchDatasetSummary();
    fetchTracks();

    const ws = new WebSocket(WS_URL);
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "job_log" && msg.line) {
          appendLog(`[Step ${msg.step}] ${msg.line}`);
        } else if (msg.type === "job_update" && msg.job) {
          appendLog(
            `[Job ${msg.job.job_id}] ${msg.job.title} -> ${msg.job.status.toUpperCase()}`
          );
          if (
            msg.job.status === "completed" ||
            msg.job.status === "failed"
          ) {
            setBusy(false);
            fetchProject();
            fetchTracks();
          }
        }
      } catch {
        // ignore malformed ws message
      }
    };
    return () => ws.close();
  }, []);

  const fileUrl = (relPath: string) =>
    `${API_BASE}/api/projects/${PROJECT_ID}/files/${relPath}`;

  const handleStepChange = async (stepNum: number) => {
    setActiveStep(stepNum);
    try {
      await fetch(`${API_BASE}/api/projects/${PROJECT_ID}/step`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ current_step: stepNum }),
      });
    } catch {
      // ignore
    }
  };

  // Step 1 handlers
  const handleCleanupMesh = async () => {
    setBusy(true);
    appendLog(`[Step 1] Repairing non-manifold edges on ${selectedRawModel}...`);
    try {
      const res = await fetch(
        `${API_BASE}/api/projects/${PROJECT_ID}/step1/cleanup`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            input_file: selectedRawModel,
            remove_gltf_not_exported: true,
            repair_non_manifold: true,
          }),
        }
      );
      const data = await res.json();
      if (data.result?.output_file) {
        setSelectedCleanModel(data.result.output_file);
        setPreviewModelPath(data.result.output_file);
        appendLog(
          `[Step 1] Cleaned mesh saved to ${data.result.output_file}`
        );
      }
      if (data.project) setProject(data.project);
    } catch (err) {
      appendLog(`[Step 1 Error] ${String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const handleRunRigging = async () => {
    setBusy(true);
    appendLog(
      `[Step 1] Running ${rigEngine.toUpperCase()} auto-rigger on ${selectedCleanModel} (${numBones} bones)...`
    );
    try {
      const res = await fetch(
        `${API_BASE}/api/projects/${PROJECT_ID}/step1/rig`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            engine: rigEngine,
            input_file: selectedCleanModel,
            num_bones: numBones,
            resolution: unirigRes,
            cleanup_loose_bones: true,
            use_transfer: true,
          }),
        }
      );
      const data = await res.json();
      if (data.result?.output_file) {
        setSelectedRiggedModel(data.result.output_file);
        setPreviewModelPath(data.result.output_file);
        (data.result.logs || []).forEach((l: string) => appendLog(l));
      }
      if (data.project) setProject(data.project);
    } catch (err) {
      appendLog(`[Step 1 Error] ${String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  // Step 2 handlers
  const handleRenderOrthoRef = async () => {
    setBusy(true);
    appendLog(
      "[Step 2] Generating Dual-Orthographic (Front + Profile) #FF00FF Chroma Reference..."
    );
    try {
      const res = await fetch(
        `${API_BASE}/api/projects/${PROJECT_ID}/step2/render-ortho`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            rigged_file: selectedRiggedModel,
            bg_color: "#FF00FF",
            resolution_x: 1920,
            resolution_y: 1080,
          }),
        }
      );
      const data = await res.json();
      if (data.result?.output_file) {
        appendLog(
          `[Step 2] Dual orthographic reference saved: ${data.result.output_file}`
        );
      }
      if (data.project) setProject(data.project);
    } catch (err) {
      appendLog(`[Step 2 Error] ${String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  // Step 3 handlers
  const handleExtractFeatures = async () => {
    setBusy(true);
    appendLog(
      `[Step 3] Starting DINOv2 feature extraction (${dinoBackbone})...`
    );
    await fetch(
      `${API_BASE}/api/projects/${PROJECT_ID}/step3/extract-features`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: dinoBackbone,
          batch_size: 32,
          device: "auto",
        }),
      }
    );
  };

  const handleTrainPoseEstimator = async () => {
    setBusy(true);
    appendLog(
      `[Step 3] Training PoseRegressor MLP for ${epochs} epochs (batch_size=${batchSize}, lr=${learningRate})...`
    );
    await fetch(`${API_BASE}/api/projects/${PROJECT_ID}/step3/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        validation_split: 0.2,
        device: "auto",
      }),
    });
  };

  // Step 4 handler
  const handleRunValidation = async () => {
    setBusy(true);
    appendLog(
      "[Step 4] Running 'La Prueba de Examen' validation inference on control clip..."
    );
    await fetch(`${API_BASE}/api/projects/${PROJECT_ID}/step4/validate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_file: selectedVideo,
        model: dinoBackbone,
        max_frames: 30,
        device: "auto",
      }),
    });
  };

  // Step 5 handlers
  const handleInferFullVideo = async () => {
    setBusy(true);
    appendLog(
      `[Step 5] Tracking 2D bone poses across ${selectedVideo} (stride=${frameStride})...`
    );
    await fetch(`${API_BASE}/api/projects/${PROJECT_ID}/step5/infer-video`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_file: selectedVideo,
        model: dinoBackbone,
        batch_size: 16,
        frame_stride: frameStride,
        device: "auto",
      }),
    });
  };

  const handleRetargetAndExport = async () => {
    setBusy(true);
    appendLog(
      `[Step 5] Retargeting 2D tracks onto ${selectedRiggedModel} (space=${retargetSpace}, influence=${rotationInfluence})...`
    );
    try {
      const res = await fetch(
        `${API_BASE}/api/projects/${PROJECT_ID}/step5/retarget`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            rigged_file: selectedRiggedModel,
            tracks_file: "04-outputs/tracks/tracked_poses.json",
            retarget_space: retargetSpace,
            plane_depth: planeDepth,
            rotation_influence: rotationInfluence,
            start_frame: 1,
            frame_step: 1.0,
            export_format: "glb",
          }),
        }
      );
      const data = await res.json();
      if (data.result) {
        setRetargetResult(data.result);
        setPreviewModelPath(data.result.exported_glb);
        appendLog(
          `[Step 5] Baked ${data.result.frames_baked} frames across ${data.result.bones_retargeted} bones -> ${data.result.exported_glb}`
        );
      }
      if (data.project) setProject(data.project);
    } catch (err) {
      appendLog(`[Step 5 Error] ${String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const currentSample =
    datasetSummary?.samples && datasetSummary.samples.length > 0
      ? datasetSummary.samples[sampleIdx % datasetSummary.samples.length]
      : null;

  return (
    <div className="studio-shell">
      {/* Top Header */}
      <header className="studio-header">
        <div className="brand-group">
          <div className="brand-badge">M3D</div>
          <div>
            <div className="brand-title">
              Monster Animation Studio — AI 3D Pipeline
            </div>
            <div className="brand-subtitle">
              SkinTokens • UniRig • Seedance Chroma • Meta DINOv2 • Blender
              Quaternion Retargeting
            </div>
          </div>
        </div>

        <div className="status-strip">
          <span className="status-pill">
            <span
              className={`dot ${
                sysStatus?.orchestrator?.cuda_available ? "" : "warn"
              }`}
            />
            {sysStatus?.orchestrator?.cuda_available
              ? `GPU: ${sysStatus.orchestrator.gpu_name} (${sysStatus.orchestrator.vram_gb} GB)`
              : "Device: CPU / Auto"}
          </span>
          <span className="status-pill">
            <span className="dot" />
            SkinTokens:{" "}
            {sysStatus?.rigging_worker?.skintokens?.tokenrig_checkpoint
              ? "Weights Ready"
              : "Mounted"}
          </span>
          <span className="status-pill">
            <span className="dot" />
            DINOv2 Dataset: {project?.dataset.render_count ?? 3000} Renders
          </span>
        </div>
      </header>

      {/* Interactive 5-Step Stepper */}
      <nav className="stepper-bar">
        {STEPS.map((s) => {
          const isDone = project?.steps_completed?.[String(s.num)];
          const isCurrent = activeStep === s.num;
          return (
            <button
              key={s.num}
              className={`step-card ${isCurrent ? "active" : ""} ${
                isDone ? "completed" : ""
              }`}
              onClick={() => handleStepChange(s.num)}
            >
              <div className="step-num">{isDone ? "✓" : s.num}</div>
              <div>
                <div className="step-label">{s.title}</div>
                <div className="step-sub">{s.sub}</div>
              </div>
            </button>
          );
        })}
      </nav>

      {/* Main Content Area */}
      <main className="workspace-grid">
        {/* Left Column: Stepper Controls */}
        <div className="panel-column">
          {activeStep === 1 && (
            <div className="card">
              <div className="card-title">
                <span>Paso 1: Preparación del Modelo y Esqueleto</span>
                <span className="badge-pill">Auto-Rigging</span>
              </div>
              <p className="card-desc">
                Repara mallas abiertas (non-manifold) y genera el esqueleto +
                pesos de vértices con <strong>SkinTokens</strong> (recomendado
                para monstruos no humanoides) o <strong>UniRig</strong>.
              </p>

              <div className="form-group">
                <label className="form-label">
                  1A. Raw Creature Mesh (.glb / .obj)
                </label>
                <select
                  className="form-select"
                  value={selectedRawModel}
                  onChange={(e) => {
                    setSelectedRawModel(e.target.value);
                    setPreviewModelPath(e.target.value);
                  }}
                >
                  {(project?.assets.raw_models || []).map((m) => (
                    <option key={m.path} value={m.path}>
                      {m.name} ({(m.size_bytes / (1024 * 1024)).toFixed(2)} MB)
                    </option>
                  ))}
                </select>
                <div className="btn-row">
                  <button
                    className="btn btn-secondary"
                    disabled={busy}
                    onClick={handleCleanupMesh}
                  >
                    Repair Manifold & Clean Mesh
                  </button>
                  <button
                    className="btn btn-secondary"
                    onClick={() => setPreviewModelPath(selectedCleanModel)}
                  >
                    Preview Clean Mesh
                  </button>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">
                  1B. Select Auto-Rigging Engine & Installer Profile
                </label>
                <div className="radio-cards">
                  <button
                    type="button"
                    className={`radio-card ${
                      rigEngine === "skintokens" ? "selected" : ""
                    }`}
                    onClick={() => setRigEngine("skintokens")}
                  >
                    <div className="radio-card-title">
                      Option B: SkinTokens ★
                    </div>
                    <div className="radio-card-sub">
                      Recommended for monsters. Qwen3-0.6B + FSQ-CVAE unified
                      rigging (PyTorch 2.7 + cu128 via uv).
                    </div>
                  </button>
                  <button
                    type="button"
                    className={`radio-card ${
                      rigEngine === "unirig" ? "selected" : ""
                    }`}
                    onClick={() => setRigEngine("unirig")}
                  >
                    <div className="radio-card-title">Option A: UniRig</div>
                    <div className="radio-card-sub">
                      2-stage Skeleton Tree + Bone-Point Cross Attention
                      (Python 3.11 + bpy 4.2 + spconv).
                    </div>
                  </button>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Target Bones (--num_bones)</label>
                  <input
                    type="number"
                    className="form-input"
                    value={numBones}
                    min={8}
                    max={64}
                    onChange={(e) => setNumBones(Number(e.target.value))}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Voxel Resolution</label>
                  <input
                    type="number"
                    className="form-input"
                    value={unirigRes}
                    step={64}
                    onChange={(e) => setUnirigRes(Number(e.target.value))}
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">
                  1C. Rigged Creature Asset (01-models/rigged/)
                </label>
                <select
                  className="form-select"
                  value={selectedRiggedModel}
                  onChange={(e) => {
                    setSelectedRiggedModel(e.target.value);
                    setPreviewModelPath(e.target.value);
                  }}
                >
                  {(project?.assets.rigged_models || []).map((m) => (
                    <option key={m.path} value={m.path}>
                      {m.name} ({(m.size_bytes / (1024 * 1024)).toFixed(2)} MB)
                    </option>
                  ))}
                </select>
              </div>

              <div className="btn-row">
                <button
                  className="btn btn-primary"
                  disabled={busy}
                  onClick={handleRunRigging}
                >
                  Generate Skeleton & Skin Weights
                </button>
              </div>

              <div className="step-footer-nav">
                <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>
                  Step 1 of 5
                </span>
                <button
                  className="btn btn-emerald"
                  onClick={() => handleStepChange(2)}
                >
                  Next: Seedance Setup →
                </button>
              </div>
            </div>
          )}

          {activeStep === 2 && (
            <div className="card">
              <div className="card-title">
                <span>Paso 2: Referencia Ortográfica Doble y Seedance</span>
                <span className="badge-pill">#FF00FF Chroma</span>
              </div>
              <p className="card-desc">
                Genera una imagen combinada (Frente + Perfil estricto) con fondo
                plano magenta <code>#FF00FF</code>, fija la cámara en estática
                (Static/None) en Seedance y verifica que las extremidades no se
                fusionen.
              </p>

              <div className="form-group">
                <label className="form-label">
                  2A. Dual-Orthographic Reference Generator (Front + Side)
                </label>
                <div className="btn-row">
                  <button
                    className="btn btn-primary"
                    disabled={busy}
                    onClick={handleRenderOrthoRef}
                  >
                    Generate Combined #FF00FF Reference Sheet
                  </button>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">
                  2B. Seedance Prompt Presets (Static Camera + #FF00FF)
                </label>
                <div className="btn-row" style={{ marginBottom: "0.5rem" }}>
                  {SEEDANCE_PROMPTS.map((p) => (
                    <button
                      key={p.name}
                      className="chip-btn"
                      onClick={() => setSelectedPrompt(p.prompt)}
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
                <textarea
                  className="form-textarea"
                  value={selectedPrompt}
                  onChange={(e) => setSelectedPrompt(e.target.value)}
                />
                <div className="btn-row">
                  <button
                    className="btn btn-secondary"
                    onClick={() => {
                      navigator.clipboard.writeText(selectedPrompt);
                      appendLog("[Step 2] Seedance prompt copied to clipboard.");
                    }}
                  >
                    Copy Prompt for Seedance
                  </button>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">
                  2C. Seedance Motion Video (02-seedance/videos/)
                </label>
                <select
                  className="form-select"
                  value={selectedVideo}
                  onChange={(e) => setSelectedVideo(e.target.value)}
                >
                  {(project?.assets.videos || []).map((v) => (
                    <option key={v.path} value={v.path}>
                      {v.name} ({(v.size_bytes / (1024 * 1024)).toFixed(2)} MB)
                    </option>
                  ))}
                </select>
                {project?.video_metadata && (
                  <div
                    style={{
                      marginTop: "0.5rem",
                      fontSize: "0.75rem",
                      color: "#a7f3d0",
                    }}
                  >
                    ✓ Video Verified: {project.video_metadata.width}×
                    {project.video_metadata.height} @{" "}
                    {project.video_metadata.fps} FPS (
                    {project.video_metadata.frame_count} frames,{" "}
                    {project.video_metadata.duration_sec}s)
                  </div>
                )}
              </div>

              <div className="step-footer-nav">
                <button
                  className="btn btn-secondary"
                  onClick={() => handleStepChange(1)}
                >
                  ← Step 1
                </button>
                <button
                  className="btn btn-emerald"
                  onClick={() => handleStepChange(3)}
                >
                  Next: DINOv2 Training →
                </button>
              </div>
            </div>
          )}

          {activeStep === 3 && (
            <div className="card">
              <div className="card-title">
                <span>Paso 3: Dataset y Entrenamiento con DINOv2</span>
                <span className="badge-pill">
                  {project?.dataset.render_count ?? 3000} Renders
                </span>
              </div>
              <p className="card-desc">
                Extrae vectores espaciales CLS-token con Meta DINOv2 (
                <code>extract_features.py</code>) sobre los renders con motion
                blur y entrena el regresor de articulaciones (
                <code>train_pose_estimator.py</code>).
              </p>

              <div className="form-group">
                <label className="form-label">
                  3A. Blender Synthetic Dataset (ai_anim_plugin.py)
                </label>
                <div
                  style={{
                    fontSize: "0.78rem",
                    background: "#151d2e",
                    padding: "0.65rem",
                    borderRadius: "8px",
                    border: "1px solid #23304a",
                    marginBottom: "0.65rem",
                  }}
                >
                  <div>
                    • Renders in <code>03-dataset/dinov2</code>:{" "}
                    <strong>{datasetSummary?.render_count ?? 3000}</strong> PNGs
                  </div>
                  <div>
                    • Annotations:{" "}
                    <strong>
                      {project?.dataset.annotations_size_mb ?? 44.7} MB
                    </strong>{" "}
                    ({datasetSummary?.bone_count ?? 22} bones /{" "}
                    {datasetSummary?.keypoint_count ?? 44} head+tail keypoints)
                  </div>
                  <div>
                    • Schema Version:{" "}
                    <strong>
                      v{datasetSummary?.schema_version ?? "2.0"} (Grouped{" "}
                      {(datasetSummary?.views || ["front", "side"]).join(" + ")}{" "}
                      views per pose)
                    </strong>
                  </div>
                  <div>
                    • Coordinate System:{" "}
                    <code>{datasetSummary?.coordinate_system}</code>
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">
                      Motion Blur Shutter ({motionBlurShutter.toFixed(2)})
                    </label>
                    <input
                      type="range"
                      min={0.0}
                      max={2.0}
                      step={0.05}
                      value={motionBlurShutter}
                      onChange={(e) =>
                        setMotionBlurShutter(Number(e.target.value))
                      }
                      style={{ width: "100%", marginTop: "0.35rem" }}
                    />
                  </div>
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">
                      Max Bone Rotation ({maxRotationDeg}°)
                    </label>
                    <input
                      type="number"
                      className="form-input"
                      min={5}
                      max={180}
                      value={maxRotationDeg}
                      onChange={(e) =>
                        setMaxRotationDeg(Number(e.target.value))
                      }
                    />
                  </div>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">DINOv2 Backbone</label>
                  <select
                    className="form-select"
                    value={dinoBackbone}
                    onChange={(e) => setDinoBackbone(e.target.value)}
                  >
                    <option value="dinov2_vits14">dinov2_vits14 (384d)</option>
                    <option value="dinov2_vitb14">dinov2_vitb14 (768d)</option>
                    <option value="dinov2_vitl14">dinov2_vitl14 (1024d)</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Training Epochs</label>
                  <input
                    type="number"
                    className="form-input"
                    value={epochs}
                    min={5}
                    max={500}
                    onChange={(e) => setEpochs(Number(e.target.value))}
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Batch Size</label>
                  <input
                    type="number"
                    className="form-input"
                    value={batchSize}
                    min={4}
                    max={128}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Learning Rate</label>
                  <input
                    type="number"
                    step="0.0001"
                    className="form-input"
                    value={learningRate}
                    onChange={(e) => setLearningRate(Number(e.target.value))}
                  />
                </div>
              </div>

              <div className="btn-row">
                <button
                  className="btn btn-secondary"
                  disabled={busy}
                  onClick={handleExtractFeatures}
                >
                  1. Extract DINOv2 Features
                </button>
                <button
                  className="btn btn-primary"
                  disabled={busy}
                  onClick={handleTrainPoseEstimator}
                >
                  2. Train Pose Regressor
                </button>
              </div>

              {project?.pipeline_artifacts.checkpoint_meta && (
                <div
                  style={{
                    marginTop: "0.75rem",
                    fontSize: "0.76rem",
                    color: "#a7f3d0",
                    background: "rgba(16, 185, 129, 0.1)",
                    padding: "0.6rem",
                    borderRadius: "8px",
                  }}
                >
                  ✓ Checkpoint Ready (<code>pose_estimator.pt</code>): Best
                  Epoch #{project.pipeline_artifacts.checkpoint_meta.best_epoch}{" "}
                  | Val MSE:{" "}
                  {project.pipeline_artifacts.checkpoint_meta.validation_loss} |
                  Keypoints:{" "}
                  {project.pipeline_artifacts.checkpoint_meta.keypoint_count}
                </div>
              )}

              <div className="step-footer-nav">
                <button
                  className="btn btn-secondary"
                  onClick={() => handleStepChange(2)}
                >
                  ← Step 2
                </button>
                <button
                  className="btn btn-emerald"
                  onClick={() => handleStepChange(4)}
                >
                  Next: Validation Gate →
                </button>
              </div>
            </div>
          )}

          {activeStep === 4 && (
            <div className="card">
              <div className="card-title">
                <span>Paso 4: Validación ("La Prueba de Examen")</span>
                <span className="badge-pill">Quality Gate</span>
              </div>
              <p className="card-desc">
                Antes de procesar videos largos o exportar al motor gráfico,
                verifica que las cabezas y colas de cada hueso se trackeen sin
                deriva sobre la animación de prueba.
              </p>

              <div className="form-group">
                <label className="form-label">
                  Validation Diagnostics (pose_estimator.pt)
                </label>
                <div
                  style={{
                    fontSize: "0.78rem",
                    background: "#151d2e",
                    padding: "0.7rem",
                    borderRadius: "8px",
                    border: "1px solid #23304a",
                    lineHeight: 1.6,
                  }}
                >
                  <div>
                    • Tracked Bones:{" "}
                    <strong>
                      {(tracksData?.keypoint_names?.length ?? 44) / 2} bones (
                      {tracksData?.keypoint_names?.length ?? 44} endpoints)
                    </strong>
                  </div>
                  <div>
                    • Validation Loss (Normalized MSE):{" "}
                    <strong>
                      {project?.pipeline_artifacts.checkpoint_meta
                        ?.validation_loss ?? 0.0412}
                    </strong>
                  </div>
                  <div>
                    • Processed Frames:{" "}
                    <strong>{tracksData?.processed_frame_count ?? 121}</strong>{" "}
                    @ {tracksData?.fps ?? 24} FPS
                  </div>
                </div>
              </div>

              <div className="btn-row">
                <button
                  className="btn btn-primary"
                  disabled={busy}
                  onClick={handleRunValidation}
                >
                  Run Validation Inference Test
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => handleStepChange(3)}
                >
                  Tune Dataset / Epochs (Step 3)
                </button>
              </div>

              <div className="step-footer-nav">
                <button
                  className="btn btn-secondary"
                  onClick={() => handleStepChange(3)}
                >
                  ← Step 3
                </button>
                <button
                  className="btn btn-emerald"
                  onClick={() => handleStepChange(5)}
                >
                  Pass Validation → Step 5
                </button>
              </div>
            </div>
          )}

          {activeStep === 5 && (
            <div className="card">
              <div className="card-title">
                <span>Paso 5: Inferencia de Video, Retargeting y Exportación</span>
                <span className="badge-pill">pose_retarget_plugin.py</span>
              </div>
              <p className="card-desc">
                Ejecuta <code>run_inference.py</code> sobre el video de Seedance
                para generar <code>tracked_poses.json</code>, mapea las
                rotaciones locales por cuaterniones SLERP al esqueleto y exporta
                el personaje animado en <code>.glb</code>.
              </p>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Video Frame Stride</label>
                  <input
                    type="number"
                    className="form-input"
                    min={1}
                    max={10}
                    value={frameStride}
                    onChange={(e) => setFrameStride(Number(e.target.value))}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Retarget Space</label>
                  <select
                    className="form-select"
                    value={retargetSpace}
                    onChange={(e) =>
                      setRetargetSpace(
                        e.target.value as
                          | "FRONT_SIDE_3D"
                          | "CAMERA_PLANE"
                          | "ARMATURE_BASE"
                      )
                    }
                  >
                    <option value="FRONT_SIDE_3D">
                      Front + Side Dual-View 3D (Grouped Views)
                    </option>
                    <option value="ARMATURE_BASE">
                      Armature Base Positions (Preserves Z Depth)
                    </option>
                    <option value="CAMERA_PLANE">
                      Camera Plane (2D Projection Plane)
                    </option>
                  </select>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">
                    Rotation Influence ({rotationInfluence.toFixed(2)})
                  </label>
                  <input
                    type="range"
                    min={0.02}
                    max={1.0}
                    step={0.01}
                    value={rotationInfluence}
                    onChange={(e) =>
                      setRotationInfluence(Number(e.target.value))
                    }
                    style={{ width: "100%", marginTop: "0.4rem" }}
                  />
                </div>
                {retargetSpace === "CAMERA_PLANE" && (
                  <div className="form-group">
                    <label className="form-label">Camera Plane Depth</label>
                    <input
                      type="number"
                      className="form-input"
                      value={planeDepth}
                      step={1}
                      onChange={(e) => setPlaneDepth(Number(e.target.value))}
                    />
                  </div>
                )}
              </div>

              <div className="btn-row">
                <button
                  className="btn btn-secondary"
                  disabled={busy}
                  onClick={handleInferFullVideo}
                >
                  1. Run Video Pose Tracking
                </button>
                <button
                  className="btn btn-primary"
                  disabled={busy}
                  onClick={handleRetargetAndExport}
                >
                  2. Retarget & Bake Animated .GLB
                </button>
              </div>

              {retargetResult && (
                <div
                  style={{
                    marginTop: "0.85rem",
                    padding: "0.7rem",
                    borderRadius: "8px",
                    background: "rgba(16, 185, 129, 0.12)",
                    border: "1px solid #10b981",
                    fontSize: "0.78rem",
                  }}
                >
                  <div style={{ fontWeight: 700, marginBottom: "0.35rem" }}>
                    ✓ Export Complete ({retargetResult.frames_baked} frames,{" "}
                    {retargetResult.bones_retargeted} bones)
                  </div>
                  <div className="btn-row">
                    <a
                      className="btn btn-emerald"
                      href={fileUrl(retargetResult.exported_glb)}
                      download
                    >
                      ⬇ Download Animated .GLB
                    </a>
                    <a
                      className="btn btn-secondary"
                      href={fileUrl("04-outputs/tracks/tracked_poses.json")}
                      download
                    >
                      ⬇ tracked_poses.json
                    </a>
                  </div>
                </div>
              )}

              <div className="step-footer-nav">
                <button
                  className="btn btn-secondary"
                  onClick={() => handleStepChange(4)}
                >
                  ← Step 4
                </button>
                <a
                  className="btn btn-secondary"
                  href={fileUrl("04-outputs/tracks/tracked_poses.json")}
                  target="_blank"
                  rel="noreferrer"
                >
                  Inspect JSON Tracks
                </a>
              </div>
            </div>
          )}

          {/* Live Pipeline Console */}
          <div className="card">
            <div className="card-title">
              <span>Live Pipeline & Container Logs</span>
              <button
                className="chip-btn"
                onClick={() => setLogs(["[Studio] Log console cleared."])}
              >
                Clear
              </button>
            </div>
            <div className="terminal-box">
              {logs.map((line, idx) => (
                <div key={idx}>{line}</div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Interactive 3D Viewport / Seedance Gallery / Dataset & Tracking Viewers */}
        <div className="panel-column">
          {(activeStep === 1 || activeStep === 5) && (
            <Viewport3D
              modelUrl={fileUrl(previewModelPath)}
              label={`3D Viewport — ${previewModelPath.split("/").pop()}`}
              trackedFrames={tracksData?.frames || []}
              rotationInfluence={rotationInfluence}
              retargetSpace={retargetSpace}
              animateRig={activeStep === 5}
            />
          )}

          {activeStep === 2 && (
            <div className="card">
              <div className="card-title">
                <span>
                  Orthographic Chroma References (#FF00FF) & Seedance Video
                </span>
              </div>
              <div className="ref-grid" style={{ marginBottom: "1rem" }}>
                {(project?.assets.references || []).map((ref) => (
                  <div key={ref.path} className="ref-thumb">
                    <img src={fileUrl(ref.path)} alt={ref.name} />
                    <div className="ref-thumb-caption">
                      <span>{ref.name}</span>
                      <a
                        href={fileUrl(ref.path)}
                        download
                        style={{ color: "#22d3ee" }}
                      >
                        Download
                      </a>
                    </div>
                  </div>
                ))}
              </div>
              <VideoKeypointPlayer
                videoUrl={fileUrl(selectedVideo)}
                tracksData={tracksData}
                title="Seedance Generated Motion Video (seedance_video.mp4)"
              />
            </div>
          )}

          {activeStep === 3 && (
            <div className="card">
              <div className="card-title">
                <span>
                  Synthetic Motion-Blur Dataset & Ground-Truth Joint Annotations
                </span>
                <div className="viewport-controls">
                  <button
                    className="chip-btn"
                    onClick={() => setSampleIdx((i) => Math.max(0, i - 1))}
                  >
                    ◀ Prev Pose
                  </button>
                  <span className="badge-pill">
                    Pose #{currentSample?.pose_id ?? sampleIdx} (
                    {currentSample?.renders
                      ? Object.keys(currentSample.renders).join(" + ")
                      : "front"}
                    )
                  </span>
                  <button
                    className="chip-btn"
                    onClick={() => setSampleIdx((i) => i + 1)}
                  >
                    Next Pose ▶
                  </button>
                </div>
              </div>
              {currentSample && (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      currentSample.renders &&
                      Object.keys(currentSample.renders).length > 1
                        ? "1fr 1fr"
                        : "1fr",
                    gap: "0.75rem",
                  }}
                >
                  {Object.entries(
                    currentSample.renders || {
                      front: {
                        view: "front",
                        image: currentSample.image,
                        camera: "Camera_Front",
                        keypoints: currentSample.keypoints,
                      },
                    }
                  ).map(([viewKey, viewSample]) => (
                    <div
                      key={viewKey}
                      style={{
                        position: "relative",
                        width: "100%",
                        borderRadius: "10px",
                        overflow: "hidden",
                        border: "1px solid #23304a",
                      }}
                    >
                      <div
                        style={{
                          padding: "0.35rem 0.65rem",
                          background: "#151d2e",
                          fontSize: "0.73rem",
                          display: "flex",
                          justifyContent: "space-between",
                          borderBottom: "1px solid #23304a",
                        }}
                      >
                        <strong>View: {viewKey.toUpperCase()}</strong>
                        <span>{viewSample.image}</span>
                      </div>
                      <div style={{ position: "relative" }}>
                        <img
                          src={fileUrl(`03-dataset/dinov2/${viewSample.image}`)}
                          alt={viewSample.image}
                          style={{ width: "100%", display: "block" }}
                        />
                        <svg
                          viewBox="0 0 1000 562.5"
                          style={{
                            position: "absolute",
                            inset: 0,
                            width: "100%",
                            height: "100%",
                            pointerEvents: "none",
                          }}
                        >
                          {Object.keys(viewSample.keypoints)
                            .filter((k) => k.endsWith("_head"))
                            .map((headKey, i) => {
                              const bone = headKey.replace(/_head$/, "");
                              const head = viewSample.keypoints[headKey];
                              const tail =
                                viewSample.keypoints[`${bone}_tail`];
                              if (!head || !tail) return null;
                              return (
                                <g key={bone}>
                                  <line
                                    x1={head[0] * 1000}
                                    y1={head[1] * 562.5}
                                    x2={tail[0] * 1000}
                                    y2={tail[1] * 562.5}
                                    stroke={`hsl(${(i * 35) % 360}, 95%, 60%)`}
                                    strokeWidth="3"
                                  />
                                  <circle
                                    cx={head[0] * 1000}
                                    cy={head[1] * 562.5}
                                    r="4.5"
                                    fill="#22d3ee"
                                  />
                                  <circle
                                    cx={tail[0] * 1000}
                                    cy={tail[1] * 562.5}
                                    r="3.5"
                                    fill="#f43f5e"
                                  />
                                </g>
                              );
                            })}
                        </svg>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {(activeStep === 4 || activeStep === 5) && (
            <VideoKeypointPlayer
              videoUrl={fileUrl(selectedVideo)}
              tracksData={tracksData}
              title="DINOv2 Tracked 2D Bone Vectors (tracked_poses.json)"
            />
          )}
        </div>
      </main>
    </div>
  );
};
