export interface AssetFile {
  name: string;
  path: string;
  size_bytes: number;
  modified_at: number;
}

export interface ProjectState {
  project_id: string;
  name: string;
  current_step: number;
  steps_completed: Record<string, boolean>;
  assets: {
    raw_models: AssetFile[];
    clean_models: AssetFile[];
    rigged_models: AssetFile[];
    references: AssetFile[];
    videos: AssetFile[];
    exports: AssetFile[];
  };
  video_metadata?: {
    fps: number;
    frame_count: number;
    width: number;
    height: number;
    duration_sec: number;
  } | null;
  dataset: {
    render_count: number;
    has_valid_annotations: boolean;
    annotations_size_mb: number;
  };
  pipeline_artifacts: {
    features_ready: boolean;
    features_size_mb: number;
    checkpoint_ready: boolean;
    checkpoint_meta?: {
      input_dim?: number;
      output_dim?: number;
      target_shape?: number[];
      keypoint_count?: number;
      bone_count?: number;
      best_epoch?: number;
      validation_loss?: number;
    } | null;
    tracks_ready: boolean;
    validation_ready: boolean;
  };
  training_history: Array<{
    epoch: number;
    total_epochs: number;
    train_mse: number;
    val_mse: number;
    val_rmse: number;
  }>;
}

export interface RenderViewSample {
  view: string;
  image: string;
  camera?: string;
  keypoints: Record<string, [number, number]>;
}

export interface DatasetSummary {
  schema_version?: string;
  views?: string[];
  render_count: number;
  pose_count?: number;
  has_annotations: boolean;
  annotated_count?: number;
  coordinate_system?: string;
  image_resolution?: [number, number];
  keypoint_count?: number;
  bone_count?: number;
  bone_names?: string[];
  samples: Array<{
    pose_id?: number;
    image: string;
    keypoints: Record<string, [number, number]>;
    renders?: Record<string, RenderViewSample>;
  }>;
}

export interface TrackedFrame {
  frame_index: number;
  timestamp_sec: number | null;
  keypoints: Record<string, [number, number]>;
  renders?: Record<string, { keypoints: Record<string, [number, number]> }>;
}

export interface TrackedPosesData {
  video_path: string;
  pose_model: string;
  feature_model: string;
  fps: number | null;
  frame_width: number;
  frame_height: number;
  frame_stride: number;
  processed_frame_count: number;
  target_shape: number[];
  keypoint_names: string[];
  frames: TrackedFrame[];
}
