import React, { useEffect, useRef, useState } from "react";
import { TrackedPosesData } from "../types";

interface VideoKeypointPlayerProps {
  videoUrl: string;
  tracksData: TrackedPosesData | null;
  title?: string;
}

export const VideoKeypointPlayer: React.FC<VideoKeypointPlayerProps> = ({
  videoUrl,
  tracksData,
  title = "DINOv2 2D Joint Tracking Overlay",
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showLabels, setShowLabels] = useState(false);
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);

  useEffect(() => {
    let rafId = 0;

    const renderOverlay = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (video && canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          if (
            canvas.width !== video.clientWidth ||
            canvas.height !== video.clientHeight
          ) {
            canvas.width = video.clientWidth || 640;
            canvas.height = video.clientHeight || 360;
          }
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          if (showOverlay && tracksData && tracksData.frames.length > 0) {
            const fps = tracksData.fps || 24;
            const rawFrame = Math.floor(video.currentTime * fps);
            const matched =
              tracksData.frames.find((f) => f.frame_index >= rawFrame) ||
              tracksData.frames[tracksData.frames.length - 1];

            if (matched) {
              setCurrentFrameIdx(matched.frame_index);
              const kp = matched.keypoints;
              const bonePrefixes = new Set<string>();
              Object.keys(kp).forEach((k) => {
                if (k.endsWith("_head")) {
                  bonePrefixes.add(k.replace(/_head$/, ""));
                }
              });

              let idx = 0;
              bonePrefixes.forEach((bone) => {
                const head = kp[`${bone}_head`];
                const tail = kp[`${bone}_tail`];
                if (!head || !tail) return;

                const hx = head[0] * canvas.width;
                const hy = head[1] * canvas.height;
                const tx = tail[0] * canvas.width;
                const ty = tail[1] * canvas.height;

                const hue = (idx * 37) % 360;
                ctx.strokeStyle = `hsla(${hue}, 95%, 60%, 0.9)`;
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                ctx.moveTo(hx, hy);
                ctx.lineTo(tx, ty);
                ctx.stroke();

                ctx.fillStyle = "#22d3ee";
                ctx.beginPath();
                ctx.arc(hx, hy, 3.5, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = "#f43f5e";
                ctx.beginPath();
                ctx.arc(tx, ty, 3, 0, Math.PI * 2);
                ctx.fill();

                if (showLabels) {
                  ctx.fillStyle = "#f8fafc";
                  ctx.font = "10px monospace";
                  ctx.fillText(bone, hx + 5, hy - 4);
                }
                idx++;
              });
            }
          }
        }
      }
      rafId = requestAnimationFrame(renderOverlay);
    };

    rafId = requestAnimationFrame(renderOverlay);
    return () => cancelAnimationFrame(rafId);
  }, [tracksData, showOverlay, showLabels]);

  return (
    <div className="viewport-card">
      <div className="viewport-toolbar">
        <span className="viewport-title">{title}</span>
        <div className="viewport-controls">
          <span className="badge-pill">Frame #{currentFrameIdx}</span>
          <button
            className={`chip-btn ${showOverlay ? "active" : ""}`}
            onClick={() => setShowOverlay(!showOverlay)}
          >
            Keypoint Skeleton
          </button>
          <button
            className={`chip-btn ${showLabels ? "active" : ""}`}
            onClick={() => setShowLabels(!showLabels)}
          >
            Bone Names
          </button>
        </div>
      </div>

      <div className="video-stage">
        <video
          ref={videoRef}
          src={videoUrl}
          controls
          loop
          muted
          playsInline
          crossOrigin="anonymous"
          className="tracked-video"
        />
        <canvas ref={canvasRef} className="tracked-canvas" />
      </div>
    </div>
  );
};
