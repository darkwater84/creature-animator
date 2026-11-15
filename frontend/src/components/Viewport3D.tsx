import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { TrackedFrame } from "../types";

interface Viewport3DProps {
  modelUrl: string | null;
  label?: string;
  trackedFrames?: TrackedFrame[];
  rotationInfluence?: number;
  retargetSpace?: "FRONT_SIDE_3D" | "CAMERA_PLANE" | "ARMATURE_BASE";
  animateRig?: boolean;
}

export const Viewport3D: React.FC<Viewport3DProps> = ({
  modelUrl,
  label,
  trackedFrames = [],
  rotationInfluence = 0.15,
  animateRig = false,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [wireframe, setWireframe] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [chromaBg, setChromaBg] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [isPlaying, setIsPlaying] = useState(animateRig);
  const [stats, setStats] = useState<{ vertices: number; faces: number; bones: number }>({
    vertices: 0,
    faces: 0,
    bones: 0,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sceneRef = useRef<THREE.Scene | null>(null);
  const skeletonHelperRef = useRef<THREE.SkeletonHelper | null>(null);
  const bonesRef = useRef<Map<string, { bone: THREE.Bone; restQuat: THREE.Quaternion }>>(
    new Map()
  );
  const controlsRef = useRef<OrbitControls | null>(null);
  const playingRef = useRef(isPlaying);
  const influenceRef = useRef(rotationInfluence);
  const framesRef = useRef(trackedFrames);

  useEffect(() => {
    playingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    setIsPlaying(animateRig);
  }, [animateRig]);

  useEffect(() => {
    influenceRef.current = rotationInfluence;
  }, [rotationInfluence]);

  useEffect(() => {
    framesRef.current = trackedFrames;
  }, [trackedFrames]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(chromaBg ? 0xff00ff : 0x0b0f17);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      45,
      container.clientWidth / Math.max(1, container.clientHeight),
      0.05,
      500
    );
    camera.position.set(0, 1.8, 4.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0.9, 0);
    controlsRef.current = controls;

    const ambient = new THREE.AmbientLight(0xffffff, 1.1);
    scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
    keyLight.position.set(4, 7, 5);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x7c3aed, 1.2);
    rimLight.position.set(-5, 3, -4);
    scene.add(rimLight);

    const grid = new THREE.GridHelper(10, 20, 0x334155, 0x1e293b);
    grid.name = "ViewportGrid";
    scene.add(grid);

    const clock = new THREE.Clock();
    let animId = 0;

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      if (playingRef.current && bonesRef.current.size > 0) {
        const frames = framesRef.current;
        const inf = influenceRef.current;
        if (frames && frames.length > 1) {
          const frameIdx = Math.floor(elapsed * 24) % frames.length;
          const kp0 = frames[0].keypoints;
          const kp = frames[frameIdx].keypoints;

          bonesRef.current.forEach(({ bone, restQuat }, name) => {
            const h0 = kp0[`${name}_head`];
            const t0 = kp0[`${name}_tail`];
            const h = kp[`${name}_head`];
            const t = kp[`${name}_tail`];
            if (h0 && t0 && h && t) {
              const restDir = new THREE.Vector3(t0[0] - h0[0], -(t0[1] - h0[1]), 0).normalize();
              const curDir = new THREE.Vector3(t[0] - h[0], -(t[1] - h[1]), 0).normalize();
              if (restDir.lengthSq() > 0.0001 && curDir.lengthSq() > 0.0001) {
                const deltaQ = new THREE.Quaternion().setFromUnitVectors(restDir, curDir);
                const blended = new THREE.Quaternion().slerp(deltaQ, inf);
                bone.quaternion.copy(restQuat).multiply(blended);
              }
            } else {
              // Subtle procedural breathing/motion for bones not in track map
              const wave = Math.sin(elapsed * 3.2 + name.length) * 0.08 * inf;
              const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(wave, 0, wave * 0.5));
              bone.quaternion.copy(restQuat).multiply(q);
            }
          });
        } else {
          // Procedural preview if no tracked_poses loaded yet
          let idx = 0;
          bonesRef.current.forEach(({ bone, restQuat }) => {
            const wave = Math.sin(elapsed * 2.5 + idx * 0.6) * 0.12 * inf;
            const q = new THREE.Quaternion().setFromEuler(
              new THREE.Euler(wave, wave * 0.3, -wave * 0.4)
            );
            bone.quaternion.copy(restQuat).multiply(q);
            idx++;
          });
        }
      }

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / Math.max(1, container.clientHeight);
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
    };
  }, []);

  // Load 3D model when modelUrl changes
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !modelUrl) return;

    setLoading(true);
    setError(null);

    const existingModel = scene.getObjectByName("LoadedCreature");
    if (existingModel) scene.remove(existingModel);
    if (skeletonHelperRef.current) {
      scene.remove(skeletonHelperRef.current);
      skeletonHelperRef.current = null;
    }
    bonesRef.current.clear();

    const finishSetup = (rootObj: THREE.Object3D) => {
      rootObj.name = "LoadedCreature";
      const box = new THREE.Box3().setFromObject(rootObj);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z, 0.001);
      const scale = 2.2 / maxDim;
      rootObj.scale.setScalar(scale);
      rootObj.position.set(-center.x * scale, -box.min.y * scale, -center.z * scale);

      let vCount = 0;
      let fCount = 0;
      let bCount = 0;

      rootObj.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          const geom = mesh.geometry;
          if (geom.attributes.position) {
            vCount += geom.attributes.position.count;
            fCount += geom.index
              ? geom.index.count / 3
              : geom.attributes.position.count / 3;
          }
          if (!mesh.material) {
            mesh.material = new THREE.MeshStandardMaterial({
              color: 0x94a3b8,
              roughness: 0.45,
              metalness: 0.15,
            });
          }
        }
        if ((child as THREE.Bone).isBone) {
          const bone = child as THREE.Bone;
          bCount++;
          bonesRef.current.set(bone.name, {
            bone,
            restQuat: bone.quaternion.clone(),
          });
        }
      });

      scene.add(rootObj);

      if (bCount > 0) {
        const skelHelper = new THREE.SkeletonHelper(rootObj);
        (skelHelper.material as THREE.LineBasicMaterial).linewidth = 2;
        skelHelper.visible = showSkeleton;
        scene.add(skelHelper);
        skeletonHelperRef.current = skelHelper;
      }

      setStats({
        vertices: Math.round(vCount),
        faces: Math.round(fCount),
        bones: bCount,
      });
      setLoading(false);
    };

    const isObj = modelUrl.toLowerCase().endsWith(".obj");
    if (isObj) {
      const objLoader = new OBJLoader();
      objLoader.load(
        modelUrl,
        (obj) => finishSetup(obj),
        undefined,
        (err) => {
          setError(`Failed to load OBJ: ${String(err)}`);
          setLoading(false);
        }
      );
    } else {
      const gltfLoader = new GLTFLoader();
      gltfLoader.load(
        modelUrl,
        (gltf) => finishSetup(gltf.scene),
        undefined,
        (err) => {
          setError(`Failed to load GLB: ${String(err)}`);
          setLoading(false);
        }
      );
    }
  }, [modelUrl]);

  // Update wireframe / skeleton / chroma background toggles
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    scene.background = new THREE.Color(chromaBg ? 0xff00ff : 0x0b0f17);
    const grid = scene.getObjectByName("ViewportGrid");
    if (grid) grid.visible = !chromaBg;

    const rootObj = scene.getObjectByName("LoadedCreature");
    if (rootObj) {
      rootObj.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mat = (child as THREE.Mesh).material;
          if (Array.isArray(mat)) {
            mat.forEach((m) => ((m as THREE.MeshStandardMaterial).wireframe = wireframe));
          } else if (mat) {
            (mat as THREE.MeshStandardMaterial).wireframe = wireframe;
          }
        }
      });
    }
    if (skeletonHelperRef.current) {
      skeletonHelperRef.current.visible = showSkeleton;
    }
    if (controlsRef.current) {
      controlsRef.current.autoRotate = autoRotate;
    }
  }, [wireframe, showSkeleton, chromaBg, autoRotate]);

  return (
    <div className="viewport-card">
      <div className="viewport-toolbar">
        <span className="viewport-title">{label || "3D Creature & Armature Viewport"}</span>
        <div className="viewport-controls">
          <button
            className={`chip-btn ${wireframe ? "active" : ""}`}
            onClick={() => setWireframe(!wireframe)}
          >
            Wireframe
          </button>
          <button
            className={`chip-btn ${showSkeleton ? "active" : ""}`}
            onClick={() => setShowSkeleton(!showSkeleton)}
          >
            Skeleton ({stats.bones})
          </button>
          <button
            className={`chip-btn ${chromaBg ? "active-magenta" : ""}`}
            onClick={() => setChromaBg(!chromaBg)}
          >
            #FF00FF Chroma
          </button>
          <button
            className={`chip-btn ${autoRotate ? "active" : ""}`}
            onClick={() => setAutoRotate(!autoRotate)}
          >
            Turntable
          </button>
          {stats.bones > 0 && (
            <button
              className={`chip-btn ${isPlaying ? "active-green" : ""}`}
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? "⏸ Pause Rig" : "▶ Animate Rig"}
            </button>
          )}
        </div>
      </div>

      <div className="viewport-canvas-wrap">
        <div ref={mountRef} className="viewport-canvas" />
        {loading && <div className="viewport-overlay">Loading 3D Asset...</div>}
        {error && <div className="viewport-overlay error">{error}</div>}
        <div className="viewport-hud">
          <span>Vertices: {stats.vertices.toLocaleString()}</span>
          <span>Faces: {stats.faces.toLocaleString()}</span>
          <span>Armature Bones: {stats.bones}</span>
        </div>
      </div>
    </div>
  );
};
