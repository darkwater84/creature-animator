"""Retarget normalized DINOv2 2D/Dual-View (Front + Side) joint tracks onto a Blender armature (Non-Blocking Modal)."""

import json
import math

import bpy
from mathutils import Quaternion, Vector


bl_info = {
    "name": "DINOv2 Pose Retargeter (Front & Side Multi-View)",
    "author": "OpenAI",
    "version": (2, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > AI Anim",
    "description": "Non-blocking modal retargeter for grouped Front/Side tracked_poses.json onto an armature in 3D or camera space",
    "category": "Animation",
}


def _parse_keypoints_dict(raw_kp, frame_number, view_label="default"):
    if not isinstance(raw_kp, dict):
        raise ValueError(f"Frame entry {frame_number} ({view_label}) has no keypoints object")
    keypoints = {}
    for name, point in raw_kp.items():
        if not isinstance(name, str) or not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError(f"Invalid keypoint {name!r} in frame entry {frame_number} ({view_label})")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Non-finite keypoint {name!r} in frame entry {frame_number} ({view_label})")
        keypoints[name] = (x, y)
    return keypoints


def load_tracks(filepath, side_filepath=""):
    with open(bpy.path.abspath(filepath), "r", encoding="utf-8-sig") as track_file:
        data = json.load(track_file)

    frames = None
    if isinstance(data, dict):
        frames = data.get("frames") or data.get("annotations")
    if not isinstance(frames, list) or not frames:
        raise ValueError("JSON must contain a non-empty 'frames' or 'annotations' list")

    side_frames_by_idx = {}
    if side_filepath and side_filepath.strip():
        with open(bpy.path.abspath(side_filepath), "r", encoding="utf-8-sig") as side_file:
            side_data = json.load(side_file)
        raw_side_list = (
            side_data.get("frames") or side_data.get("annotations")
            if isinstance(side_data, dict)
            else None
        )
        if isinstance(raw_side_list, list):
            for idx, sf in enumerate(raw_side_list):
                if isinstance(sf, dict):
                    f_idx = int(sf.get("frame_index", sf.get("pose_id", idx)))
                    if isinstance(sf.get("keypoints"), dict):
                        side_frames_by_idx[f_idx] = _parse_keypoints_dict(sf["keypoints"], idx, "side_file")
                    elif isinstance(sf.get("renders"), dict) and "side" in sf["renders"]:
                        side_frames_by_idx[f_idx] = _parse_keypoints_dict(
                            sf["renders"]["side"].get("keypoints"), idx, "side_file"
                        )

    validated_frames = []
    for frame_number, frame_data in enumerate(frames):
        if not isinstance(frame_data, dict):
            raise ValueError(f"Frame entry {frame_number} must be an object")

        frame_index = int(frame_data.get("frame_index", frame_data.get("pose_id", frame_number)))
        timestamp = frame_data.get("timestamp_sec")
        renders_map = {}

        raw_renders = frame_data.get("renders") or frame_data.get("views")
        if isinstance(raw_renders, dict):
            for view_name, view_obj in raw_renders.items():
                if isinstance(view_obj, dict) and isinstance(view_obj.get("keypoints"), dict):
                    renders_map[view_name.lower()] = _parse_keypoints_dict(
                        view_obj["keypoints"], frame_number, view_name
                    )
                elif isinstance(view_obj, dict):
                    renders_map[view_name.lower()] = _parse_keypoints_dict(
                        view_obj, frame_number, view_name
                    )

        if isinstance(frame_data.get("keypoints"), dict):
            flat_kp = _parse_keypoints_dict(frame_data["keypoints"], frame_number, "flat")
            if "front" not in renders_map:
                renders_map["front"] = flat_kp

        if frame_index in side_frames_by_idx and "side" not in renders_map:
            renders_map["side"] = side_frames_by_idx[frame_index]

        if not renders_map:
            raise ValueError(f"Frame entry {frame_number} has neither 'renders' nor 'keypoints'")

        primary_kp = renders_map.get("front") or next(iter(renders_map.values()))
        validated_frames.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": float(timestamp) if timestamp is not None else None,
                "keypoints": primary_kp,
                "renders": renders_map,
            }
        )

    return validated_frames


def camera_plane_point(scene, camera, normalized_x, normalized_y, depth):
    normalized_x = min(max(normalized_x, 0.0), 1.0)
    normalized_y = min(max(normalized_y, 0.0), 1.0)
    frame = camera.data.view_frame(scene=scene)
    min_x = min(point.x for point in frame)
    max_x = max(point.x for point in frame)
    min_y = min(point.y for point in frame)
    max_y = max(point.y for point in frame)
    local_z = -max(float(depth), camera.data.clip_start)

    if camera.data.type == "ORTHO":
        local_x = min_x + normalized_x * (max_x - min_x)
        local_y = max_y - normalized_y * (max_y - min_y)
    else:
        frame_depth = abs(sum(point.z for point in frame) / len(frame))
        if frame_depth <= 0:
            raise ValueError("Active camera returned an invalid view frame")
        scale = abs(local_z) / frame_depth
        local_x = (min_x + normalized_x * (max_x - min_x)) * scale
        local_y = (max_y - normalized_y * (max_y - min_y)) * scale

    return camera.matrix_world @ Vector((local_x, local_y, local_z))


def armature_base_bounds(bone_tracks):
    base_points = [
        point
        for pose_bone, _, _ in bone_tracks
        for point in (pose_bone.bone.head_local, pose_bone.bone.tail_local)
    ]
    return (
        min(point.x for point in base_points),
        max(point.x for point in base_points),
        min(point.y for point in base_points),
        max(point.y for point in base_points),
        min(point.z for point in base_points),
        max(point.z for point in base_points),
    )


def armature_base_point(normalized_x, normalized_y, depth, bounds):
    normalized_x = min(max(normalized_x, 0.0), 1.0)
    normalized_y = min(max(normalized_y, 0.0), 1.0)
    min_x, max_x, min_y, max_y = bounds[:4]
    return Vector(
        (
            min_x + normalized_x * (max_x - min_x),
            max_y - normalized_y * (max_y - min_y),
            depth,
        )
    )


def armature_dual_view_3d_point(front_xy, side_xy, bounds):
    fx = min(max(front_xy[0], 0.0), 1.0)
    fy = min(max(front_xy[1], 0.0), 1.0)
    sx = min(max(side_xy[0], 0.0), 1.0)
    sy = min(max(side_xy[1], 0.0), 1.0)

    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    combined_vertical = 0.5 * (fy + sy)
    return Vector(
        (
            min_x + fx * (max_x - min_x),
            min_y + sx * (max_y - min_y),
            max_z - combined_vertical * (max_z - min_z),
        )
    )


def frame_number_for_track(scene, track, start_frame, frame_step):
    timestamp = track["timestamp_sec"]
    if timestamp is not None:
        scene_fps = scene.render.fps / scene.render.fps_base
        return start_frame + timestamp * scene_fps
    return start_frame + track["frame_index"] * frame_step


def apply_track_to_armature(
    scene,
    context,
    armature,
    armature_inverse,
    bone_tracks,
    track,
    settings,
    camera,
    base_bounds,
):
    """Apply a single frame track onto the armature's pose bones and insert keyframes."""
    frame_number = frame_number_for_track(
        scene,
        track,
        settings.start_frame,
        settings.frame_step,
    )
    whole_frame = math.floor(frame_number)
    scene.frame_set(whole_frame, subframe=frame_number - whole_frame)
    context.view_layer.update()

    renders = track.get("renders", {})
    front_kp = renders.get("front", track["keypoints"])
    side_kp = renders.get("side")

    for pose_bone, head_name, tail_name in bone_tracks:
        head_2d = front_kp[head_name]
        tail_2d = front_kp[tail_name]

        if (
            settings.retarget_space == "FRONT_SIDE_3D"
            and side_kp is not None
            and head_name in side_kp
            and tail_name in side_kp
        ):
            head_side = side_kp[head_name]
            tail_side = side_kp[tail_name]
            target_direction = armature_dual_view_3d_point(
                tail_2d, tail_side, base_bounds
            ) - armature_dual_view_3d_point(
                head_2d, head_side, base_bounds
            )
        elif settings.retarget_space == "CAMERA_PLANE" and camera is not None:
            head_world = camera_plane_point(
                scene, camera, head_2d[0], head_2d[1], settings.plane_depth
            )
            tail_world = camera_plane_point(
                scene, camera, tail_2d[0], tail_2d[1], settings.plane_depth
            )
            target_direction = armature_inverse.to_3x3() @ (
                tail_world - head_world
            )
        else:
            target_direction = armature_base_point(
                tail_2d[0],
                tail_2d[1],
                pose_bone.bone.tail_local.z,
                base_bounds,
            ) - armature_base_point(
                head_2d[0],
                head_2d[1],
                pose_bone.bone.head_local.z,
                base_bounds,
            )

        if target_direction.length <= 1e-8 or pose_bone.bone.length <= 1e-8:
            continue

        rest_direction = pose_bone.bone.tail_local - pose_bone.bone.head_local
        if rest_direction.length <= 1e-8:
            continue
        rest_rotation = pose_bone.bone.matrix_local.to_quaternion()
        direction_delta = rest_direction.rotation_difference(target_direction)
        local_rotation_delta = (
            rest_rotation.inverted() @ direction_delta @ rest_rotation
        )
        pose_rotation = Quaternion((1.0, 0.0, 0.0, 0.0)).slerp(
            local_rotation_delta, settings.rotation_influence
        )
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = pose_rotation
        pose_bone.keyframe_insert(
            data_path="rotation_quaternion", frame=frame_number
        )


class DINOv2RetargetSettings(bpy.types.PropertyGroup):
    tracks_path: bpy.props.StringProperty(
        name="Tracked Poses JSON",
        description="tracked_poses.json (supports v2 grouped Front+Side renders or single view)",
        subtype="FILE_PATH",
    )
    side_tracks_path: bpy.props.StringProperty(
        name="Side View Tracks JSON (Optional)",
        description="Optional separate side-view tracked_poses.json if not already grouped in main JSON",
        subtype="FILE_PATH",
    )
    target_armature: bpy.props.PointerProperty(
        name="Armature",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE",
    )
    target_camera: bpy.props.PointerProperty(
        name="Front Camera",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "CAMERA",
    )
    side_camera: bpy.props.PointerProperty(
        name="Side Camera",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "CAMERA",
    )
    retarget_space: bpy.props.EnumProperty(
        name="Retarget Space",
        description="Choose how 2D or grouped Front+Side coordinates are mapped onto the armature",
        items=(
            (
                "FRONT_SIDE_3D",
                "Front + Side Dual-View 3D",
                "Reconstruct 3D bone directions from grouped Front and Side view renders",
            ),
            (
                "CAMERA_PLANE",
                "Camera Plane",
                "Map tracks onto a plane in front of the active/front camera",
            ),
            (
                "ARMATURE_BASE",
                "Armature Base Positions",
                "Map tracks into the armature rest-pose bounds without using a camera",
            ),
        ),
        default="FRONT_SIDE_3D",
    )
    plane_depth: bpy.props.FloatProperty(
        name="Camera Plane Depth",
        description="Distance from the camera to the flat plane used for 2D retargeting",
        default=10.0,
        min=0.001,
        soft_max=1000.0,
    )
    rotation_influence: bpy.props.FloatProperty(
        name="Rotation Influence",
        description="Blend toward tracked directions; lower values give gentler motion",
        default=0.15,
        min=0.0,
        max=1.0,
    )
    start_frame: bpy.props.IntProperty(
        name="Start Frame",
        description="Timeline frame for video time zero",
        default=1,
    )
    frame_step: bpy.props.FloatProperty(
        name="Frame Step",
        description="Timeline frames per video frame when timestamps/FPS are unavailable",
        default=1.0,
        min=0.001,
    )
    is_retargeting: bpy.props.BoolProperty(
        name="Retargeting",
        default=False,
    )
    cancel_requested: bpy.props.BoolProperty(
        name="Cancel Retargeting",
        default=False,
    )
    progress_percent: bpy.props.FloatProperty(
        name="Progress",
        subtype="PERCENTAGE",
        default=0.0,
        min=0.0,
        max=100.0,
    )


class OBJECT_OT_dinov2_cancel_retarget(bpy.types.Operator):
    bl_idname = "object.dinov2_cancel_retarget"
    bl_label = "Stop Retargeting"
    bl_description = "Cancel the non-blocking retargeting loop"

    def execute(self, context):
        context.scene.dinov2_retarget_settings.cancel_requested = True
        return {"FINISHED"}


class OBJECT_OT_dinov2_retarget(bpy.types.Operator):
    bl_idname = "object.dinov2_retarget"
    bl_label = "Retarget Tracked Poses (Non-Blocking)"
    bl_description = "Create a new armature action from DINOv2 tracked pose JSON without freezing Blender"
    bl_options = {"REGISTER", "UNDO"}

    _timer = None
    _tracks = None
    _track_idx = 0
    _bone_tracks = None
    _base_bounds = None
    _armature_inverse = None
    _camera = None
    _action = None
    _original_frame = 1
    _original_mode = "OBJECT"
    _previous_active = None

    def _prepare(self, context):
        settings = context.scene.dinov2_retarget_settings
        if settings.is_retargeting:
            self.report({"WARNING"}, "Retargeting is already in progress.")
            return False

        armature = settings.target_armature
        camera = settings.target_camera or context.scene.camera
        if armature is None:
            self.report({"ERROR"}, "Select the target armature.")
            return False
        if settings.retarget_space == "CAMERA_PLANE" and camera is None:
            self.report({"ERROR"}, "Select a camera or set an active scene camera.")
            return False
        if not settings.tracks_path:
            self.report({"ERROR"}, "Choose a tracked_poses.json file.")
            return False

        try:
            tracks = load_tracks(settings.tracks_path, settings.side_tracks_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            self.report({"ERROR"}, f"Could not load tracked poses: {error}")
            return False

        bone_tracks = []
        missing_bones = []
        for pose_bone in armature.pose.bones:
            head_name = f"{pose_bone.name}_head"
            tail_name = f"{pose_bone.name}_tail"
            if all(head_name in track["keypoints"] and tail_name in track["keypoints"] for track in tracks):
                bone_tracks.append((pose_bone, head_name, tail_name))
            else:
                missing_bones.append(pose_bone.name)
        bone_tracks.sort(
            key=lambda item: sum(1 for _ in self._bone_ancestors(item[0]))
        )

        if not bone_tracks:
            self.report(
                {"ERROR"},
                "No armature bone has matching '<bone>_head' and '<bone>_tail' tracks in every frame.",
            )
            return False

        if missing_bones:
            print(
                "DINOv2 retarget skipped bones without complete tracks: "
                + ", ".join(missing_bones)
            )

        scene = context.scene
        self._original_frame = scene.frame_current
        self._original_mode = armature.mode
        self._previous_active = context.view_layer.objects.active

        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        context.view_layer.objects.active = armature
        armature.select_set(True)
        bpy.ops.object.mode_set(mode="POSE")

        armature.animation_data_create()
        self._action = bpy.data.actions.new(name=f"DINOv2 Retarget {armature.name}")
        armature.animation_data.action = self._action

        self._tracks = tracks
        self._track_idx = 0
        self._bone_tracks = bone_tracks
        self._base_bounds = armature_base_bounds(bone_tracks)
        self._armature_inverse = armature.matrix_world.inverted()
        self._camera = camera

        settings.is_retargeting = True
        settings.cancel_requested = False
        settings.progress_percent = 0.0
        return True

    def _finish(self, context, cancelled=False):
        scene = context.scene
        settings = scene.dinov2_retarget_settings
        armature = settings.target_armature

        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        scene.frame_set(self._original_frame)
        try:
            if armature and armature.mode != self._original_mode:
                if context.object and context.object.mode != "OBJECT":
                    bpy.ops.object.mode_set(mode="OBJECT")
                context.view_layer.objects.active = (
                    armature if self._original_mode != "OBJECT" else self._previous_active
                )
                if self._original_mode != "OBJECT":
                    bpy.ops.object.mode_set(mode=self._original_mode)
            context.view_layer.objects.active = self._previous_active
        except Exception:
            pass

        settings.is_retargeting = False
        settings.cancel_requested = False

        if cancelled:
            self.report({"WARNING"}, f"Retargeting stopped at frame {self._track_idx}/{len(self._tracks)}.")
        else:
            self.report(
                {"INFO"},
                f"Created action '{self._action.name}' for {len(self._bone_tracks)} bones "
                f"across {len(self._tracks)} tracked frames.",
            )

    def invoke(self, context, event):
        if bpy.app.background:
            return self.execute(context)

        if not self._prepare(context):
            return {"CANCELLED"}

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        settings = context.scene.dinov2_retarget_settings
        if event.type == "ESC" or settings.cancel_requested:
            self._finish(context, cancelled=True)
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        armature = settings.target_armature
        if armature is None or self._tracks is None:
            self._finish(context, cancelled=True)
            return {"CANCELLED"}

        # Process up to 5 frames per timer tick for smooth UI responsiveness
        batch_end = min(self._track_idx + 5, len(self._tracks))
        while self._track_idx < batch_end:
            apply_track_to_armature(
                context.scene,
                context,
                armature,
                self._armature_inverse,
                self._bone_tracks,
                self._tracks[self._track_idx],
                settings,
                self._camera,
                self._base_bounds,
            )
            self._track_idx += 1

        settings.progress_percent = (self._track_idx / max(1, len(self._tracks))) * 100.0
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()

        if self._track_idx >= len(self._tracks):
            self._finish(context, cancelled=False)
            return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def execute(self, context):
        if not self._prepare(context):
            return {"CANCELLED"}

        settings = context.scene.dinov2_retarget_settings
        armature = settings.target_armature
        try:
            for track in self._tracks:
                apply_track_to_armature(
                    context.scene,
                    context,
                    armature,
                    self._armature_inverse,
                    self._bone_tracks,
                    track,
                    settings,
                    self._camera,
                    self._base_bounds,
                )
                self._track_idx += 1
        finally:
            self._finish(context, cancelled=False)
        return {"FINISHED"}

    @staticmethod
    def _bone_ancestors(pose_bone):
        parent = pose_bone.parent
        while parent is not None:
            yield parent
            parent = parent.parent


class VIEW3D_PT_dinov2_retarget(bpy.types.Panel):
    bl_label = "DINOv2 Pose Retargeter"
    bl_idname = "VIEW3D_PT_dinov2_pose_retargeter"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Anim"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.dinov2_retarget_settings
        layout.prop(settings, "tracks_path")
        layout.prop(settings, "side_tracks_path")
        layout.prop(settings, "target_armature")
        layout.prop(settings, "retarget_space")
        if settings.retarget_space == "CAMERA_PLANE":
            layout.prop(settings, "target_camera")
            layout.prop(settings, "plane_depth")
        elif settings.retarget_space == "FRONT_SIDE_3D":
            layout.prop(settings, "target_camera")
            layout.prop(settings, "side_camera")
        layout.prop(settings, "rotation_influence")
        layout.prop(settings, "start_frame")
        layout.prop(settings, "frame_step")

        if settings.is_retargeting:
            box = layout.box()
            box.prop(settings, "progress_percent", slider=True)
            box.operator("object.dinov2_cancel_retarget", icon="CANCEL", text="Stop Retargeting (ESC)")
        else:
            layout.operator("object.dinov2_retarget", icon="ARMATURE_DATA")


CLASSES = (
    DINOv2RetargetSettings,
    OBJECT_OT_dinov2_cancel_retarget,
    OBJECT_OT_dinov2_retarget,
    VIEW3D_PT_dinov2_retarget,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dinov2_retarget_settings = bpy.props.PointerProperty(
        type=DINOv2RetargetSettings
    )


def unregister():
    del bpy.types.Scene.dinov2_retarget_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
