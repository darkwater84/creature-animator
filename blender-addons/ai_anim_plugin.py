import bpy
import json
import random
import math
import os
import shutil
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

bl_info = {
    "name": "DINOv2 Dataset Generator (Dual Front & Side Views)",
    "author": "Creador",
    "version": (2, 1),
    "blender": (3, 0, 0),
    "location": "View3D > N Panel > AI Anim",
    "description": "Automatiza renders no bloqueantes (Modal) desde cámaras Front y Side con limpieza previa de carpeta, motion blur y JSON agrupado por posición",
    "category": "Animation",
}


def poll_mesh(self, object):
    return object.type == 'MESH'


def poll_armature(self, object):
    return object.type == 'ARMATURE'


def poll_camera(self, object):
    return object.type == 'CAMERA'


def clean_output_directory(output_dir):
    """Safely remove all existing files and subdirectories inside output_dir before rendering."""
    abs_dir = os.path.abspath(output_dir)
    drive, tail = os.path.splitdrive(abs_dir)
    if tail in ("", "\\", "/") or len(abs_dir) <= 4:
        raise ValueError(f"Ruta de salida insegura para limpiar: {abs_dir}")

    if not os.path.exists(abs_dir):
        os.makedirs(abs_dir, exist_ok=True)
        return 0

    removed_count = 0
    for entry in os.listdir(abs_dir):
        entry_path = os.path.join(abs_dir, entry)
        try:
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.remove(entry_path)
                removed_count += 1
            elif os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
                removed_count += 1
        except OSError as exc:
            print(f"[AI Anim] Aviso: no se pudo eliminar {entry_path}: {exc}")

    print(f"[AI Anim] Carpeta limpia ({removed_count} elementos eliminados): {abs_dir}")
    return removed_count


def ensure_front_and_side_cameras(context, armature, settings):
    """Return (front_camera, side_camera), creating or positioning them if needed."""
    scene = context.scene
    front_cam = settings.front_camera
    side_cam = settings.side_camera

    if front_cam is None:
        for obj in scene.objects:
            if obj.type == 'CAMERA' and 'front' in obj.name.lower():
                front_cam = obj
                break
    if side_cam is None:
        for obj in scene.objects:
            if obj.type == 'CAMERA' and ('side' in obj.name.lower() or 'profile' in obj.name.lower()):
                side_cam = obj
                break

    if front_cam is None and scene.camera is not None and scene.camera != side_cam:
        front_cam = scene.camera

    if not settings.auto_create_cameras and (front_cam is None or side_cam is None):
        return front_cam, side_cam

    bone_points = []
    for bone in armature.data.bones:
        bone_points.append(armature.matrix_world @ bone.head_local)
        bone_points.append(armature.matrix_world @ bone.tail_local)

    if bone_points:
        min_v = Vector((min(p.x for p in bone_points), min(p.y for p in bone_points), min(p.z for p in bone_points)))
        max_v = Vector((max(p.x for p in bone_points), max(p.y for p in bone_points), max(p.z for p in bone_points)))
        center = (min_v + max_v) * 0.5
        extent = max((max_v - min_v).length, 1.0)
    else:
        center = armature.matrix_world.translation.copy()
        extent = 2.5

    distance = extent * 2.2

    if front_cam is None:
        cam_data = bpy.data.cameras.new(name="Camera_Front")
        front_cam = bpy.data.objects.new("Camera_Front", cam_data)
        scene.collection.objects.link(front_cam)
        front_cam.location = Vector((center.x, center.y - distance, center.z))
        direction = center - front_cam.location
        front_cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    if side_cam is None:
        cam_data = bpy.data.cameras.new(name="Camera_Side")
        if front_cam and front_cam.data:
            cam_data.type = front_cam.data.type
            cam_data.lens = front_cam.data.lens
            cam_data.ortho_scale = front_cam.data.ortho_scale
        side_cam = bpy.data.objects.new("Camera_Side", cam_data)
        scene.collection.objects.link(side_cam)
        dist = (front_cam.location - center).length if front_cam else distance
        side_cam.location = Vector((center.x + dist, center.y, center.z))
        direction = center - side_cam.location
        side_cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    settings.front_camera = front_cam
    settings.side_camera = side_cam
    return front_cam, side_cam


def write_annotations_json(output_dir, resolution, use_blur, shutter_val, num_poses, view_names, annotations):
    """Atomically write the grouped v2.0 annotations.json file."""
    annotations_path = os.path.join(output_dir, "annotations.json")
    temporary_path = annotations_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as annotation_file:
        json.dump(
            {
                "schema_version": "2.0",
                "coordinate_system": "normalized_image_xy_origin_top_left",
                "image_resolution": list(resolution),
                "motion_blur_enabled": use_blur,
                "motion_blur_shutter": float(shutter_val) if use_blur else 0.0,
                "views": list(view_names),
                "pose_count": len(annotations),
                "target_pose_count": num_poses,
                "total_renders": len(annotations) * len(view_names),
                "annotations": annotations,
            },
            annotation_file,
            indent=2,
        )
        annotation_file.write("\n")
    os.replace(temporary_path, annotations_path)
    return annotations_path


def render_single_pose(scene, context, armature, pose_index, max_rot, view_cameras, output_dir):
    """Randomize armature bones for a single pose and render both Front and Side views."""
    for frame in (1, 2):
        scene.frame_set(frame)
        for bone in armature.pose.bones:
            bone.rotation_mode = 'XYZ'
            rx = random.uniform(-max_rot, max_rot)
            ry = random.uniform(-max_rot, max_rot)
            rz = random.uniform(-max_rot, max_rot)
            bone.rotation_euler = (rx, ry, rz)
            bone.keyframe_insert(data_path="rotation_euler", frame=frame)

    scene.frame_set(2)
    context.view_layer.update()
    evaluated_armature = armature.evaluated_get(context.evaluated_depsgraph_get())

    keypoints_3d = {}
    world_endpoints = []
    for pose_bone in evaluated_armature.pose.bones:
        for endpoint, position in (
            ("head", pose_bone.head),
            ("tail", pose_bone.tail),
        ):
            joint_name = f"{pose_bone.name}_{endpoint}"
            world_position = evaluated_armature.matrix_world @ position
            world_endpoints.append((joint_name, world_position))
            keypoints_3d[joint_name] = [
                float(world_position.x),
                float(world_position.y),
                float(world_position.z),
            ]

    pose_renders = {}
    for view_name, cam in view_cameras:
        scene.camera = cam
        context.view_layer.update()

        view_keypoints = {}
        for joint_name, world_position in world_endpoints:
            projected = world_to_camera_view(scene, cam, world_position)
            view_keypoints[joint_name] = [
                float(projected.x),
                float(1.0 - projected.y),
            ]

        image_name = f"pose_blur_{pose_index:04d}_{view_name}.png"
        scene.render.filepath = os.path.join(output_dir, image_name)
        bpy.ops.render.render(write_still=True)

        pose_renders[view_name] = {
            "view": view_name,
            "image": image_name,
            "camera": cam.name,
            "keypoints": view_keypoints,
        }

    for bone in armature.pose.bones:
        bone.keyframe_delete(data_path="rotation_euler", frame=1)
        bone.keyframe_delete(data_path="rotation_euler", frame=2)

    return {
        "pose_id": pose_index,
        "pose_name": f"pose_{pose_index:04d}",
        "keypoints_3d": keypoints_3d,
        "renders": pose_renders,
    }


class AIDatasetSettings(bpy.types.PropertyGroup):
    target_mesh: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Modelo",
        description="Selecciona la malla (Mesh) del monstruo",
        poll=poll_mesh,
    )
    target_armature: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Esqueleto",
        description="Selecciona el esqueleto (Armature) a animar",
        poll=poll_armature,
    )
    front_camera: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Cámara Frontal",
        description="Cámara para renderizar la vista frontal (Front)",
        poll=poll_camera,
    )
    side_camera: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Cámara Lateral",
        description="Cámara para renderizar la vista lateral / perfil (Side)",
        poll=poll_camera,
    )
    auto_create_cameras: bpy.props.BoolProperty(
        name="Auto-crear Cámaras Front/Side",
        description="Crea y posiciona automáticamente las cámaras Frontal y Lateral si no están asignadas",
        default=True,
    )
    clean_before_render: bpy.props.BoolProperty(
        name="Limpiar Carpeta Antes de Renderizar",
        description="Elimina todos los renders y anotaciones anteriores en la carpeta de salida antes de iniciar",
        default=True,
    )
    output_dir: bpy.props.StringProperty(
        name="Ruta",
        description="Carpeta donde se guardarán los renders",
        subtype='DIR_PATH',
        default="D:/clean-implementation/workspace/projects/spider-rider/03-dataset/dinov2/",
    )
    num_renders: bpy.props.IntProperty(
        name="Cant. Posiciones (Poses)",
        description="Número total de posturas aleatorias (cada postura se renderiza en vista Front y Side)",
        default=3000,
        min=1,
    )
    max_rotation: bpy.props.FloatProperty(
        name="Ángulo Máx",
        description="Límite máximo de rotación aleatoria para los huesos",
        subtype='ANGLE',
        default=math.radians(45.0),
        min=0.0,
        max=math.radians(180.0),
    )
    use_motion_blur: bpy.props.BoolProperty(
        name="Activar Motion Blur",
        description="Habilita o deshabilita el desenfoque de movimiento durante el render",
        default=True,
    )
    motion_blur_shutter: bpy.props.FloatProperty(
        name="Cantidad de Motion Blur (Shutter)",
        description="Intensidad del desenfoque de movimiento (tiempo de obturación en fotogramas; 0.0 = desactivado, 0.65 = recomendado)",
        default=0.65,
        min=0.0,
        soft_max=2.0,
        max=10.0,
        step=5,
        precision=2,
    )
    # Runtime status properties for non-blocking modal execution
    is_rendering: bpy.props.BoolProperty(
        name="Renderizando",
        default=False,
    )
    cancel_requested: bpy.props.BoolProperty(
        name="Cancelar Render",
        default=False,
    )
    current_pose_index: bpy.props.IntProperty(
        name="Pose Actual",
        default=0,
    )
    progress_percent: bpy.props.FloatProperty(
        name="Progreso",
        subtype='PERCENTAGE',
        default=0.0,
        min=0.0,
        max=100.0,
    )


class OBJECT_OT_setup_front_side_cameras(bpy.types.Operator):
    bl_idname = "object.setup_front_side_cameras"
    bl_label = "Configurar Cámaras Front & Side"
    bl_description = "Crea o detecta automáticamente las cámaras Frontal y Lateral enfocadas al esqueleto"

    def execute(self, context):
        settings = context.scene.ai_dataset_settings
        armature = settings.target_armature
        if not armature:
            self.report({'ERROR'}, "Selecciona un Esqueleto (Armature) primero.")
            return {'CANCELLED'}
        front_cam, side_cam = ensure_front_and_side_cameras(context, armature, settings)
        if not front_cam or not side_cam:
            self.report({'ERROR'}, "No se pudieron configurar las cámaras Front y Side.")
            return {'CANCELLED'}
        self.report(
            {'INFO'},
            f"Cámaras listas: Front='{front_cam.name}', Side='{side_cam.name}'",
        )
        return {'FINISHED'}


class OBJECT_OT_cancel_ai_dataset(bpy.types.Operator):
    bl_idname = "object.cancel_ai_dataset"
    bl_label = "Detener Generación de Dataset"
    bl_description = "Detiene el renderizado no bloqueante después de la postura actual y guarda las anotaciones generadas hasta el momento"

    def execute(self, context):
        settings = context.scene.ai_dataset_settings
        settings.cancel_requested = True
        self.report({'INFO'}, "Deteniendo generación tras completar la postura actual...")
        return {'FINISHED'}


class OBJECT_OT_generate_ai_dataset(bpy.types.Operator):
    bl_idname = "object.generate_ai_dataset"
    bl_label = "Generar Dataset Front & Side (No Bloqueante)"
    bl_description = "Limpia la carpeta de salida y genera renders Front y Side por postura en modo no bloqueante (ESC o botón para cancelar)"

    _timer = None
    _pose_index = 0
    _num_poses = 0
    _max_rot = 0.0
    _use_blur = True
    _shutter_val = 0.65
    _output_dir = ""
    _resolution = (1920, 1080)
    _view_cameras = ()
    _annotations = None
    _original_camera = None

    def _prepare_session(self, context):
        settings = context.scene.ai_dataset_settings
        armature = settings.target_armature
        scene = context.scene

        if settings.is_rendering:
            self.report({'WARNING'}, "Ya hay una generación de dataset en curso.")
            return False
        if not armature:
            self.report({'ERROR'}, "Debes seleccionar un Esqueleto (Armature) primero.")
            return False
        if not armature.pose.bones:
            self.report({'ERROR'}, "El esqueleto seleccionado no contiene huesos.")
            return False

        front_cam, side_cam = ensure_front_and_side_cameras(context, armature, settings)
        if not front_cam or not side_cam:
            self.report(
                {'ERROR'},
                "Se requieren ambas cámaras (Front y Side). Asígnalas o activa 'Auto-crear Cámaras Front/Side'.",
            )
            return False

        output_dir = bpy.path.abspath(settings.output_dir)
        try:
            if settings.clean_before_render:
                clean_output_directory(output_dir)
            else:
                os.makedirs(output_dir, exist_ok=True)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return False

        use_blur = bool(settings.use_motion_blur and settings.motion_blur_shutter > 0.0)
        scene.render.use_motion_blur = use_blur
        scene.render.motion_blur_shutter = float(settings.motion_blur_shutter)
        scene.render.film_transparent = False

        world = scene.world
        if world is None:
            world = bpy.data.worlds.new("World_Magenta")
            scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs[0].default_value = (1.0, 0.0, 1.0, 1.0)

        self._original_camera = scene.camera
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')

        self._pose_index = 0
        self._num_poses = int(settings.num_renders)
        self._max_rot = float(settings.max_rotation)
        self._use_blur = use_blur
        self._shutter_val = float(settings.motion_blur_shutter)
        self._output_dir = output_dir
        self._resolution = (
            round(scene.render.resolution_x * scene.render.resolution_percentage / 100),
            round(scene.render.resolution_y * scene.render.resolution_percentage / 100),
        )
        self._view_cameras = (
            ("front", front_cam),
            ("side", side_cam),
        )
        self._annotations = []

        settings.is_rendering = True
        settings.cancel_requested = False
        settings.current_pose_index = 0
        settings.progress_percent = 0.0
        return True

    def _finalize_session(self, context, cancelled=False):
        scene = context.scene
        settings = scene.ai_dataset_settings

        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        scene.camera = self._original_camera
        try:
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

        annotations_path = ""
        if self._annotations is not None:
            annotations_path = write_annotations_json(
                self._output_dir,
                self._resolution,
                self._use_blur,
                self._shutter_val,
                self._num_poses,
                [v[0] for v in self._view_cameras],
                self._annotations,
            )

        settings.is_rendering = False
        settings.cancel_requested = False
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        count = len(self._annotations) if self._annotations else 0
        if cancelled:
            self.report(
                {'WARNING'},
                f"Generación detenida en pose {count}/{self._num_poses}. Guardado parcial: {annotations_path}",
            )
        else:
            self.report(
                {'INFO'},
                f"Se generaron {count} posturas ({count * 2} renders Front+Side): {annotations_path}",
            )

    def invoke(self, context, event):
        if bpy.app.background:
            return self.execute(context)

        if not self._prepare_session(context):
            return {'CANCELLED'}

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        print(
            f"[AI Anim] Render no bloqueante iniciado: {self._num_poses} posturas x 2 vistas (Front + Side)..."
        )
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        settings = context.scene.ai_dataset_settings

        if event.type == 'ESC' or settings.cancel_requested:
            self._finalize_session(context, cancelled=True)
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._pose_index >= self._num_poses:
            self._finalize_session(context, cancelled=False)
            return {'FINISHED'}

        armature = settings.target_armature
        if not armature:
            self._finalize_session(context, cancelled=True)
            return {'CANCELLED'}

        pose_entry = render_single_pose(
            context.scene,
            context,
            armature,
            self._pose_index,
            self._max_rot,
            self._view_cameras,
            self._output_dir,
        )
        self._annotations.append(pose_entry)
        self._pose_index += 1

        settings.current_pose_index = self._pose_index
        settings.progress_percent = (self._pose_index / max(1, self._num_poses)) * 100.0

        # Periodically flush annotations.json every 25 poses so progress is persisted
        if self._pose_index % 25 == 0:
            write_annotations_json(
                self._output_dir,
                self._resolution,
                self._use_blur,
                self._shutter_val,
                self._num_poses,
                [v[0] for v in self._view_cameras],
                self._annotations,
            )

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        if self._pose_index >= self._num_poses:
            self._finalize_session(context, cancelled=False)
            return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def execute(self, context):
        """Synchronous execution fallback for headless CLI (blender -b)."""
        if not self._prepare_session(context):
            return {'CANCELLED'}

        settings = context.scene.ai_dataset_settings
        armature = settings.target_armature
        try:
            for i in range(self._num_poses):
                pose_entry = render_single_pose(
                    context.scene,
                    context,
                    armature,
                    i,
                    self._max_rot,
                    self._view_cameras,
                    self._output_dir,
                )
                self._annotations.append(pose_entry)
                self._pose_index = i + 1
                settings.current_pose_index = self._pose_index
                settings.progress_percent = (self._pose_index / max(1, self._num_poses)) * 100.0
        finally:
            self._finalize_session(context, cancelled=False)
        return {'FINISHED'}


class VIEW3D_PT_ai_dataset_panel(bpy.types.Panel):
    bl_label = "DINOv2 Dataset Generator"
    bl_idname = "VIEW3D_PT_ai_dataset_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'AI Anim'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.ai_dataset_settings

        box = layout.box()
        box.label(text="Referencias 3D:", icon='OUTLINER_OB_MESH')
        box.prop(settings, "target_mesh")
        box.prop(settings, "target_armature")

        box = layout.box()
        box.label(text="Cámaras Multivista (Front & Side):", icon='CAMERA_DATA')
        box.prop(settings, "front_camera")
        box.prop(settings, "side_camera")
        box.prop(settings, "auto_create_cameras")
        box.operator("object.setup_front_side_cameras", icon='VIEW_CAMERA')

        box = layout.box()
        box.label(text="Parámetros de Pose y Motion Blur:", icon='BONE_DATA')
        box.prop(settings, "max_rotation")
        box.prop(settings, "use_motion_blur")
        blur_row = box.row()
        blur_row.enabled = settings.use_motion_blur
        blur_row.prop(settings, "motion_blur_shutter", slider=True)

        box = layout.box()
        box.label(text="Salida (Output):", icon='RENDER_RESULT')
        box.prop(settings, "num_renders")
        box.prop(settings, "output_dir")
        box.prop(settings, "clean_before_render")

        layout.separator()
        if settings.is_rendering:
            prog_box = layout.box()
            prog_box.label(
                text=f"Renderizando Pose {settings.current_pose_index} / {settings.num_renders} (Front + Side)",
                icon='RENDER_ANIMATION',
            )
            prog_box.prop(settings, "progress_percent", slider=True)
            prog_box.operator(
                "object.cancel_ai_dataset",
                icon='CANCEL',
                text="Detener Render (ESC)",
            )
        else:
            layout.operator(
                "object.generate_ai_dataset",
                icon='PLAY',
                text="Generar Dataset (Front + Side)",
            )


classes = (
    AIDatasetSettings,
    OBJECT_OT_setup_front_side_cameras,
    OBJECT_OT_cancel_ai_dataset,
    OBJECT_OT_generate_ai_dataset,
    VIEW3D_PT_ai_dataset_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ai_dataset_settings = bpy.props.PointerProperty(type=AIDatasetSettings)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.ai_dataset_settings


if __name__ == "__main__":
    register()
