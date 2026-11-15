# **Guía Detallada: Pipeline de Animación 3D con IA**

## **Paso 1: Preparación del Modelo y Esqueleto (05:54 \- 06:26)**

* **Generación y Limpieza del Modelo:** Obtén la malla (mesh) base usando herramientas como Tripo, o impórtala desde tus scripts de generación. Asegúrate de reparar mallas abiertas (no-manifold) para que el algoritmo pueda leer el volumen cerrado.  
* **Opción A \- Auto-Rigging con UniRig:**  
  * Instala el addon UniRig en Blender.  
  * Selecciona tu malla en modo Objeto (Object Mode).  
  * Para automatizarlo por consola de Python dentro de Blender, ejecuta:

```py
import bpy
bpy.ops.unirig.generate_rig(resolution=256, cleanup_loose_bones=True)
```

  * El sistema proyectará la estructura ósea interna basándose en la topología.  
* **Opción B \- Alternativa con SkinTokens (Recomendado para monstruos):**  
  * SkinTokens utiliza IA para predecir los pesos de los vértices (skin weights) y generar esqueletos en topologías complejas o no humanoides.  
  * Abre tu terminal, clona el repositorio de SkinTokens y prepara el entorno usando `pyenv` y `uv`:  
  * 

```sh
pyenv install 3.9.18
pyenv local 3.9.18
uv venv
# En Windows usa: .venv\Scripts\activate
source .venv/bin/activate
uv pip install -r requirements.txt
```

  * Ejecuta el comando de inferencia indicando tu malla y la cantidad de huesos deseados (ej. 24):

```sh
python infer.py --input ./modelo_limpio.obj --output ./modelo_rigged.glb --num_bones 24
```

  * Importa el archivo resultante `modelo_rigged.glb` a Blender.  
* **Refinamiento del Esqueleto:** El paso más crítico es la limpieza manual. Revisa que no haya huesos "flotantes" que sobresalgan de la malla. Ajusta los pivotes (joints) en áreas de doblez (codos, rodillas, falanges) para asegurar que se alineen con los *edge loops* de la geometría.

## **Paso 2: Generación de Video con Referencia en Seedance (02:27 \- 02:42)**

* **Preparación de la Referencia:** Crea un render de referencia de tu criatura usando el fondo plano contrastante (ej. magenta chroma `#FF00FF`).  
* **Vista Ortográfica Doble:** En la *misma imagen*, debes colocar a tu personaje visto desde el frente y en perfil estricto para evitar problemas de profundidad espacial. Sube esta imagen combinada a la interfaz de Seedance como imagen inicial (Image Prompt).  
* **Configuración en Seedance:**  
  * **Cámara Estática:** Es crucial fijar los movimientos de cámara en nulo (Static/None). Si la cámara hace un paneo, el fondo plano se desplazará y confundirá el tracking 2D de DINOv2 en el siguiente paso.  
  * **Fidelidad (Image Weight):** Ajusta el peso de la imagen de referencia a un valor alto para asegurar que la IA respete la topología y no mutile extremidades.  
* **Ejemplos de Prompts para Seedance:**  
  * *Ciclo de Caminata (Walk Cycle):* "A low-poly 3D creature walking forward continuously, heavy and menacing steps, maintaining the exact same flat magenta background, static locked camera, no environment, stable silhouette."  
  * *Ataque Rápido:* "The creature performs a quick horizontal sword slash, aggressive dynamic motion, dark fantasy 80s anime aesthetic, flat \#FF00FF background, static camera angle, no background details."  
  * *Animación de Reposo (Idle):* "The monster standing still and breathing heavily, idle pose, slight shoulder and chest movement, flat magenta background, static locked camera."  
* **Generación de Movimiento:** Genera el video y descárgalo. Revísalo minuciosamente asegurándote de que los brazos y piernas no se fusionen entre sí durante el movimiento, ya que el modelo DINOv2 necesita diferenciar cada extremidad claramente.

## **Paso 3: Entrenamiento del Lector de Articulaciones con DINOv2 (06:27 \- 07:04)**

* **Generación del Dataset y Anotaciones (JSON):** El plugin `DINOv2_Dataset_Generator/ai_anim_plugin.py` genera imágenes con posturas aleatorias, *motion blur* y un `annotations.json` con las coordenadas normalizadas de cabeza y cola de cada hueso, proyectadas por la cámara activa. Configura la ruta de salida del plugin como `D:/monster_animation/spider-renders/dinov2/` para que las imágenes y anotaciones queden disponibles para el pipeline de Docker.  
* **Extracción de Características (Features) con DINOv2:**  
  * En tu terminal con el entorno de Meta DINOv2 configurado, procesa las imágenes para extraer los vectores espaciales:

```sh
python extract_features.py --dataset_path C:/tmp/dataset_monstruo/ --output_dir C:/tmp/dinov2_features/ --model dinov2_vits14
```

* **Entrenamiento del Estimador de Poses:**  
  * Entrena la red de regresión para que aprenda la anatomía específica de tu monstruo basándose en los *features* extraídos y un JSON de anotaciones (generado a la par de las imágenes):

```sh
python train_pose_estimator.py --features_dir C:/tmp/dinov2_features/ --annotations_path C:/tmp/dataset_monstruo/annotations.json --epochs 50 --batch_size 16
```

* **Ejecución con Docker Compose:** También puedes ejecutar extracción, entrenamiento e inferencia en una sola operación. Genera el dataset en `spider-renders/dinov2/` y coloca `seedance_video.mp4` en `spider-renders/`; ejecuta desde la raíz del proyecto:

```sh
docker compose run --build --rm monster-animation
```

Los resultados se guardan en `pipeline-output/`. La anotación puede ser un objeto que asocie cada nombre de imagen con sus puntos, o una lista `annotations` con entradas `image` y `keypoints`/`joints`. Para una GPU NVIDIA disponible en Docker Desktop con soporte GPU configurado, añade `-f docker-compose.gpu.yml`:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --build --rm monster-animation
```

## **Paso 4: Proceso de Animación y Exportación (09:08 \- 09:31)**

* **Inferencia del Video:**  
  * Toma el video de Seedance y procésalo con tu estimador recién entrenado:

```sh
python run_inference.py --video_path ./seedance_video.mp4 --pose_model ./checkpoints/best_pose_model.pth --output_json ./tracked_poses.json
```

* **Retargeting en Blender:**  
  * Instala `DINOv2_Dataset_Generator/pose_retarget_plugin.py` como un addon de Blender y actívalo en Preferences > Add-ons. En la vista 3D, abre la pestaña **AI Anim**, selecciona `tracked_poses.json` y el armature. En **Retarget Space**, elige **Camera Plane** para mapear el tracking delante de una cámara, o **Armature Base Positions** para mapearlo dentro de los límites XY del esqueleto en pose de reposo, sin cámara. Este último modo conserva la profundidad Z de cada extremo del hueso. El addon solo anima rotaciones locales (sin modificar matrices de pose, posiciones ni escalas) y limita las coordenadas predichas al rango normalizado 0–1. Reduce **Rotation Influence** si el movimiento deforma demasiado la silueta. Ambos modos son aproximaciones 2D, no reconstruyen profundidad real.  
* **Exportación:** Haz un *Bake* de la animación a keyframes, y exporta tu personaje en formato `.glb`, `.gltf` o como spritesheet secuencial para integrarlo nativamente en el motor gráfico.

> Docker Compose automatiza la extracción de características, el entrenamiento y el tracking de video; el retargeting y la exportación en Blender siguen siendo pasos manuales porque este proyecto aún no incluye esos scripts.

## **Paso 5: Validación (07:40 \- 08:31)**

* **La "Prueba de Examen":** Antes de procesar videos largos, renderiza una animación controlada corta en Blender.  
* **Evaluación y Ajuste:** Pásala por el script `run_inference.py`. Si los huesos no se trackean correctamente en una animación limpia, debes incrementar el dataset del Paso 3 o afinar las épocas de entrenamiento (`--epochs`).
