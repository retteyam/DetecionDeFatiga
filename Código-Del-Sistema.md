# Sistema de Detección de Somnolencia en Tiempo Real

Esta aplicación es una herramienta de seguridad desarrollada en Python que utiliza **Visión por Computadora** e **Inteligencia Artificial** para monitorear el estado de alerta de un conductor o usuario en tiempo real.

Combina una interfaz gráfica moderna construida con **PyQt6** y un backend de procesamiento de imágenes que integra **OpenCV** (para detección facial y de ojos) y **TensorFlow/Keras** (para clasificación de bostezos mediante Deep Learning).

---

## 📋 Características Principales

✅ **Interfaz Gráfica (GUI)**: Ventana amigable con visualización de video en vivo e indicadores de estado codificados por colores.

✅ **Detección de Rostro y Ojos**: Utiliza Haar Cascades para localizar el rostro y verificar si los ojos están abiertos o cerrados.

✅ **Detección de Bostezos**: Emplea un modelo de red neuronal (CNN) cargado desde Keras para predecir si el usuario está bostezando.

✅ **Alertas Visuales**:
- 🟢 **Verde**: Estado normal/atento.
- 🟠 **Naranja**: Bostezo detectado (Advertencia).
- 🔴 **Rojo**: Ojos cerrados por tiempo prolongado (Alerta Crítica).

✅ **Multihilo (Threading)**: El procesamiento de video se ejecuta en un hilo separado para no congelar la interfaz de usuario.

---

## 🛠️ Requisitos del Sistema

Para ejecutar este proyecto, necesitas tener instalado **Python 3.8 o superior** y las siguientes librerías:

```bash
pip install opencv-python numpy PyQt6 tensorflow
```

**Dependencias principales:**
- `opencv-python` - Visión por Computadora
- `numpy` - Operaciones con arrays
- `PyQt6` - Interfaz Gráfica
- `tensorflow` - Deep Learning

---

## 📂 Estructura del Proyecto

Para que el código funcione correctamente, especialmente la función `resource_path`, se espera una estructura de directorios similar a la siguiente:

```
DetecionDeFatiga/
├── src/
│   └── main.py            # El código principal de la aplicación
├── assets/
│   └── drowiness.keras    # Tu modelo entrenado (REQUERIDO)
├── requirements.txt
└── README.md
```

> **Nota:** El script busca automáticamente los clasificadores Haar Cascade dentro de la instalación de `cv2`, pero el archivo `drowiness.keras` debe ser provisto por ti y colocado en la carpeta `assets`.

---

## 🧠 Arquitectura del Código

El código está dividido en **tres componentes principales**:

### 1. `resource_path(relative_path)`

Una función utilitaria diseñada para manejar rutas de archivos de manera robusta. Permite que la aplicación funcione tanto en el entorno de desarrollo (ejecutando el script `.py`) como cuando se empaqueta en un ejecutable (usando PyInstaller), gestionando las rutas temporales de extracción (`sys._MEIPASS`).

**Funcionamiento:**
```python
def resource_path(relative_path):
    try:
        # PyInstaller crea una carpeta temporal y almacena la ruta en _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # En desarrollo, obtenemos el directorio del script actual (src/)
        # y subimos un nivel para acceder a la raíz del proyecto
        base_path = Path(__file__).resolve().parent.parent
    
    return os.path.join(base_path, relative_path)
```

**Uso:**
```python
model_path = resource_path("assets/drowiness.keras")
```

---

### 2. Clase `VideoThread` (QThread)

Es el **núcleo lógico** de la aplicación. Se ejecuta en segundo plano para capturar y procesar video.

#### 🔧 Atributos Principales

| Atributo | Tipo | Descripción |
|----------|------|-------------|
| `change_pixmap_signal` | `pyqtSignal(np.ndarray)` | Señal para enviar frames procesados a la UI |
| `status_signal` | `pyqtSignal(str, str)` | Señal para actualizar estado (texto, color) |
| `model` | `tf.keras.Model` | Modelo de red neuronal para clasificación |
| `face_cascade` | `cv2.CascadeClassifier` | Detector de rostros HaarCascade |
| `eye_cascade` | `cv2.CascadeClassifier` | Detector de ojos HaarCascade |
| `IMG_SIZE` | `145` | Tamaño de entrada del modelo (145x145 píxeles) |
| `eyes_closed_frames` | `int` | Contador de frames consecutivos sin ojos detectados |
| `EYES_CLOSED_THRESHOLD` | `3` | Umbral de frames para alerta de ojos cerrados |

#### 🚀 Inicialización: `load_resources()`

Carga el modelo `.keras` y los clasificadores XML (Haar Cascades).

**Recursos cargados:**
1. **Modelo de detección de bostezos**: `assets/drowiness.keras`
2. **HaarCascade para rostros**: `haarcascade_frontalface_default.xml`
3. **HaarCascade para ojos**: `haarcascade_eye.xml`

```python
# Cargar modelo Keras
model_path = resource_path("assets/drowiness.keras")
self.model = tf.keras.models.load_model(model_path)

# Cargar HaarCascades
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
```

#### 🔄 Ciclo `run()` - Loop Principal

1. **Captura un frame** de la cámara web.
2. **Detecta el rostro** usando HaarCascade.
3. **Lógica de Ojos**: Busca ojos dentro de la región del rostro. Si no detecta ojos durante `EYES_CLOSED_THRESHOLD` (3 frames consecutivos), activa la alerta de ojos cerrados.
4. **Preprocesamiento**: 
   - Recorta el rostro
   - Lo redimensiona a **145x145 píxeles**
   - Convierte de BGR a RGB
   - Normaliza los valores a rango [0, 1]
5. **Inferencia (TensorFlow)**: Pasa la imagen procesada al modelo para obtener la probabilidad de bostezo (`yawn` vs `no_yawn`).
6. **Emisión de Señales**: Envía la imagen procesada y el estado (texto y color) a la interfaz gráfica.

**Diagrama del flujo:**

```
┌─→ Capturar frame de cámara
│   ↓
│   Convertir a escala de grises
│   ↓
│   Detectar rostros (HaarCascade)
│   ↓
│   ┌─ SI HAY ROSTRO ────────────────────┐
│   │                                     │
│   │  Detectar ojos en región del rostro │
│   │  ↓                                  │
│   │  Actualizar contador ojos cerrados  │
│   │  ↓                                  │
│   │  Recortar y preparar ROI facial     │
│   │  ↓                                  │
│   │  Redimensionar a 145x145            │
│   │  ↓                                  │
│   │  Normalizar valores RGB [0,1]       │
│   │  ↓                                  │
│   │  Inferencia modelo TensorFlow       │
│   │  ↓                                  │
│   │  Interpretar predicciones:          │
│   │  [prob_no_yawn, prob_yawn]          │
│   │  ↓                                  │
│   │  Determinar estado final:           │
│   │  - Ojos cerrados → ALERTA CRÍTICA   │
│   │  - Bostezo → ADVERTENCIA            │
│   │  - Normal → OK                      │
│   │  ↓                                  │
│   │  Emitir status_signal               │
│   │  ↓                                  │
│   │  Dibujar rectángulos y etiquetas    │
│   │                                     │
│   └─────────────────────────────────────┘
│   ↓
│   NO HAY ROSTRO → Emitir "Buscando rostro..."
│   ↓
│   Convertir BGR → RGB para PyQt
│   ↓
│   Emitir change_pixmap_signal
│   ↓
└─ Loop (mientras _run_flag = True)
```

#### 🎯 Métodos Clave

##### `prepare_face(image_bgr, face_coords)`
Prepara una región facial para la inferencia del modelo.

**Proceso:**
1. Extrae ROI (Region of Interest) del rostro
2. Convierte de BGR a RGB
3. Redimensiona a 145x145 píxeles
4. Normaliza valores a rango [0, 1]
5. Reshape a formato del modelo: `(1, 145, 145, 3)`

##### `detect_eyes(face_gray, face_coords)`
Detecta la presencia de ojos en un rostro usando HaarCascade.

**Configuración:**
```python
eyes = self.eye_cascade.detectMultiScale(
    roi_gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(20, 20)
)
return len(eyes) >= 1  # True si detecta al menos 1 ojo
```

##### `get_status_and_color(is_yawning, eyes_are_closed)`
Determina el estado del conductor y el color de alerta según prioridad:

1. 🔴 **Alerta Crítica**: Ojos cerrados por 3+ frames
2. 🟠 **Advertencia**: Bostezo detectado
3. 🟢 **Normal**: Estado atento

**Retorno:** Tupla `(mensaje_estado, color_hex)`

---

### 3. Clase `DrowsinessDetectionApp` (QMainWindow)

Maneja la **presentación visual** y la **interacción con el usuario**.

#### 🖼️ Layout de la Interfaz

Utiliza un diseño vertical (`QVBoxLayout`) con los siguientes componentes:

```
┌──────────────────────────────────────────────┐
│  🚗 Sistema de Detección de Somnolencia      │
├──────────────────────────────────────────────┤
│  ┌────────────────────────────────────────┐  │
│  │                                        │  │
│  │     Video en Vivo (640x480)           │  │
│  │                                        │  │
│  └────────────────────────────────────────┘  │
├──────────────────────────────────────────────┤
│  Estado Actual:                              │
│  ┌────────────────────────────────────────┐  │
│  │  ✓ Estado Normal: Atento               │  │
│  └────────────────────────────────────────┘  │
├──────────────────────────────────────────────┤
│  [▶ Iniciar Detección]  [⏹ Detener]         │
├──────────────────────────────────────────────┤
│  💡 Consejo: Mantén buena iluminación...    │
└──────────────────────────────────────────────┘
```

#### 🎨 Componentes de la UI

1. **Título**: Label con fuente Arial 20pt, negrita, color #2C3E50
2. **Frame de video**: QLabel con borde, fondo negro, tamaño mínimo 640x480
3. **Panel de estado**: Frame con fondo dinámico según alertas
4. **Botones de control**:
   - **Iniciar**: Verde (#27AE60), habilitado por defecto
   - **Detener**: Rojo (#E74C3C), deshabilitado por defecto
5. **Consejo informativo**: Label en cursiva con color gris

#### 🔌 Slots (Conectores de Señales)

##### `update_image(cv_img)`
Recibe el array numpy de la imagen desde `VideoThread`, lo convierte a `QPixmap` y lo muestra en el `QLabel`.

**Proceso:**
1. Extrae dimensiones (alto, ancho, canales)
2. Convierte a `QImage`
3. Escala manteniendo aspect ratio (`KeepAspectRatio`)
4. Actualiza el `QLabel` con el nuevo `QPixmap`

##### `update_status(status_text, color)`
Cambia dinámicamente el color y texto del panel de estado según las alertas recibidas del hilo de video.

**Estados posibles:**
- 🚨 **ALERTA CRÍTICA: Ojos Cerrados** → Rojo (#FF0000)
- ⚠️ **ADVERTENCIA: Bostezo Detectado** → Naranja (#FFA500)
- ✓ **Estado Normal: Atento** → Verde (#00FF00)
- ⌛ **Buscando rostro...** → Amarillo

##### `start_detection()`
Inicia el proceso de detección:
1. Deshabilita botón "Iniciar"
2. Habilita botón "Detener"
3. Inicia el thread de video

##### `stop_detection()`
Detiene el proceso de detección:
1. Detiene el thread de video
2. Habilita botón "Iniciar"
3. Deshabilita botón "Detener"
4. Limpia el label de video
5. Resetea estado a "Sistema Detenido"

---

## 🚀 Uso

1. **Asegúrate de tener tu modelo entrenado** guardado como `assets/drowiness.keras`.

2. **Ejecuta el script principal**:
   ```bash
   python src/main.py
   ```

3. **Haz clic en el botón "▶ Iniciar Detección"**.

4. **Permite el acceso a la cámara web**.

5. **Para detener el sistema**, presiona "⏹ Detener".

---

## ⚙️ Configuración de Parámetros

### Detección de Rostros

```python
faces = face_cascade.detectMultiScale(
    gray_frame, 
    scaleFactor=1.3,    # Mayor = más rápido pero menos preciso
    minNeighbors=5,     # Mayor = menos falsos positivos
    minSize=(30, 30)    # Tamaño mínimo del rostro en píxeles
)
```

### Detección de Ojos

```python
eyes = eye_cascade.detectMultiScale(
    roi_gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(20, 20)
)
```

### Umbrales de Alerta

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `EYES_CLOSED_THRESHOLD` | **3 frames** | Frames consecutivos sin ojos para alerta crítica |
| `IMG_SIZE` | **145 píxeles** | Tamaño de entrada del modelo (145x145) |
| Umbral de bostezo | **Probabilidad relativa** | `prob_bostezo > prob_no_bostezo` |

---

## 🔍 Formato de Salida del Modelo

El modelo de TensorFlow retorna un array con **2 probabilidades**:

```python
predictions = model.predict(prepared_face, verbose=0)
# Formato: [prob_no_bostezo, prob_bostezo]
# Ejemplo: [0.85, 0.15] → 85% Normal, 15% Bostezo

yawn_prob = predictions[0][1]      # Probabilidad de bostezo
no_yawn_prob = predictions[0][0]   # Probabilidad de no bostezo

# Clasificación
is_yawning = yawn_prob > no_yawn_prob
```

---

## ⚠️ Solución de Problemas Comunes

### ❌ Error "No se pudo abrir la cámara"
**Solución**: Verifica que ninguna otra aplicación (Zoom, Teams, Skype) esté usando la cámara.

```bash
# En Linux, verifica dispositivos de video
ls /dev/video*
```

### ❌ Error al cargar `drowiness.keras`
**Solución**: Asegúrate de que la ruta del archivo sea correcta respecto a `src/main.py`. El código sube un nivel desde `src/` para buscar `assets/`.

```
✓ Correcto:
DetecionDeFatiga/
├── src/main.py
└── assets/drowiness.keras

✗ Incorrecto:
DetecionDeFatiga/
├── main.py  (debería estar en src/)
└── assets/drowiness.keras
```

### 🐌 Lentitud en la detección
**Solución**: TensorFlow puede ser pesado para CPU. 

**Opciones:**
- Si tienes una **GPU NVIDIA** configurada, TensorFlow la usará automáticamente.
- Reduce la resolución de captura.
- Aumenta el intervalo de procesamiento.
- Considera usar TensorFlow Lite para dispositivos con recursos limitados.

### ❌ Error "HaarCascade not found"
**Solución**: Reinstala OpenCV con los datos incluidos:

```bash
pip uninstall opencv-python
pip install opencv-python opencv-contrib-python
```

---

## 🎨 Características de la Interfaz

### Estilos CSS/QSS Aplicados

**Botón Iniciar:**
```css
background-color: #27AE60 (Verde)
color: white
padding: 10px
border-radius: 5px
Hover: #229954
```

**Botón Detener:**
```css
background-color: #E74C3C (Rojo)
color: white
padding: 10px
border-radius: 5px
Hover: #C0392B
```

**Panel de Estado (Dinámico):**
- 🔴 Alerta Crítica: `#FF0000`
- 🟠 Advertencia: `#FFA500`
- 🟢 Normal: `#00FF00`
- ⚪ Detenido: `#BDC3C7`

---

## 🚀 Optimizaciones Implementadas

1. ⚡ **Thread Separado**: Evita congelar la UI durante procesamiento intensivo
2. 🔇 **Predicción Silenciosa**: `verbose=0` para no saturar logs
3. 📊 **Detección Jerárquica**: Primero rostro, luego ojos (más eficiente)
4. 🎯 **Normalización**: Valores [0, 1] mejoran rendimiento del modelo
5. 📐 **Escalado Proporcional**: Mantiene calidad visual sin distorsión

---

## 💡 Posibles Mejoras Futuras

- 🔊 **Alertas sonoras** cuando se detecte somnolencia crítica
- 📝 **Registro de eventos** (logs con timestamps)
- ⚙️ **Configuración de umbrales** desde la UI
- 📹 **Soporte multi-cámara**
- 👀 **Detección de distracción** (mirada fuera de la carretera)
- 📄 **Exportación de reportes** en PDF/CSV
- 🎛️ **Calibración personalizada** por usuario
- 🚗 **Integración con APIs** de telemetría vehicular

---

## 📝 Créditos

Desarrollado utilizando:

- **PyQt6** - Interfaz Gráfica
- **OpenCV** - Visión por Computadora
- **TensorFlow** - Deep Learning

---

**Autor**: Equipo de Desarrollo  
**Última actualización**: Noviembre 2025  
**Versión**: 1.0
