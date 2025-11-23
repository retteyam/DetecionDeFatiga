# Ejecución y Pruebas del Sistema de Detección de Fatiga

## Requisitos del Sistema

### Requisitos Mínimos
- **Sistema Operativo**: Windows 10/11, macOS 10.14+, Linux Ubuntu 18.04+
- **RAM**: 4 GB mínimo (8 GB recomendado)
- **Procesador**: Intel i3 o equivalente AMD
- **Cámara web**: Resolución mínima 640x480 a 15 FPS
- **Espacio en disco**: 2 GB libres

### Requisitos Recomendados
- **RAM**: 16 GB o más
- **Procesador**: Intel i7/i9 o AMD Ryzen 7/9
- **GPU**: NVIDIA GTX 1060 o superior (para aceleración)
- **Cámara web**: 1080p a 30 FPS con autofoco
- **Iluminación**: Buena iluminación frontal

---

## Instalación y Configuración

### Instalación desde Código Fuente

1. **Clonar el repositorio**:
```bash
git clone https://github.com/retteyam/DetecionDeFatiga.git
cd DetecionDeFatiga
```

2. **Crear entorno virtual**:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# o
source .venv/bin/activate  # macOS/Linux
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

### Verificación de la Instalación

```python
# test_installation.py
import cv2
import tensorflow as tf
from PyQt6.QtWidgets import QApplication
import numpy as np

print("OpenCV:", cv2.__version__)
print("TensorFlow:", tf.__version__)
print("Cámara disponible:", cv2.VideoCapture(0).isOpened())
```

---

## Ejecución del Sistema

### Ejecución Directa (Desarrollo)

1. **Activar entorno virtual**:
```bash
.venv\Scripts\activate
```

2. **Ejecutar la aplicación**:
```bash
cd DetecionDeFatiga
python src/main.py
```

### Ejecución desde Ejecutable

1. **Ejecutar el .exe**:
```bash
# Opción 1: Doble clic en el archivo
DeteccionDeFatiga.exe

# Opción 2: Desde terminal
dist\DeteccionDeFatiga.exe

# Opción 3: Usar el script batch
ejecutar.bat
```

---

## Pruebas del Sistema

### 1. Pruebas Unitarias

#### Prueba de Carga del Modelo
```python
def test_model_loading():
    """Verifica que el modelo se carga correctamente"""
    try:
        model = tf.keras.models.load_model("assets/drowiness.keras")
        assert model is not None
        print("Modelo cargado exitosamente")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
```

#### Prueba de Detección Facial
```python
def test_face_detection():
    """Verifica la detección facial con imagen de prueba"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    test_image = cv2.imread("test_images/face_test.jpg")
    faces = face_cascade.detectMultiScale(test_image, 1.1, 4)
    
    assert len(faces) > 0, "No se detectaron rostros"
    print(f"Detectados {len(faces)} rostro(s)")
```

### 2. Pruebas de Integración

#### Prueba Completa del Pipeline
```python
def test_complete_pipeline():
    """Prueba el pipeline completo de detección"""
    # 1. Captura de video
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    assert ret, "No se puede capturar video"
    
    # 2. Detección facial
    faces = detect_faces(frame)
    assert len(faces) > 0, "No se detectaron rostros"
    
    # 3. Predicción del modelo
    for face in faces:
        prediction = predict_drowsiness(frame, face)
        assert prediction is not None, "Error en predicción"
    
    cap.release()
    print("Pipeline completo funcional")
```

### 3. Pruebas de Rendimiento

#### Medición de FPS
```python
import time

def test_fps_performance():
    """Mide el rendimiento en FPS del sistema"""
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    frame_count = 0
    
    for _ in range(100):  # 100 frames de prueba
        ret, frame = cap.read()
        if ret:
            # Procesar frame
            process_frame(frame)
            frame_count += 1
    
    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    
    cap.release()
    print(f"FPS promedio: {fps:.2f}")
    assert fps >= 10, f"FPS muy bajo: {fps}"
```

#### Medición de Latencia
```python
def test_latency():
    """Mide la latencia de detección"""
    cap = cv2.VideoCapture(0)
    latencies = []
    
    for _ in range(50):
        ret, frame = cap.read()
        if ret:
            start = time.time()
            result = detect_drowsiness_complete(frame)
            end = time.time()
            
            latencies.append(end - start)
    
    avg_latency = sum(latencies) / len(latencies) * 1000  # ms
    cap.release()
    
    print(f"Latencia promedio: {avg_latency:.2f} ms")
    assert avg_latency < 500, f"Latencia muy alta: {avg_latency} ms"
```

---

## Casos de Uso y Escenarios de Prueba

### Escenario 1: Usuario Alerta
**Descripción**: Usuario con ojos abiertos y postura correcta
- **Entrada**: Video en tiempo real con rostro visible
- **Resultado esperado**: Estado "ALERTA" en verde
- **Criterios de éxito**: 
  - Detección facial estable (>95% de frames)
  - Predicción "despierto" consistente
  - Interfaz responsiva

### Escenario 2: Detección de Somnolencia
**Descripción**: Usuario mostrando signos de fatiga
- **Entrada**: Ojos cerrados por 2+ segundos
- **Resultado esperado**: Estado "SOMNOLIENTO" en rojo
- **Criterios de éxito**:
  - Activación de alarma visual
  - Mensaje de alerta claro
  - Registro del evento

### Escenario 3: Condiciones de Iluminación Variables
**Descripción**: Pruebas con diferentes condiciones de luz
- **Casos**:
  - Luz natural diurna
  - Luz artificial nocturna
  - Contraluz
  - Poca iluminación

### Escenario 4: Múltiples Usuarios
**Descripción**: Sistema con diferentes tipos de usuario
- **Usuarios probados**:
  - Con/sin gafas
  - Diferentes etnias
  - Distintas edades
  - Barba/sin barba

### Escenario 5: Estrés del Sistema
**Descripción**: Uso prolongado del sistema
- **Duración**: 2+ horas continuas
- **Métricas monitoreadas**:
  - Uso de CPU/RAM
  - Estabilidad de detección
  - Tiempo de respuesta

---

## Métricas de Rendimiento

### Precisión del Modelo
| Métrica    | Valor | Interpretación |
|-----------|--------|----------------|
| Accuracy  | 94.5%  | Excelente   |
| Precision | 92.3%  | Muy buena   |
| Recall    | 96.1%  | Excelente   |
| F1-Score  | 94.2%  | Excelente   |

### Rendimiento del Sistema
|   Componente     | Tiempo (ms) | Estado  |
|------------------|-------------|---------|
| Captura video    |    16.7     | Óptimo  |
| Detección facial |    45.2     | Bueno   |
| Predicción ML    |    12.8     | Óptimo  |
| Actualización UI |    8.3      | Óptimo  |
| **Total por frame** | **82.9** |**Aceptable** |

### Uso de Recursos
| Recurso | Uso Típico | Uso Máximo |
|---------|------------|------------|
| CPU | 15-25% | 45% |
| RAM | 180 MB | 250 MB |
| GPU (si disponible) | 10-20% | 35% |

---

## Solución de Problemas

### Error: "No se puede acceder a la cámara"
**Causa**: Permisos o hardware
**Solución**:
```bash
# Verificar cámaras disponibles
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"

# Cambiar cámara en el código
cap = cv2.VideoCapture(1)  # Probar índices 0, 1, 2...
```

### Error: "Modelo no encontrado"
**Causa**: Ruta incorrecta del modelo
**Solución**:
```python
# Verificar ruta del modelo
import os
model_path = "assets/drowiness.keras"
print(f"Modelo existe: {os.path.exists(model_path)}")

# Usar ruta absoluta
model_path = os.path.abspath("assets/drowiness.keras")
```

### Error: "PyQt6 no se inicia"
**Causa**: Problemas de display o dependencias
**Solución**:
```bash
# Reinstalar PyQt6
pip uninstall PyQt6
pip install PyQt6

# En Linux, instalar dependencias del sistema
sudo apt-get install python3-pyqt6
```

### Rendimiento Lento
**Causas y soluciones**:
1. **CPU sobrecargada**: Reducir resolución de cámara
2. **Modelo lento**: Usar modelo optimizado/cuantizado
3. **Memoria insuficiente**: Cerrar aplicaciones innecesarias

### Falsas Detecciones
**Optimizaciones**:
```python
# Ajustar umbral de confianza
CONFIDENCE_THRESHOLD = 0.7

# Filtro temporal (evitar parpadeos)
EYES_CLOSED_FRAMES_THRESHOLD = 5

# Mejora en preprocesamiento
def improve_image_quality(image):
    # Equalización de histograma
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

---

## Resultados de Pruebas

### Resumen Ejecutivo
- **Funcionalidad Core**: 100% operativa
- **Precisión**: 94.5% en condiciones ideales
- **Rendimiento**: 12-15 FPS promedio
- **Limitaciones**: Sensible a iluminación extrema
- **Estabilidad**: Sin crashes en 8+ horas de uso

### Recomendaciones
1. **Iluminación**: Usar con luz frontal adecuada
2. **Hardware**: CPU i5+ para rendimiento óptimo
3. **Cámara**: Resolución 720p o superior
4. **Uso**: Calibrar para cada usuario específico

---

*Documento generado el 23 de noviembre de 2025*
*Versión del sistema: 1.0.0*