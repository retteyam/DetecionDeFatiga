# Documentación del Proyecto - Sistema de Detección de Fatiga en Conductores

## Descripción General

Sistema de detección de fatiga en tiempo real utilizando visión por computadora para identificar signos de somnolencia en conductores. El sistema analiza imágenes faciales mediante una Red Neuronal Convolucional (CNN) para detectar ojos cerrados y bostezos, generando alertas oportunas para prevenir accidentes viales.

## Objetivos

- Detectar fatiga mediante análisis de señales faciales (ojos cerrados, bostezos)
- Clasificar automáticamente el estado del conductor en tiempo real
- Alertar al conductor mediante señales visuales y sonoras
- Prevenir accidentes causados por somnolencia al volante
- Lograr alta precisión (>95%) en la detección de señales de fatiga

## Tecnologías Utilizadas

- **Python 3.x:** Lenguaje principal de desarrollo
- **TensorFlow/Keras:** Framework de Deep Learning para la CNN
- **OpenCV:** Procesamiento de video e imágenes en tiempo real
- **NumPy:** Cálculos numéricos y manipulación de matrices
- **HaarCascade:** Detección facial y extracción de ROI
- **scikit-learn:** Métricas de evaluación y particionamiento de datos
- **Matplotlib:** Visualización de resultados y métricas
- **PIL/Pillow:** Manipulación de imágenes

## Arquitectura del Sistema

### Componentes Principales

1. **Módulo de Captura:** Captura video desde webcam en tiempo real (720p/30fps)
2. **Módulo de Detección:** Detecta rostros usando HaarCascade y extrae ROI facial
3. **Módulo de Preprocesamiento:** Redimensiona (145×145px) y normaliza imágenes [0-1]
4. **Módulo CNN:** Red neuronal que clasifica en 4 categorías (Open/Closed/Yawn/No_Yawn)
5. **Módulo de Análisis:** Compara confianza de predicción con umbrales configurados
6. **Módulo de Alerta:** Genera alertas visuales/sonoras y registra eventos

### Diagrama de Flujo

```
[Webcam] → [Captura Video] → [Detección Facial (HaarCascade)] → 
[Preprocesamiento ROI] → [Modelo CNN (drowsiness_new7.h5)] → 
[Clasificación 4 clases] → [Evaluación de Umbral] → [Sistema de Alertas]
                                                    ↓
                                            [Base de Datos]
```

## Estructura del Proyecto

```
DeteccionDeFatiga/
├── assets/                              # Recursos del proyecto
│   └── drowiness.keras
├── src/                                       # Código fuente
│   └── main.py                               # Script principal de ejecución
├── drowsiness_new7.h5                         # Modelo CNN entrenado (3.78 MB)
├── drowsiness_new7.keras                     # Modelo formato Keras
├── requirements.txt                          # Dependencias Python
├── .gitignore
├── Copia_de_driver_drowsiness_notebook.ipynb
├── README.md                                  # Documentación de usuario
├── Documentacion-Del-Proyecto.md              # Este documento
├── Ejecucion-y-Pruebas-del-Sistema.md         # Guía de ejecución
└── Codigo-Del-Sistema.md                      # Documentación del código
```

## Modelo de Red Neuronal Convolucional

### Arquitectura CNN

La red convolucional consta de 4 bloques Conv2D + MaxPooling, seguidos de capas densas:

| Capa | Tipo | Shape Salida | Parámetros |
|------|------|--------------|------------|
| conv2d_4 | Conv2D (256 filtros) | (143, 143, 256) | 7,168 |
| max_pooling2d_4 | MaxPooling2D | (71, 71, 256) | 0 |
| conv2d_5 | Conv2D (128 filtros) | (69, 69, 128) | 295,040 |
| max_pooling2d_5 | MaxPooling2D | (34, 34, 128) | 0 |
| conv2d_6 | Conv2D (64 filtros) | (32, 32, 64) | 73,792 |
| max_pooling2d_6 | MaxPooling2D | (16, 16, 64) | 0 |
| conv2d_7 | Conv2D (32 filtros) | (14, 14, 32) | 18,464 |
| max_pooling2d_7 | MaxPooling2D | (7, 7, 32) | 0 |
| flatten_1 | Flatten | (1568) | 0 |
| dropout_1 | Dropout (0.5) | (1568) | 0 |
| dense_2 | Dense (64 unidades) | (64) | 100,416 |
| dense_3 | Dense (4 unidades) | (4) | 260 |

**Total de parámetros:** 990,282 (3.78 MB)

### Hiperparámetros de Entrenamiento

- **Épocas:** 50
- **Optimizador:** Adam
- **Función de pérdida:** categorical_crossentropy
- **Regularización:** Dropout (0.5)
- **Data Augmentation:** zoom_range=0.2, horizontal_flip=True
- **Normalización:** rescale=1/255 (píxeles de [0-255] a [0-1])

## Dataset

### Composición del Dataset

| Categoría | Descripción | Cantidad |
|-----------|-------------|----------|
| **Open** | Ojos abiertos (estado alerta) | 726 imágenes |
| **Closed** | Ojos cerrados (microsueño) | 726 imágenes |
| **Yawn** | Expresión de bostezo | 725 imágenes |
| **No_Yawn** | Sin expresión de bostezo | 750 imágenes |

**Total:** 2,927 imágenes

### Particionamiento

- **Entrenamiento:** 70% (1,349 imágenes)
- **Prueba/Validación:** 30% (578 imágenes)
- **Semilla aleatoria:** 42 (para reproducibilidad)

## Configuración

### Parámetros Ajustables del Sistema

```python
# Umbrales de Detección
CONFIDENCE_THRESHOLD = 0.90    # Umbral de confianza para disparar alertas
EAR_THRESHOLD = 0.25          # Eye Aspect Ratio (no implementado en v1)
CONSEC_FRAMES = 20            # Frames consecutivos para activar alarma
MAR_THRESHOLD = 0.6           # Mouth Aspect Ratio (no implementado en v1)

# Configuración de Cámara
CAMERA_RESOLUTION = (720, 480) # Resolución de captura
FRAME_RATE = 30               # FPS objetivo

# Sistema de Alertas
ALERT_VOLUME = 70             # Volumen de alerta sonora (0-100)
ALERT_DURATION = 1            # Duración de alerta sonora (segundos)
```

## API y Funciones Principales

### `load_model(model_path)`
Carga el modelo CNN entrenado desde archivo .h5

- **Parámetros:** model_path (str) - Ruta al archivo del modelo
- **Retorna:** modelo Keras compilado
- **Ejemplo:** `model = load_model('drowsiness_new7.h5')`

### `preprocess_frame(frame)`
Preprocesa un frame de video para inferencia

- **Parámetros:** frame (numpy.array) - Frame BGR de OpenCV
- **Proceso:** Detección facial → Recorte ROI → Resize 145×145 → Normalización
- **Retorna:** numpy.array de forma (1, 145, 145, 3)

### `predict_fatigue(model, frame)`
Realiza predicción del estado de fatiga

- **Parámetros:** 
  - model: Modelo CNN cargado
  - frame: Frame preprocesado
- **Retorna:** tupla (clase_predicha, confianza)
- **Clases:** 0=Closed, 1=Open, 2=No_Yawn, 3=Yawn

### `trigger_alert(alert_type, confidence)`
Dispara alertas visuales y sonoras

- **Parámetros:**
  - alert_type (str): 'closed_eyes' o 'yawn'
  - confidence (float): Nivel de confianza [0-1]
- **Acciones:** Muestra banner, emite sonido, registra evento

### `face_for_yawn(img_path)`
Extrae ROI facial usando HaarCascade

- **Parámetros:** img_path (str) - Ruta a imagen
- **Proceso:** Detecta rostro → Recorta últimos N píxeles
- **Retorna:** numpy.array con ROI facial o None

## Resultados

### Métricas de Rendimiento del Modelo

| Métrica | Valor |
|---------|-------|
| **Accuracy Global** | 97.23% |
| **Loss (Prueba)** | 0.0697 |
| **F1-Score Macro** | 96.69% |
| **F1-Score Weighted** | 97.23% |

### Resultados por Clase

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Yawn** | 95.31% | 96.83% | 96.06% | 63 |
| **No_Yawn** | 95.89% | 94.59% | 95.24% | 74 |
| **Closed** | 96.38% | 99.07% | 97.71% | 215 |
| **Open** | 99.09% | 96.46% | 97.76% | 226 |

### Rendimiento en Tiempo Real

- **Tiempo de inferencia:** 1.21 ms por imagen
- **FPS alcanzado:** 829.41 frames/segundo
- **Requisito mínimo:** 30 FPS Superado ampliamente

### Matriz de Confusión

- **Predicciones correctas:** 562/578 (97.23%)
- **Errores totales:** 16 (2.77%)
- **Error más común:** Open → Closed (7 casos, por ojos entrecerrados/baja luz)

## Trabajo Futuro

### Mejoras a Corto Plazo
- [ ] Mejorar precisión en condiciones de baja iluminación
- [ ] Optimizar detección de estados intermedios (ojos entrecerrados)
- [ ] Implementar filtros adaptativos según condiciones de luz
- [ ] Añadir calibración por usuario

### Mejoras a Mediano Plazo
- [ ] Implementar detección de distracción (mirada fuera de la vía)
- [ ] Añadir análisis de posición de cabeza (cabeceos)
- [ ] Incorporar métricas de confianza temporal
- [ ] Desarrollar versión móvil (Android/iOS)

### Mejoras a Largo Plazo
- [ ] Añadir soporte para múltiples cámaras
- [ ] Fusión con telemetría del vehículo (velocidad, aceleración)
- [ ] Personalización del modelo por conductor
- [ ] Implementación en sistemas embebidos (Raspberry Pi, Jetson Nano)

## Autores

**Universidad Privada Antenor Orrego**  
Programa de Ingeniería de Sistemas e Inteligencia Artificial

- Díaz Uceda, Carlos
- Malca Delgado, Alba
- Mamani Azabache, Eduardo
- Merino Pérez, Grezia
- Valencia Galarreta, Maryory

**Docente:** Sagastegui Chigne, Teobaldo Hernán  
**Ciclo:** VI  
**Fecha:** Septiembre 2025

## Licencia

Proyecto académico desarrollado para el curso de Inteligencia Artificial - Principios y Técnicas.  
Universidad Privada Antenor Orrego, Trujillo, Perú.

## Referencias

1. **García Daza, I. (2011).** Detección de somnolencia mediante señales fisiológicas y parámetros de conducción. [Tesis doctoral]

2. **Clement, S., Vashistha, P., & Rane, M. (2015).** Driver Drowsiness Detection System using Visual Features. IEEE Conference.

3. **Man, S., & Hui-Ling, C. (2015).** Driver Fatigue Detection Based on Multiple Facial Features. Journal of Computer Science.

4. **Kong, W., Zhou, Z., Zhao, L., et al. (2015).** A System of Driving Fatigue Detection Based on Machine Learning. International Conference on Intelligent Computing.

5. **Park, S., Pan, F., Kang, S., & Yoo, C. (2017).** Driver Drowsiness Detection System Based on Feature Representation Learning using Deep Learning. Human-centric Computing and Information Sciences.

6. **Ale, L., & Junior, J. (2019).** Drowsiness Detection for Single Channel EEG by DWT Best m-Term Approximation. Pattern Recognition Letters.

7. **Documentación de TensorFlow/Keras:** https://www.tensorflow.org/api_docs

8. **Documentación de OpenCV:** https://docs.opencv.org/

---

**Versión:** 1.0  
**Última actualización:** Septiembre 2025  
**Estado:** Proyecto Completado