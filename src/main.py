"""
Aplicación de Detección de Somnolencia en Tiempo Real
Utiliza PyQt6 para la interfaz gráfica y OpenCV + TensorFlow para la detección
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QPushButton, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont
import tensorflow as tf


def resource_path(relative_path):
    """
    Obtiene la ruta absoluta del recurso, funciona tanto en desarrollo como en PyInstaller.
    Como main.py está en src/, debemos subir un nivel para llegar a assets/
    """
    try:
        # PyInstaller crea una carpeta temporal y almacena la ruta en _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # En desarrollo, obtenemos el directorio del script actual (src/)
        # y subimos un nivel para acceder a la raíz del proyecto
        base_path = Path(__file__).resolve().parent.parent
    
    return os.path.join(base_path, relative_path)


class VideoThread(QThread):
    """
    Thread separado para captura de video e inferencia del modelo.
    Evita congelar la interfaz gráfica.
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str, str)  # (estado, color)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.model = None
        self.face_cascade = None
        self.eye_cascade = None
        self.IMG_SIZE = 145
        self.eyes_closed_frames = 0
        
        # THRESHOLDS para clasificación (umbrales de confianza)
        self.YAWN_THRESHOLD = 0.85      # 85% confianza para bostezo (ajustado según logs)
        self.CLOSED_THRESHOLD = 0.65    # No se usa (HaarCascade en su lugar)
        
        # Configuración para detección de ojos con HaarCascade
        self.EYES_CLOSED_THRESHOLD = 3   # Frames consecutivos sin ojos para alertar (ajustado de 5 a 3)
        
    def load_resources(self):
        """Carga el modelo y el clasificador de rostros"""
        try:
            # Cargar modelo Keras
            model_path = resource_path("assets/drowiness.keras")
            print(f"Cargando modelo desde: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            print("✓ Modelo cargado exitosamente")
            
            # Cargar HaarCascade para detección facial
            cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
            face_cascade_path = os.path.join(cv2_base_dir, "data/haarcascade_frontalface_default.xml")
            
            if not os.path.exists(face_cascade_path):
                # Ruta alternativa
                face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            print("✓ HaarCascade para rostros cargado exitosamente")
            
            # Cargar HaarCascade para detección de ojos
            eye_cascade_path = os.path.join(cv2_base_dir, "data/haarcascade_eye.xml")
            if not os.path.exists(eye_cascade_path):
                eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            print("✓ HaarCascade para ojos cargado exitosamente")
            
            if self.face_cascade.empty():
                raise Exception("No se pudo cargar el clasificador HaarCascade de rostros")
            
            if self.eye_cascade.empty():
                raise Exception("No se pudo cargar el clasificador HaarCascade de ojos")
                
            return True
            
        except Exception as e:
            print(f"✗ Error al cargar recursos: {e}")
            self.status_signal.emit(f"Error: {str(e)}", "red")
            return False
    
    def prepare_face(self, image_bgr, face_coords):
        try:
            x, y, w, h = face_coords
            roi_bgr = image_bgr[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(roi_rgb, (self.IMG_SIZE, self.IMG_SIZE))
            normalized = resized.astype(np.float32) / 255.0
            return normalized.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)
        except Exception as e:
            print(f"Error en prepare_face: {e}")
            return None
    
    def detect_eyes(self, face_gray, face_coords):
        try:
            x, y, w, h = face_coords
            roi_gray = face_gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,   # Reducido de 1.1 a 1.05 para más precisión
                minNeighbors=3,      # Reducido de 5 a 3 para ser más sensible
                minSize=(15, 15)     # Reducido de (20,20) a (15,15) para detectar ojos más pequeños
            )
            return len(eyes) >= 1
        except Exception as e:
            print(f"Error en detect_eyes: {e}")
            return True
    
    def get_status_and_color(self, is_yawning, eyes_are_closed, closed_prob=0, yawn_prob=0):
        """
        Determina el estado de somnolencia basado en las predicciones del modelo.
        
        Estados:
        - Normal: Ojos abiertos, sin bostezar
        - Fatiga: Bostezo detectado (señal temprana)
        - Somnoliento: Ojos cerrados (peligro)
        - Peligro Extremo: Ambas señales combinadas
        """
        # PELIGRO EXTREMO: Ojos cerrados + bostezo
        if eyes_are_closed and is_yawning:
            return "🔴 PELIGRO EXTREMO: Somnoliento - Ojos Cerrados + Bostezo", "#8B0000"
        
        # CRÍTICO: Ojos cerrados (señal principal de somnolencia)
        if eyes_are_closed:
            return f"🚨 ALERTA CRÍTICA: Somnoliento - Ojos Cerrados ({closed_prob*100:.0f}%)", "#FF0000"
        
        # ADVERTENCIA: Bostezo (señal temprana de fatiga)
        if is_yawning:
            return f"⚠️ ADVERTENCIA: Fatiga - Bostezo Detectado ({yawn_prob*100:.0f}%)", "#FFA500"
        
        # NORMAL: Atento
        return "✅ Estado Normal: Atento", "#00FF00"
    
    def run(self):
        """Loop principal del thread de video"""
        # Cargar recursos
        if not self.load_resources():
            return
        
        # Inicializar captura de video
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Error: No se pudo abrir la cámara")
            self.status_signal.emit("Error: No se pudo abrir la cámara", "red")
            return
        
        print("✓ Cámara iniciada")
        self.status_signal.emit("Buscando rostro...", "yellow")
        
        while self._run_flag:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Error al capturar frame")
                continue
            
            # Convertir a escala de grises para detección de rostros/ojos
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros
            faces = self.face_cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=1.3, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Tomar el primer rostro detectado
                (x, y, w, h) = faces[0]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                eyes_detected = self.detect_eyes(gray_frame, (x, y, w, h))
                
                if not eyes_detected:
                    self.eyes_closed_frames += 1
                else:
                    self.eyes_closed_frames = 0
                
                eyes_are_closed = self.eyes_closed_frames >= self.EYES_CLOSED_THRESHOLD
                
                prepared_face = self.prepare_face(frame, (x, y, w, h))
                
                if prepared_face is not None:
                    predictions = self.model.predict(prepared_face, verbose=0)
                    
                    # Leer las 4 clases correctamente según el entrenamiento:
                    # Índice 0: yawn, Índice 1: no_yawn, Índice 2: Closed, Índice 3: Open
                    yawn_prob = predictions[0][0]      # Correcto: índice 0 = yawn
                    no_yawn_prob = predictions[0][1]   # Correcto: índice 1 = no_yawn
                    closed_prob = predictions[0][2]    # Nuevo: índice 2 = Closed
                    open_prob = predictions[0][3]      # Nuevo: índice 3 = Open
                    
                    # # DEBUG: Ver las predicciones en consola
                    # print(f"\n🔍 Predicciones del modelo:")
                    # print(f"  [0] yawn    : {yawn_prob*100:5.1f}%")
                    # print(f"  [1] no_yawn : {no_yawn_prob*100:5.1f}%")
                    # print(f"  [2] Closed  : {closed_prob*100:5.1f}%")
                    # print(f"  [3] Open    : {open_prob*100:5.1f}%")
                    
                    # ESTRATEGIA HÍBRIDA: Usar yawn/no_yawn del modelo + HaarCascade para ojos
                    # El modelo fue entrenado con rostros completos para yawn/no_yawn
                    # Pero con imágenes de ojos aislados para Closed/Open
                    # Por eso Closed/Open siempre dan 0% cuando pasamos rostro completo
                    
                    # Evaluar bostezos usando el modelo (funciona bien)
                    is_yawning = yawn_prob > self.YAWN_THRESHOLD
                    
                    # Evaluar ojos cerrados usando HaarCascade (más confiable en este caso)
                    eyes_detected_now = self.detect_eyes(gray_frame, (x, y, w, h))
                    
                    if not eyes_detected_now:
                        self.eyes_closed_frames += 1
                    else:
                        self.eyes_closed_frames = 0
                    
                    eyes_are_closed = self.eyes_closed_frames >= self.EYES_CLOSED_THRESHOLD
                    
                    # print(f"  → is_yawning: {is_yawning} (threshold: {self.YAWN_THRESHOLD})")
                    # print(f"  → eyes_closed: {eyes_are_closed} (HaarCascade frames: {self.eyes_closed_frames}/{self.EYES_CLOSED_THRESHOLD})")
                    
                    # Obtener estado y color basado en las predicciones del modelo
                    status, color = self.get_status_and_color(is_yawning, eyes_are_closed, closed_prob, yawn_prob)
                    self.status_signal.emit(status, color)
                    
                    # Visualización mejorada en el video
                    # Mostrar información de bostezos
                    yawn_label = f"Bostezo: {yawn_prob*100:.0f}%" if is_yawning else f"Normal: {no_yawn_prob*100:.0f}%"
                    yawn_color = (255, 165, 0) if is_yawning else (0, 255, 0)
                    cv2.putText(frame, yawn_label, (x, y-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, yawn_color, 2)
                    
                    # # Mostrar información de estado de ojos (usando HaarCascade)
                    # if eyes_are_closed:
                    #     eyes_label = f"Ojos: CERRADOS! (Frames: {self.eyes_closed_frames})"
                    #     eyes_color = (255, 0, 0)  # Rojo
                    # else:
                    #     eyes_label = f"Ojos: Abiertos (detectados: {eyes_detected_now})"
                    #     eyes_color = (0, 255, 0)  # Verde
                    
                    # cv2.putText(frame, eyes_label, (x, y-10), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, eyes_color, 2)
            else:
                # No se detectó rostro
                self.status_signal.emit("⌛ Buscando rostro...", "yellow")
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Convertir BGR a RGB para PyQt6
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Emitir frame procesado
            self.change_pixmap_signal.emit(rgb_frame)
        
        # Liberar recursos
        cap.release()
        print("✓ Cámara liberada")
    
    def stop(self):
        """Detiene el thread de forma segura"""
        self._run_flag = False
        self.wait()


class DrowsinessDetectionApp(QMainWindow):
    """
    Ventana principal de la aplicación de detección de somnolencia.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Detección de Somnolencia")
        self.setGeometry(100, 100, 1000, 700)
        
        # Thread de video
        self.video_thread = VideoThread()
        
        # Configurar UI
        self.init_ui()
        
        # Conectar señales
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.status_signal.connect(self.update_status)
        
    def init_ui(self):
        """Inicializa la interfaz de usuario"""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Título
        title_label = QLabel("🚗 Sistema de Detección de Somnolencia")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Arial", 20, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2C3E50; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # Frame del video
        video_frame = QFrame()
        video_frame.setFrameShape(QFrame.Shape.Box)
        video_frame.setLineWidth(2)
        video_frame.setStyleSheet("background-color: #000000;")
        
        video_layout = QVBoxLayout()
        video_frame.setLayout(video_layout)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #34495E;")
        video_layout.addWidget(self.video_label)
        
        main_layout.addWidget(video_frame)
        
        # Panel de estado
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        status_frame.setStyleSheet("background-color: #ECF0F1; border-radius: 10px;")
        
        status_layout = QVBoxLayout()
        status_frame.setLayout(status_layout)
        
        status_title = QLabel("Estado Actual:")
        status_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        status_layout.addWidget(status_title)
        
        self.status_label = QLabel("Sistema Iniciando...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_font = QFont("Arial", 16, QFont.Weight.Bold)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet("""
            padding: 20px;
            border-radius: 5px;
            background-color: #BDC3C7;
            color: #2C3E50;
        """)
        status_layout.addWidget(self.status_label)
        
        main_layout.addWidget(status_frame)
        
        # Botones
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("▶ Iniciar Detección")
        self.start_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.start_button.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹ Detener")
        self.stop_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        main_layout.addLayout(button_layout)
        
        # Información
        info_label = QLabel("💡 Consejo: Mantén buena iluminación y posiciona tu rostro frente a la cámara")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("color: #7F8C8D; font-style: italic; padding: 5px;")
        main_layout.addWidget(info_label)
    
    def update_image(self, cv_img):
        """Actualiza el frame de video en la interfaz"""
        try:
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            qt_image = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
            self.video_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error al actualizar imagen: {e}")
    
    def update_status(self, status_text, color):
        """Actualiza el texto y color del estado"""
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"""
            padding: 20px;
            border-radius: 5px;
            background-color: {color};
            color: white;
            font-weight: bold;
        """)
    
    def start_detection(self):
        """Inicia la detección"""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.video_thread.start()
    
    def stop_detection(self):
        """Detiene la detección"""
        self.video_thread.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Sistema Detenido")
        self.status_label.setStyleSheet("""
            padding: 20px;
            border-radius: 5px;
            background-color: #BDC3C7;
            color: #2C3E50;
        """)
        self.video_label.clear()
        self.video_label.setText("Video detenido")
    
    def closeEvent(self, event):
        """Maneja el cierre de la aplicación"""
        self.video_thread.stop()
        event.accept()


def main():
    """Función principal de la aplicación"""
    app = QApplication(sys.argv)
    
    # Estilo global
    app.setStyle('Fusion')
    
    # Crear y mostrar ventana principal
    window = DrowsinessDetectionApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()