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
        self.IMG_SIZE = 145
        self.labels = ["Yawn", "No_yawn", "Closed", "Open"]
        
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
            print("✓ HaarCascade cargado exitosamente")
            
            if self.face_cascade.empty():
                raise Exception("No se pudo cargar el clasificador HaarCascade")
                
            return True
            
        except Exception as e:
            print(f"✗ Error al cargar recursos: {e}")
            self.status_signal.emit(f"Error: {str(e)}", "red")
            return False
    
    def prepare_face(self, image, face_coords):
        """
        Prepara la región del rostro para la inferencia.
        Aplica la misma lógica que en el notebook: resize y normalización.
        """
        try:
            x, y, w, h = face_coords
            roi = image[y:y+h, x:x+w]
            
            # Resize a 145x145 (como en el notebook)
            resized = cv2.resize(roi, (self.IMG_SIZE, self.IMG_SIZE))
            
            # Normalización (dividir por 255)
            normalized = resized / 255.0
            
            # Reshape para el modelo: (1, 145, 145, 3)
            return normalized.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)
            
        except Exception as e:
            print(f"Error en prepare_face: {e}")
            return None
    
    def get_status_and_color(self, prediction_idx):
        """
        Determina el estado y color según la predicción.
        0: Yawn (Bostezo) - Amarillo/Naranja (Advertencia)
        1: No_yawn (Sin bostezo) - Verde (Normal)
        2: Closed (Ojos cerrados) - Rojo (Alerta)
        3: Open (Ojos abiertos) - Verde (Normal)
        """
        if prediction_idx == 0:  # Yawn
            return "⚠️ ADVERTENCIA: Bostezo Detectado", "#FFA500"
        elif prediction_idx == 1:  # No_yawn
            return "✓ Normal: Sin Bostezo", "#00FF00"
        elif prediction_idx == 2:  # Closed (ojos cerrados)
            return "🚨 ALERTA: Ojos Cerrados", "#FF0000"
        elif prediction_idx == 3:  # Open
            return "✓ Normal: Ojos Abiertos", "#00FF00"
        else:
            return "Desconocido", "#FFFFFF"
    
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
            
            # Convertir a RGB para procesamiento
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                
                # Dibujar rectángulo en el rostro
                cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Preparar imagen para predicción
                prepared_face = self.prepare_face(rgb_frame, (x, y, w, h))
                
                if prepared_face is not None:
                    # Realizar predicción
                    predictions = self.model.predict(prepared_face, verbose=0)
                    prediction_idx = np.argmax(predictions[0])
                    confidence = predictions[0][prediction_idx]
                    
                    # Obtener estado y color
                    status, color = self.get_status_and_color(prediction_idx)
                    status_with_conf = f"{status} ({confidence*100:.1f}%)"
                    
                    # Emitir señal de estado
                    self.status_signal.emit(status_with_conf, color)
                    
                    # Agregar texto en el frame
                    label = f"{self.labels[prediction_idx]} ({confidence*100:.0f}%)"
                    cv2.putText(rgb_frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # No se detectó rostro
                self.status_signal.emit("⌛ Buscando rostro...", "yellow")
                cv2.putText(rgb_frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
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