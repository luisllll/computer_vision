# src/main.py - Aplicación Principal
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QGroupBox, QGridLayout, QStatusBar, QSpinBox,
                             QFileDialog, QMessageBox, QInputDialog, QComboBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import json
import os

# Importar módulos personalizados
from yolo_detector import YOLODetector
from ruler_calibrator import RulerCalibrator
from measurement_processor import MeasurementProcessor

class YOLOMeasurementApp(QMainWindow):
    """
    Aplicación principal para medición de objetos usando YOLO
    """
    def __init__(self):
        super().__init__()
        
        # Configuración inicial
        self.webcam_id = 1
        self.roi = None
        self.mode = "normal"  # normal, calibration, roi_selection, measuring
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
        # Configuración de medición
        self.min_length_mm = 15
        self.max_length_mm = 30
        self.confidence_threshold = 0.3
        
        # Inicializar módulos
        self.yolo_detector = YOLODetector()
        self.ruler_calibrator = RulerCalibrator()
        self.measurement_processor = MeasurementProcessor()
        
        # Cámara
        self.capture = None
        
        # Configurar interfaz
        self.initUI()
        
        # Iniciar cámara
        self.start_camera()
        
        # Timer para frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS
    
    def initUI(self):
        """Inicializar interfaz de usuario"""
        self.setWindowTitle("Sistema YOLO de Medición de Almendras")
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Panel izquierdo (controles)
        left_panel = self._create_control_panel()
        left_panel.setFixedWidth(350)
        
        # Panel derecho (visualización)
        right_panel = self._create_display_panel()
        
        # Añadir paneles al layout principal
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)
        
        # Barra de estado
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Sistema listo - Selecciona modelo YOLO")
        
        # Conectar señales
        self._connect_signals()
    
    def _create_control_panel(self):
        """Crear panel de controles"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Grupo de modelo YOLO
        model_group = QGroupBox("Modelo YOLO")
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolov8n.pt (6MB - Rápido)",
            "yolov8s.pt (22MB - Recomendado)",
            "yolov8m.pt (50MB - Alta precisión)",
            "yolo11s.pt (25MB - Más reciente)"
        ])
        self.model_combo.setCurrentIndex(1)  # yolov8s por defecto
        
        self.load_model_btn = QPushButton("Cargar Modelo")
        self.model_status_label = QLabel("Modelo no cargado")
        
        model_layout.addWidget(QLabel("Seleccionar modelo:"))
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.load_model_btn)
        model_layout.addWidget(self.model_status_label)
        model_group.setLayout(model_layout)
        
        # Grupo de calibración
        calibration_group = QGroupBox("Calibración con Regla")
        calibration_layout = QVBoxLayout()
        
        self.ruler_length_spin = QSpinBox()
        self.ruler_length_spin.setRange(10, 500)
        self.ruler_length_spin.setValue(100)
        self.ruler_length_spin.setSuffix(" mm")
        
        self.calibrate_btn = QPushButton("Calibrar con Regla")
        self.calibration_status_label = QLabel("No calibrado")
        
        calibration_layout.addWidget(QLabel("Longitud de segmento de regla:"))
        calibration_layout.addWidget(self.ruler_length_spin)
        calibration_layout.addWidget(self.calibrate_btn)
        calibration_layout.addWidget(self.calibration_status_label)
        calibration_group.setLayout(calibration_layout)
        
        # Grupo de ROI
        roi_group = QGroupBox("Área de Medición (ROI)")
        roi_layout = QVBoxLayout()
        
        self.select_roi_btn = QPushButton("Seleccionar Área")
        self.clear_roi_btn = QPushButton("Limpiar Área")
        self.roi_status_label = QLabel("Área completa")
        
        roi_layout.addWidget(self.select_roi_btn)
        roi_layout.addWidget(self.clear_roi_btn)
        roi_layout.addWidget(self.roi_status_label)
        roi_group.setLayout(roi_layout)
        
        # Grupo de configuración de detección
        detection_group = QGroupBox("Configuración de Detección")
        detection_layout = QGridLayout()
        
        # Confianza mínima
        detection_layout.addWidget(QLabel("Confianza mínima:"), 0, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 90)
        self.confidence_slider.setValue(int(self.confidence_threshold * 100))
        detection_layout.addWidget(self.confidence_slider, 0, 1)
        self.confidence_label = QLabel(f"{self.confidence_threshold:.2f}")
        detection_layout.addWidget(self.confidence_label, 0, 2)
        
        detection_group.setLayout(detection_layout)
        
        # Grupo de rango de medición
        range_group = QGroupBox("Rango de Longitud Válida")
        range_layout = QGridLayout()
        
        range_layout.addWidget(QLabel("Mínimo (mm):"), 0, 0)
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(5, 100)
        self.min_length_spin.setValue(self.min_length_mm)
        range_layout.addWidget(self.min_length_spin, 0, 1)
        
        range_layout.addWidget(QLabel("Máximo (mm):"), 1, 0)
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(10, 200)
        self.max_length_spin.setValue(self.max_length_mm)
        range_layout.addWidget(self.max_length_spin, 1, 1)
        
        range_group.setLayout(range_layout)
        
        # Grupo de medición
        measurement_group = QGroupBox("Control de Medición")
        measurement_layout = QVBoxLayout()
        
        self.measure_btn = QPushButton("Iniciar Medición")
        self.measure_btn.setCheckable(True)
        self.measure_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
            }
        """)
        
        measurement_layout.addWidget(self.measure_btn)
        measurement_group.setLayout(measurement_layout)
        
        # Grupo de configuración
        config_group = QGroupBox("Configuración")
        config_layout = QVBoxLayout()
        
        self.save_config_btn = QPushButton("Guardar Configuración")
        self.load_config_btn = QPushButton("Cargar Configuración")
        
        config_layout.addWidget(self.save_config_btn)
        config_layout.addWidget(self.load_config_btn)
        config_group.setLayout(config_layout)
        
        # Estadísticas
        self.stats_label = QLabel("Estadísticas:\nNo hay mediciones")
        self.stats_label.setAlignment(Qt.AlignTop)
        self.stats_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        
        # Añadir grupos al layout
        layout.addWidget(model_group)
        layout.addWidget(calibration_group)
        layout.addWidget(roi_group)
        layout.addWidget(detection_group)
        layout.addWidget(range_group)
        layout.addWidget(measurement_group)
        layout.addWidget(config_group)
        layout.addWidget(QLabel("Estadísticas:"))
        layout.addWidget(self.stats_label)
        layout.addStretch()
        
        return panel
    
    def _create_display_panel(self):
        """Crear panel de visualización"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Vista de cámara principal
        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(800, 600)
        self.camera_view.setStyleSheet("border: 2px solid #ccc;")
        self.camera_view.mousePressEvent = self.camera_mouse_press
        self.camera_view.mouseMoveEvent = self.camera_mouse_move
        self.camera_view.mouseReleaseEvent = self.camera_mouse_release
        
        # Vista de detecciones (miniatura)
        detection_layout = QHBoxLayout()
        detection_layout.addWidget(QLabel("Vista de Detecciones:"))
        detection_layout.addStretch()
        
        self.detection_view = QLabel()
        self.detection_view.setAlignment(Qt.AlignCenter)
        self.detection_view.setFixedSize(320, 240)
        self.detection_view.setStyleSheet("border: 1px solid #ccc;")
        
        layout.addWidget(QLabel("Vista Principal:"))
        layout.addWidget(self.camera_view)
        layout.addLayout(detection_layout)
        layout.addWidget(self.detection_view)
        
        return panel
    
    def _connect_signals(self):
        """Conectar señales de la interfaz"""
        self.load_model_btn.clicked.connect(self.load_yolo_model)
        self.calibrate_btn.clicked.connect(self.start_calibration)
        self.select_roi_btn.clicked.connect(self.start_roi_selection)
        self.clear_roi_btn.clicked.connect(self.clear_roi)
        self.measure_btn.clicked.connect(self.toggle_measurement)
        self.save_config_btn.clicked.connect(self.save_configuration)
        self.load_config_btn.clicked.connect(self.load_configuration)
        
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        self.min_length_spin.valueChanged.connect(self.update_min_length)
        self.max_length_spin.valueChanged.connect(self.update_max_length)
    
    def start_camera(self):
        """Iniciar cámara"""
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        
        self.capture = cv2.VideoCapture(self.webcam_id)
        if not self.capture.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo abrir la cámara")
            return False
        
        # Configurar resolución
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.statusBar.showMessage("Cámara iniciada")
        return True
    
    def load_yolo_model(self):
        """Cargar modelo YOLO seleccionado"""
        model_text = self.model_combo.currentText()
        model_name = model_text.split(" ")[0]  # Extraer solo el nombre del modelo
        
        self.statusBar.showMessage(f"Cargando modelo {model_name}...")
        
        try:
            success = self.yolo_detector.load_model(model_name)
            if success:
                self.model_status_label.setText(f"✅ {model_name} cargado")
                self.statusBar.showMessage(f"Modelo {model_name} listo")
            else:
                self.model_status_label.setText("❌ Error al cargar")
                self.statusBar.showMessage("Error cargando modelo")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error cargando modelo: {str(e)}")
            self.model_status_label.setText("❌ Error al cargar")
    
    def start_calibration(self):
        """Iniciar modo de calibración"""
        if not self.yolo_detector.model_loaded:
            QMessageBox.warning(self, "Aviso", "Primero carga un modelo YOLO")
            return
        
        self.mode = "calibration"
        self.measure_btn.setChecked(False)
        self.start_point = None
        self.end_point = None
        self.statusBar.showMessage("Modo calibración: Dibuja una línea sobre una regla")
    
    def start_roi_selection(self):
        """Iniciar selección de ROI"""
        self.mode = "roi_selection"
        self.measure_btn.setChecked(False)
        self.start_point = None
        self.end_point = None
        self.statusBar.showMessage("Selección de área: Dibuja un rectángulo")
    
    def clear_roi(self):
        """Limpiar ROI"""
        self.roi = None
        self.roi_status_label.setText("Área completa")
        self.statusBar.showMessage("Área de medición limpiada")
    
    def toggle_measurement(self, checked):
        """Alternar modo de medición"""
        if not self.yolo_detector.model_loaded:
            QMessageBox.warning(self, "Aviso", "Primero carga un modelo YOLO")
            self.measure_btn.setChecked(False)
            return
        
        if not self.ruler_calibrator.is_calibrated():
            result = QMessageBox.question(
                self, "No calibrado", 
                "El sistema no está calibrado. ¿Continuar con mediciones en píxeles?",
                QMessageBox.Yes | QMessageBox.No
            )
            if result == QMessageBox.No:
                self.measure_btn.setChecked(False)
                return
        
        if checked:
            self.mode = "measuring"
            self.measure_btn.setText("Detener Medición")
            self.statusBar.showMessage("🔍 Midiendo almendras...")
        else:
            self.mode = "normal"
            self.measure_btn.setText("Iniciar Medición")
            self.statusBar.showMessage("Medición detenida")
    
    def update_confidence(self, value):
        """Actualizar umbral de confianza"""
        self.confidence_threshold = value / 100.0
        self.confidence_label.setText(f"{self.confidence_threshold:.2f}")
        self.yolo_detector.set_confidence_threshold(self.confidence_threshold)
    
    def update_min_length(self, value):
        """Actualizar longitud mínima"""
        self.min_length_mm = value
        if self.min_length_mm > self.max_length_mm:
            self.max_length_mm = self.min_length_mm
            self.max_length_spin.setValue(self.max_length_mm)
    
    def update_max_length(self, value):
        """Actualizar longitud máxima"""
        self.max_length_mm = value
        if self.min_length_mm > self.max_length_mm:
            self.min_length_mm = self.max_length_mm
            self.min_length_spin.setValue(self.min_length_mm)
    
    def update_frame(self):
        """Actualizar frame de cámara"""
        if not self.capture or not self.capture.isOpened():
            return
        
        ret, frame = self.capture.read()
        if not ret:
            return
        
        display_frame = frame.copy()
        detection_frame = None
        
        # Procesar según el modo
        if self.mode == "calibration" and self.start_point and self.end_point:
            self._draw_calibration_line(display_frame)
        
        elif self.mode == "roi_selection" and self.start_point and self.end_point:
            self._draw_roi_rectangle(display_frame)
        
        elif self.mode in ["normal", "measuring"]:
            if self.roi:
                self._draw_roi_border(display_frame)
            
            if self.mode == "measuring":
                display_frame, detection_frame = self._process_measurements(frame, display_frame)
        
        # Mostrar información de estado
        self._draw_status_info(display_frame)
        
        # Actualizar vistas
        self._update_camera_view(display_frame)
        if detection_frame is not None:
            self._update_detection_view(detection_frame)
    
    def _draw_calibration_line(self, frame):
        """Dibujar línea de calibración"""
        cv2.line(frame, self.start_point, self.end_point, (0, 255, 0), 3)
        pixel_length = np.sqrt((self.end_point[0] - self.start_point[0])**2 + 
                              (self.end_point[1] - self.start_point[1])**2)
        cv2.putText(frame, f"Longitud: {pixel_length:.1f} px", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def _draw_roi_rectangle(self, frame):
        """Dibujar rectángulo de ROI"""
        cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 255), 2)
    
    def _draw_roi_border(self, frame):
        """Dibujar borde del ROI"""
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "ÁREA DE MEDICIÓN", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _process_measurements(self, original_frame, display_frame):
        """Procesar mediciones con YOLO"""
        # Obtener región de interés
        if self.roi:
            x1, y1, x2, y2 = self.roi
            roi_frame = original_frame[y1:y2, x1:x2]
            roi_offset = (x1, y1)
        else:
            roi_frame = original_frame
            roi_offset = (0, 0)
        
        # Detectar almendras con YOLO
        detections = self.yolo_detector.detect_objects(roi_frame)
        
        # Procesar mediciones
        measurements = self.measurement_processor.process_detections(
            detections, 
            self.ruler_calibrator.get_mm_per_pixel(),
            self.min_length_mm,
            self.max_length_mm
        )
        
        # Dibujar resultados
        detection_frame = self._draw_detections(roi_frame.copy(), detections, measurements)
        self._draw_detections_on_main(display_frame, detections, measurements, roi_offset)
        
        # Actualizar estadísticas
        self._update_statistics(measurements)
        
        return display_frame, detection_frame
    
    def _draw_detections(self, frame, detections, measurements):
        """Dibujar detecciones en frame ROI"""
        for detection, measurement in zip(detections, measurements):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Color según si está en rango
            color = (0, 255, 0) if measurement['in_range'] else (0, 0, 255)
            
            # Dibujar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar etiqueta
            if measurement['length_mm'] is not None:
                label = f"{measurement['length_mm']:.1f}mm ({confidence:.2f})"
            else:
                label = f"{measurement['length_px']:.0f}px ({confidence:.2f})"
            
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def _draw_detections_on_main(self, frame, detections, measurements, roi_offset):
        """Dibujar detecciones en frame principal"""
        offset_x, offset_y = roi_offset
        
        for i, (detection, measurement) in enumerate(zip(detections, measurements)):
            x1, y1, x2, y2 = detection['bbox']
            
            # Ajustar coordenadas por offset del ROI
            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y
            
            # Color según si está en rango
            color = (0, 255, 0) if measurement['in_range'] else (0, 0, 255)
            
            # Dibujar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar ID
            cv2.putText(frame, f"A{i+1}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_status_info(self, frame):
        """Dibujar información de estado"""
        h, w = frame.shape[:2]
        
        # Información de calibración
        if self.ruler_calibrator.is_calibrated():
            calib_text = f"Calibrado: {self.ruler_calibrator.get_mm_per_pixel():.4f} mm/px"
            color = (0, 255, 0)
        else:
            calib_text = "No calibrado"
            color = (0, 0, 255)
        
        cv2.putText(frame, calib_text, (w-350, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Información del modelo
        if self.yolo_detector.model_loaded:
            model_text = f"Modelo: {self.yolo_detector.model_name}"
            cv2.putText(frame, model_text, (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def _update_statistics(self, measurements):
        """Actualizar estadísticas"""
        if not measurements:
            self.stats_label.setText("Estadísticas:\nNo hay detecciones")
            return
        
        total = len(measurements)
        in_range = sum(1 for m in measurements if m['in_range'])
        out_range = total - in_range
        
        if total > 0:
            percentage = (in_range / total) * 100
            
            lengths = [m['length_mm'] for m in measurements if m['length_mm'] is not None]
            if lengths:
                avg_length = np.mean(lengths)
                min_length = min(lengths)
                max_length = max(lengths)
                
                stats_text = f"""Estadísticas:
Total detectadas: {total}
En rango: {in_range}
Fuera de rango: {out_range}
Porcentaje válido: {percentage:.1f}%

Longitud promedio: {avg_length:.1f}mm
Rango detectado: {min_length:.1f}-{max_length:.1f}mm
Rango válido: {self.min_length_mm}-{self.max_length_mm}mm"""
            else:
                stats_text = f"""Estadísticas:
Total detectadas: {total}
En rango: {in_range}
Fuera de rango: {out_range}
Porcentaje válido: {percentage:.1f}%

(Mediciones en píxeles)"""
        else:
            stats_text = "Estadísticas:\nNo hay mediciones válidas"
        
        self.stats_label.setText(stats_text)
    
    def _update_camera_view(self, frame):
        """Actualizar vista de cámara"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.camera_view.width(), self.camera_view.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_view.setPixmap(pixmap)
    
    def _update_detection_view(self, frame):
        """Actualizar vista de detecciones"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.detection_view.width(), self.detection_view.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.detection_view.setPixmap(pixmap)
    
    def camera_mouse_press(self, event):
        """Manejo de clic en cámara"""
        if self.mode in ["calibration", "roi_selection"]:
            self.drawing = True
            self.start_point = self._get_frame_coordinates(event.x(), event.y())
            self.end_point = self.start_point
    
    def camera_mouse_move(self, event):
        """Manejo de movimiento del mouse"""
        if self.drawing and self.mode in ["calibration", "roi_selection"]:
            self.end_point = self._get_frame_coordinates(event.x(), event.y())
    
    def camera_mouse_release(self, event):
        """Manejo de liberación del mouse"""
        if self.drawing and self.mode in ["calibration", "roi_selection"]:
            self.drawing = False
            self.end_point = self._get_frame_coordinates(event.x(), event.y())
            
            if self.mode == "calibration":
                self._complete_calibration()
            elif self.mode == "roi_selection":
                self._complete_roi_selection()
    
    def _get_frame_coordinates(self, x, y):
        """Convertir coordenadas de pantalla a coordenadas de frame"""
        if not self.capture or not self.capture.isOpened():
            return (0, 0)
        
        frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        display_width = self.camera_view.width()
        display_height = self.camera_view.height()
        
        scale_x = frame_width / display_width
        scale_y = frame_height / display_height
        
        return (int(x * scale_x), int(y * scale_y))
    
    def _complete_calibration(self):
        """Completar calibración"""
        pixel_length = np.sqrt((self.end_point[0] - self.start_point[0])**2 + 
                              (self.end_point[1] - self.start_point[1])**2)
        
        if pixel_length < 10:
            QMessageBox.warning(self, "Error", "Línea muy corta para calibración")
            self.mode = "normal"
            return
        
        ruler_length_mm = self.ruler_length_spin.value()
        
        success = self.ruler_calibrator.calibrate_with_line(
            self.start_point, self.end_point, ruler_length_mm
        )
        
        if success:
            mm_per_pixel = self.ruler_calibrator.get_mm_per_pixel()
            self.calibration_status_label.setText(f"✅ {mm_per_pixel:.4f} mm/px")
            self.statusBar.showMessage(f"Calibración exitosa: {mm_per_pixel:.4f} mm/px")
        else:
            self.calibration_status_label.setText("❌ Error en calibración")
            self.statusBar.showMessage("Error en calibración")
        
        self.mode = "normal"
    
    def _complete_roi_selection(self):
        """Completar selección de ROI"""
        x1 = min(self.start_point[0], self.end_point[0])
        y1 = min(self.start_point[1], self.end_point[1])
        x2 = max(self.start_point[0], self.end_point[0])
        y2 = max(self.start_point[1], self.end_point[1])
        
        if abs(x2 - x1) < 50 or abs(y2 - y1) < 50:
            QMessageBox.warning(self, "Error", "Área muy pequeña")
            self.mode = "normal"
            return
        
        self.roi = (x1, y1, x2, y2)
        self.roi_status_label.setText(f"({x1}, {y1}) - ({x2}, {y2})")
        self.statusBar.showMessage(f"Área seleccionada: {x2-x1}x{y2-y1} píxeles")
        self.mode = "normal"
    
    def save_configuration(self):
        """Guardar configuración"""
        config = {
            'calibration_mm_per_pixel': self.ruler_calibrator.get_mm_per_pixel(),
            'roi': self.roi,
            'min_length_mm': self.min_length_mm,
            'max_length_mm': self.max_length_mm,
            'confidence_threshold': self.confidence_threshold,
            'model_name': self.yolo_detector.model_name,
            'ruler_length_mm': self.ruler_length_spin.value()
        }
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Configuración", 
            "configuracion_almendras.json", 
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=4)
                self.statusBar.showMessage(f"Configuración guardada: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error guardando: {str(e)}")
    
    def load_configuration(self):
        """Cargar configuración"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Configuración", 
            "", 
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                # Aplicar configuración
                if config.get('calibration_mm_per_pixel'):
                    self.ruler_calibrator.set_calibration(config['calibration_mm_per_pixel'])
                    self.calibration_status_label.setText(f"✅ {config['calibration_mm_per_pixel']:.4f} mm/px")
                
                if config.get('roi'):
                    self.roi = tuple(config['roi'])
                    x1, y1, x2, y2 = self.roi
                    self.roi_status_label.setText(f"({x1}, {y1}) - ({x2}, {y2})")
                
                self.min_length_mm = config.get('min_length_mm', 15)
                self.max_length_mm = config.get('max_length_mm', 30)
                self.confidence_threshold = config.get('confidence_threshold', 0.3)
                
                # Actualizar interfaz
                self.min_length_spin.setValue(self.min_length_mm)
                self.max_length_spin.setValue(self.max_length_mm)
                self.confidence_slider.setValue(int(self.confidence_threshold * 100))
                self.ruler_length_spin.setValue(config.get('ruler_length_mm', 100))
                
                self.statusBar.showMessage(f"Configuración cargada: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error cargando: {str(e)}")
    
    def closeEvent(self, event):
        """Manejar cierre de aplicación"""
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Verificar dependencias
    try:
        import ultralytics
        print("✅ Ultralytics disponible")
    except ImportError:
        QMessageBox.critical(None, "Error", 
                           "Ultralytics no está instalado.\n\n"
                           "Instala con: pip install ultralytics")
        sys.exit(1)
    
    # Crear y mostrar aplicación
    window = YOLOMeasurementApp()
    window.show()
    
    print("="*60)
    print("🌰 SISTEMA YOLO DE MEDICIÓN DE ALMENDRAS")
    print("="*60)
    print("1. Carga un modelo YOLO (recomendado: yolov8s.pt)")
    print("2. Calibra con una regla")
    print("3. Selecciona área de medición (opcional)")
    print("4. Inicia medición")
    print("="*60)
    
    sys.exit(app.exec_())