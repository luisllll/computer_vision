import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QGroupBox, QGridLayout, QStatusBar, QSpinBox,
                             QFileDialog, QMessageBox, QInputDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class MeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initial configuration
        self.webcam_id = 1
        self.calibration_mm_per_pixel = None
        self.roi = None
        self.min_length_mm = 0
        self.max_length_mm = 100
        self.threshold_value = 128
        self.min_area = 50
        self.capture = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.mode = "normal"  # Modes: normal, calibration, roi_selection, measuring
        
        # Set up the user interface
        self.initUI()
        
        # Start the camera
        self.start_camera()
        
        # Start a timer to update frames (~33 fps)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
    
    def initUI(self):
        # Main window settings
        self.setWindowTitle("Object Measurement System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel (controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)
        
        # Calibration group
        calibration_group = QGroupBox("Calibration")
        calibration_layout = QVBoxLayout()
        self.calibrate_btn = QPushButton("Calibrate Camera")
        self.calibration_value_label = QLabel("Not calibrated")
        calibration_layout.addWidget(self.calibrate_btn)
        calibration_layout.addWidget(self.calibration_value_label)
        calibration_group.setLayout(calibration_layout)
        
        # ROI group
        roi_group = QGroupBox("Region of Interest (ROI)")
        roi_layout = QVBoxLayout()
        self.select_roi_btn = QPushButton("Select ROI")
        self.roi_value_label = QLabel("Not selected")
        roi_layout.addWidget(self.select_roi_btn)
        roi_layout.addWidget(self.roi_value_label)
        roi_group.setLayout(roi_layout)
        
        # Length range group
        range_group = QGroupBox("Length Range")
        range_layout = QGridLayout()
        range_layout.addWidget(QLabel("Minimum (mm):"), 0, 0)
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(0, 100)
        self.min_length_spin.setValue(self.min_length_mm)
        range_layout.addWidget(self.min_length_spin, 0, 1)
        
        range_layout.addWidget(QLabel("Maximum (mm):"), 1, 0)
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(0, 500)
        self.max_length_spin.setValue(self.max_length_mm)
        range_layout.addWidget(self.max_length_spin, 1, 1)
        range_group.setLayout(range_layout)
        
        # Processing settings group
        process_group = QGroupBox("Processing Settings")
        process_layout = QGridLayout()
        process_layout.addWidget(QLabel("Threshold:"), 0, 0)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(self.threshold_value)
        process_layout.addWidget(self.threshold_slider, 0, 1)
        self.threshold_value_label = QLabel(str(self.threshold_value))
        process_layout.addWidget(self.threshold_value_label, 0, 2)
        
        process_layout.addWidget(QLabel("Minimum Area:"), 1, 0)
        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(10, 500)
        self.min_area_slider.setValue(self.min_area)
        process_layout.addWidget(self.min_area_slider, 1, 1)
        self.min_area_value_label = QLabel(str(self.min_area))
        process_layout.addWidget(self.min_area_value_label, 1, 2)
        process_group.setLayout(process_layout)
        
        # Configuration group
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        self.save_config_btn = QPushButton("Save Configuration")
        self.load_config_btn = QPushButton("Load Configuration")
        config_layout.addWidget(self.save_config_btn)
        config_layout.addWidget(self.load_config_btn)
        config_group.setLayout(config_layout)
        
        # Measurement toggle button
        self.measure_btn = QPushButton("Start Measurement")
        self.measure_btn.setCheckable(True)
        
        # Add groups to the left panel
        left_layout.addWidget(calibration_group)
        left_layout.addWidget(roi_group)
        left_layout.addWidget(range_group)
        left_layout.addWidget(process_group)
        left_layout.addWidget(config_group)
        left_layout.addWidget(self.measure_btn)
        left_layout.addStretch()
        
        # Right panel (visualization)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Camera view label
        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(640, 480)
        self.camera_view.mousePressEvent = self.camera_mouse_press
        self.camera_view.mouseMoveEvent = self.camera_mouse_move
        self.camera_view.mouseReleaseEvent = self.camera_mouse_release
        
        # Threshold view label
        self.threshold_view = QLabel()
        self.threshold_view.setAlignment(Qt.AlignCenter)
        self.threshold_view.setMinimumSize(320, 240)
        
        # Statistics label (displayed outside the image)
        self.stats_label = QLabel("Statistics:")
        self.stats_label.setAlignment(Qt.AlignLeft)
        
        # Add views to the right panel
        right_layout.addWidget(QLabel("Camera View"))
        right_layout.addWidget(self.camera_view)
        right_layout.addWidget(QLabel("Threshold View"))
        right_layout.addWidget(self.threshold_view)
        right_layout.addWidget(self.stats_label)
        
        # Add panels to the main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)  # right panel takes remaining space
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Connect signals
        self.calibrate_btn.clicked.connect(self.start_calibration)
        self.select_roi_btn.clicked.connect(self.start_roi_selection)
        self.save_config_btn.clicked.connect(self.save_config)
        self.load_config_btn.clicked.connect(self.load_config)
        self.measure_btn.clicked.connect(self.toggle_measurement)
        
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.min_area_slider.valueChanged.connect(self.update_min_area)
        self.min_length_spin.valueChanged.connect(self.update_min_length)
        self.max_length_spin.valueChanged.connect(self.update_max_length)
    
    def start_camera(self):
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        
        self.capture = cv2.VideoCapture(self.webcam_id)
        if not self.capture.isOpened():
            QMessageBox.critical(self, "Error", "Could not open webcam. Check connection or camera ID.")
            self.statusBar.showMessage("Error: Webcam could not be opened")
        else:
            self.statusBar.showMessage("Camera started")
    
    def update_frame(self):
        """Update the camera frame and process the image if needed"""
        if self.capture is None or not self.capture.isOpened():
            return
        
        ret, frame = self.capture.read()
        if not ret:
            self.statusBar.showMessage("Error capturing frame")
            return
        
        # Create a copy for drawing overlays
        display_frame = frame.copy()
        threshold_frame = None
        
        # Calibration mode: draw calibration line
        if self.mode == "calibration" and self.start_point and self.end_point:
            cv2.line(display_frame, self.start_point, self.end_point, (0, 255, 0), 2)
            pixel_length = np.sqrt((self.end_point[0] - self.start_point[0])**2 + 
                                   (self.end_point[1] - self.start_point[1])**2)
            cv2.putText(display_frame, f"Length: {pixel_length:.1f} px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ROI selection mode: draw ROI rectangle
        elif self.mode == "roi_selection" and self.start_point and self.end_point:
            cv2.rectangle(display_frame, self.start_point, self.end_point, (0, 255, 0), 2)
        
        # Normal or measuring mode
        elif self.mode in ["normal", "measuring"]:
            if self.roi:
                x1, y1, x2, y2 = self.roi
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if self.mode == "measuring":
                    if y1 < y2 and x1 < x2 and y1 >= 0 and x1 >= 0 and y2 <= frame.shape[0] and x2 <= frame.shape[1]:
                        roi_frame = frame[y1:y2, x1:x2].copy()
                        
                        # Get current threshold and min_area values
                        threshold_value = self.threshold_value
                        min_area = self.min_area
                        
                        # Convert to grayscale and apply threshold
                        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
                        
                        # Apply morphological operations to improve segmentation
                        kernel = np.ones((3, 3), np.uint8)
                        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
                        
                        # Save threshold image for display
                        threshold_frame = binary
                        
                        # Find contours
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Filter out small contours (noise)
                        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
                        
                        # Counters for objects within and out of range
                        within_range_count = 0
                        outside_range_count = 0
                        
                        # Process each contour and draw bounding boxes on the main frame
                        for cnt in filtered_contours:
                            try:
                                # Get minimum area rotated rectangle
                                rect = cv2.minAreaRect(cnt)
                                box = cv2.boxPoints(rect)
                                box = np.int32(box)
                                
                                width, height = rect[1]
                                length_px = max(width, height)
                                
                                if self.calibration_mm_per_pixel:
                                    length_mm = length_px * self.calibration_mm_per_pixel
                                    unit = "mm"
                                    within_range = self.min_length_mm <= length_mm <= self.max_length_mm
                                    contour_color = (0, 255, 0) if within_range else (0, 0, 255)
                                    if within_range:
                                        within_range_count += 1
                                    else:
                                        outside_range_count += 1
                                else:
                                    length_mm = length_px
                                    unit = "px"
                                    contour_color = (255, 0, 0)
                                
                                # Adjust bounding box points relative to the main frame by adding the ROI offset
                                box[:, 0] += x1
                                box[:, 1] += y1
                                
                                cv2.drawContours(display_frame, [box], 0, contour_color, 2)
                                
                                # Calculate the center of the contour for placing text
                                M = cv2.moments(cnt)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"]) + x1
                                    cy = int(M["m01"] / M["m00"]) + y1
                                    text = f"{length_mm:.1f} {unit}"
                                    (text_width, text_height), baseline = cv2.getTextSize(
                                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                    cv2.rectangle(display_frame,
                                                  (cx - 5, cy - text_height - 5),
                                                  (cx + text_width + 5, cy + 5),
                                                  (0, 0, 0), -1)
                                    cv2.putText(display_frame, text, (cx, cy),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            except Exception as e:
                                print(f"Error processing contour: {e}")
                        
                        # MODIFICACIÓN: Eliminar las estadísticas dibujadas en la imagen
                        # Solo mantener la información de calibración
                        
                        # Actualizar las estadísticas en el widget externo
                        total_objects = len(filtered_contours)
                        if total_objects > 0:
                            percentage = (within_range_count / total_objects) * 100
                        else:
                            percentage = 0
                        
                        stats_text = (
                            f"Total objects: {total_objects}\n"
                            f"Within range: {within_range_count}\n"
                            f"Out of range: {outside_range_count}\n"
                            f"Percentage in range: {percentage:.1f}%\n"
                            f"Range: {self.min_length_mm}-{self.max_length_mm} mm"
                        )
                        self.stats_label.setText(stats_text)
                    else:
                        self.statusBar.showMessage("Error: ROI out of image bounds")
            else:
                cv2.putText(display_frame, "No ROI selected. Press 'Select ROI'.", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display calibration info (mantener esta información en la imagen)
            if self.calibration_mm_per_pixel:
                cv2.putText(display_frame, f"Calibration: {self.calibration_mm_per_pixel:.4f} mm/px", 
                            (display_frame.shape[1] - 350, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Not calibrated", (display_frame.shape[1] - 200, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Convert the frame to RGB and display it
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_view.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.camera_view.width(), self.camera_view.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Display the threshold image if available
        if threshold_frame is not None:
            threshold_rgb = cv2.cvtColor(threshold_frame, cv2.COLOR_GRAY2RGB)
            h, w, ch = threshold_rgb.shape
            bytes_per_line = ch * w
            q_img_threshold = QImage(threshold_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.threshold_view.setPixmap(QPixmap.fromImage(q_img_threshold).scaled(
                self.threshold_view.width(), self.threshold_view.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def camera_mouse_press(self, event):
        if self.mode in ["calibration", "roi_selection"]:
            self.drawing = True
            if self.capture and self.capture.isOpened():
                frame_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                display_width = self.camera_view.width()
                display_height = self.camera_view.height()
                scale_x = frame_width / display_width
                scale_y = frame_height / display_height
                x = int(event.x() * scale_x)
                y = int(event.y() * scale_y)
                self.start_point = (x, y)
                self.end_point = self.start_point
    
    def camera_mouse_move(self, event):
        if self.drawing and self.mode in ["calibration", "roi_selection"]:
            if self.capture and self.capture.isOpened():
                frame_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                display_width = self.camera_view.width()
                display_height = self.camera_view.height()
                scale_x = frame_width / display_width
                scale_y = frame_height / display_height
                x = int(event.x() * scale_x)
                y = int(event.y() * scale_y)
                self.end_point = (x, y)
    
    def camera_mouse_release(self, event):
        if self.drawing and self.mode in ["calibration", "roi_selection"]:
            self.drawing = False
            if self.capture and self.capture.isOpened():
                frame_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                display_width = self.camera_view.width()
                display_height = self.camera_view.height()
                scale_x = frame_width / display_width
                scale_y = frame_height / display_height
                x = int(event.x() * scale_x)
                y = int(event.y() * scale_y)
                self.end_point = (x, y)
            if self.mode == "calibration":
                pixel_length = np.sqrt((self.end_point[0] - self.start_point[0])**2 + 
                                       (self.end_point[1] - self.start_point[1])**2)
                if pixel_length > 10:
                    value, ok = QInputDialog.getDouble(self, "Calibration", 
                                                       "Enter real length in millimeters:",
                                                       10.0, 0.1, 500.0, 1)
                    if ok:
                        self.calibration_mm_per_pixel = value / pixel_length
                        self.calibration_value_label.setText(f"{self.calibration_mm_per_pixel:.4f} mm/px")
                        self.statusBar.showMessage(f"Calibration complete: {self.calibration_mm_per_pixel:.4f} mm/px")
                        self.mode = "normal"
            elif self.mode == "roi_selection":
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                self.roi = (int(x1), int(y1), int(x2), int(y2))
                self.roi_value_label.setText(f"({x1}, {y1}), ({x2}, {y2})")
                self.statusBar.showMessage(f"ROI selected: {self.roi}")
                self.mode = "normal"
    
    def start_calibration(self):
        """Activate calibration mode"""
        self.mode = "calibration"
        self.statusBar.showMessage("Calibration mode: Draw a line on an object with a known length")
        self.measure_btn.setChecked(False)
        self.start_point = None
        self.end_point = None
        self.drawing = False
    
    def start_roi_selection(self):
        """Activate ROI selection mode"""
        self.mode = "roi_selection"
        self.statusBar.showMessage("ROI selection mode: Draw a rectangle to select the measurement area")
        self.measure_btn.setChecked(False)
        self.start_point = None
        self.end_point = None
        self.drawing = False
    
    def update_threshold(self, value):
        """Update the threshold value"""
        self.threshold_value = value
        self.threshold_value_label.setText(str(value))
    
    def update_min_area(self, value):
        """Update the minimum area value"""
        self.min_area = value
        self.min_area_value_label.setText(str(value))
    
    def update_min_length(self, value):
        """Update the minimum length value"""
        self.min_length_mm = value
        if self.min_length_mm > self.max_length_mm:
            self.max_length_mm = self.min_length_mm
            self.max_length_spin.setValue(self.max_length_mm)
    
    def update_max_length(self, value):
        """Update the maximum length value"""
        self.max_length_mm = value
        if self.min_length_mm > self.max_length_mm:
            self.min_length_mm = self.max_length_mm
            self.min_length_spin.setValue(self.min_length_mm)
    
    def toggle_measurement(self, checked):
        """Toggle between normal and measuring mode"""
        if checked:
            self.mode = "measuring"
            self.measure_btn.setText("Stop Measurement")
            self.statusBar.showMessage("Measuring objects...")
            print("================================")
            print("MEASUREMENT MODE ACTIVATED")
            print(f"Calibration: {self.calibration_mm_per_pixel}")
            print(f"ROI: {self.roi}")
            print(f"Length Range: {self.min_length_mm} - {self.max_length_mm} mm")
            print("================================")
            if not self.calibration_mm_per_pixel:
                QMessageBox.warning(self, "Warning", 
                                    "Camera is not calibrated. Measurements will be shown in pixels.")
            if not self.roi:
                result = QMessageBox.question(self, "No ROI selected", 
                                              "No measurement area selected. Would you like to select one now?",
                                              QMessageBox.Yes | QMessageBox.No)
                if result == QMessageBox.Yes:
                    self.start_roi_selection()
                    self.measure_btn.setChecked(False)
                    return
        else:
            self.mode = "normal"
            self.measure_btn.setText("Start Measurement")
            self.statusBar.showMessage("Ready")
    
    def save_config(self):
        """Save the current configuration to a file"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", 
                                                   "measurement_params.txt", 
                                                   "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(f"CALIBRATION_MM_PER_PIXEL={self.calibration_mm_per_pixel}\n")
                    f.write(f"MIN_LENGTH_MM={self.min_length_mm}\n")
                    f.write(f"MAX_LENGTH_MM={self.max_length_mm}\n")
                    if self.roi:
                        f.write(f"ROI={self.roi[0]},{self.roi[1]},{self.roi[2]},{self.roi[3]}\n")
                self.statusBar.showMessage(f"Configuration saved to '{file_path}'")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving configuration: {e}")
    
    def load_config(self):
        """Load configuration from a file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", 
                                                   "", 
                                                   "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            if key == "CALIBRATION_MM_PER_PIXEL" and value != "None":
                                self.calibration_mm_per_pixel = float(value)
                                self.calibration_value_label.setText(f"{self.calibration_mm_per_pixel:.4f} mm/px")
                            elif key == "MIN_LENGTH_MM":
                                self.min_length_mm = int(float(value))
                                self.min_length_spin.setValue(self.min_length_mm)
                            elif key == "MAX_LENGTH_MM":
                                self.max_length_mm = int(float(value))
                                self.max_length_spin.setValue(self.max_length_mm)
                            elif key == "ROI":
                                roi_values = value.split(",")
                                if len(roi_values) == 4:
                                    self.roi = tuple(int(float(v)) for v in roi_values)
                                    self.roi_value_label.setText(f"({self.roi[0]}, {self.roi[1]}), ({self.roi[2]}, {self.roi[3]})")
                self.statusBar.showMessage(f"Configuration loaded from '{file_path}'")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading configuration: {e}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MeasurementApp()
    window.show()
    sys.exit(app.exec_())