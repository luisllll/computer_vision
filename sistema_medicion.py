import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import threading
import time
import os
from PIL import Image, ImageTk

# Configuración inicial
WEBCAM_ID = 1  # ID de la webcam (normalmente 0 para la webcam por defecto)
CALIBRATION_MM_PER_PIXEL = None  # Valor de calibración
ROI = None  # Región de interés (Region Of Interest)
MIN_LENGTH_MM = 1  # Longitud mínima del rango aceptable (en mm)
MAX_LENGTH_MM = 100  # Longitud máxima del rango aceptable (en mm)

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Medición de Objetos")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables para la interfaz
        self.cap = None
        self.thread = None
        self.thread_running = False
        self.current_frame = None
        self.binary_frame = None
        self.threshold_value = tk.IntVar(value=128)
        self.min_area = tk.IntVar(value=50)
        self.calibration_label = tk.StringVar(value="Sin calibrar")
        self.roi_label = tk.StringVar(value="Sin área seleccionada")
        self.length_range_label = tk.StringVar(value=f"Rango: {MIN_LENGTH_MM}-{MAX_LENGTH_MM} mm")
        self.total_objects = tk.StringVar(value="Total objetos: 0")
        self.within_range = tk.StringVar(value="Dentro de rango: 0")
        self.outside_range = tk.StringVar(value="Fuera de rango: 0")
        self.status_var = tk.StringVar(value="Listo")
        
        # Variables para el dibujo del ROI
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.drawing = False
        
        # Variables para calibración
        self.calibration_mode = False
        self.cal_start_x = None
        self.cal_start_y = None
        self.cal_end_x = None
        self.cal_end_y = None
        self.cal_drawing = False
        
        # Cargar configuración si existe
        self.load_params_from_file()
        
        # Crear estructura de la interfaz
        self.create_ui()
        
        # Iniciar cámara
        self.start_camera()

    def create_ui(self):
        # Marco principal dividido en dos paneles
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo (cámara)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Marco para la cámara
        self.camera_frame = ttk.LabelFrame(left_frame, text="Visor de Cámara")
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visor de cámara
        self.panel = ttk.Label(self.camera_frame)
        self.panel.pack(fill=tk.BOTH, expand=True)
        
        # Enlazar eventos del mouse para dibujar ROI
        self.panel.bind("<ButtonPress-1>", self.on_mouse_down)
        self.panel.bind("<B1-Motion>", self.on_mouse_move)
        self.panel.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Marco para la imagen binarizada
        self.binary_frame_container = ttk.LabelFrame(left_frame, text="Imagen Binarizada")
        self.binary_frame_container.pack(fill=tk.X, padx=5, pady=5)
        
        # Visor de imagen binarizada
        self.binary_panel = ttk.Label(self.binary_frame_container)
        self.binary_panel.pack(fill=tk.BOTH, expand=True)
        
        # Panel derecho (controles)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Marco para información
        info_frame = ttk.LabelFrame(right_frame, text="Información")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(info_frame, textvariable=self.calibration_label).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, textvariable=self.roi_label).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, textvariable=self.length_range_label).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Separator(info_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(info_frame, textvariable=self.total_objects).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, textvariable=self.within_range).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, textvariable=self.outside_range).pack(anchor=tk.W, padx=5, pady=2)
        
        # Marco para ajustes
        settings_frame = ttk.LabelFrame(right_frame, text="Ajustes")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Umbral:").pack(anchor=tk.W, padx=5, pady=2)
        threshold_scale = ttk.Scale(settings_frame, from_=0, to=255, variable=self.threshold_value, 
                                  orient=tk.HORIZONTAL, length=200, command=lambda _: self.update_threshold())
        threshold_scale.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Área mínima:").pack(anchor=tk.W, padx=5, pady=2)
        area_scale = ttk.Scale(settings_frame, from_=0, to=500, variable=self.min_area, 
                              orient=tk.HORIZONTAL, length=200, command=lambda _: self.update_threshold())
        area_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Marco para configuración de rango
        range_frame = ttk.LabelFrame(right_frame, text="Rango de Longitud")
        range_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Variables para el rango mínimo y máximo
        self.min_length_var = tk.IntVar(value=MIN_LENGTH_MM)
        self.max_length_var = tk.IntVar(value=MAX_LENGTH_MM)
        
        ttk.Label(range_frame, text="Mínimo (mm):").pack(anchor=tk.W, padx=5, pady=2)
        min_scale = ttk.Scale(range_frame, from_=0, to=100, variable=self.min_length_var,
                             orient=tk.HORIZONTAL, length=200, command=self.update_range)
        min_scale.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(range_frame, text="Máximo (mm):").pack(anchor=tk.W, padx=5, pady=2)
        max_scale = ttk.Scale(range_frame, from_=0, to=100, variable=self.max_length_var,
                             orient=tk.HORIZONTAL, length=200, command=self.update_range)
        max_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Marco para botones
        buttons_frame = ttk.Frame(right_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Botones principales
        calibrate_btn = ttk.Button(buttons_frame, text="Calibrar", command=self.calibrate_camera)
        calibrate_btn.pack(fill=tk.X, pady=5)
        
        select_roi_btn = ttk.Button(buttons_frame, text="Seleccionar Área", command=self.select_roi)
        select_roi_btn.pack(fill=tk.X, pady=5)
        
        reset_roi_btn = ttk.Button(buttons_frame, text="Restablecer Área", command=self.reset_roi)
        reset_roi_btn.pack(fill=tk.X, pady=5)
        
        save_btn = ttk.Button(buttons_frame, text="Guardar Configuración", command=self.save_params_to_file)
        save_btn.pack(fill=tk.X, pady=5)
        
        load_btn = ttk.Button(buttons_frame, text="Cargar Configuración", command=self.load_params_from_file)
        load_btn.pack(fill=tk.X, pady=5)
        
        # Barra de estado
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(WEBCAM_ID)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "No se pudo abrir la webcam. Verifique la conexión y el ID de la cámara.")
                return
            
            self.thread_running = True
            self.thread = threading.Thread(target=self.video_loop)
            self.thread.daemon = True
            self.thread.start()
            self.status_var.set("Cámara iniciada")
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar la cámara: {str(e)}")
    
    def video_loop(self):
        try:
            while self.thread_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error al capturar el frame.")
                    time.sleep(0.1)
                    continue
                
                # Reducir tamaño para mejorar rendimiento (opcional)
                # frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                
                self.current_frame = frame.copy()
                
                # Crear una copia del frame para dibujar sobre él
                display_frame = frame.copy()
                
                # Dibujar el ROI si está definido
                if ROI:
                    x1, y1, x2, y2 = ROI
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Solo procesar el ROI si es válido
                    if x1 < x2 and y1 < y2 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                        # Extraer la región de interés
                        roi_frame = frame[y1:y2, x1:x2]
                        
                        # Procesar imagen solo si ROI es válido
                        if roi_frame.size > 0:
                            # Procesar la imagen ROI
                            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                            _, binary = cv2.threshold(gray, self.threshold_value.get(), 255, cv2.THRESH_BINARY_INV)
                            
                            # Operaciones morfológicas
                            kernel = np.ones((3, 3), np.uint8)
                            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
                            
                            self.binary_frame = binary.copy()
                            
                            # Encontrar contornos
                            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            # Filtrar contornos pequeños
                            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area.get()]
                            
                            # Contadores
                            within_range_count = 0
                            outside_range_count = 0
                            
                            # Analizar contornos
                            for cnt in filtered_contours:
                                # Obtener rectángulo rotado mínimo
                                rect = cv2.minAreaRect(cnt)
                                box = cv2.boxPoints(rect)
                                box = np.array(box, dtype=np.int32)
                                
                                # Obtener dimensiones
                                width, height = rect[1]
                                
                                # Usar el lado más largo como longitud
                                length_px = max(width, height)
                                
                                # Convertir a milímetros si está calibrado
                                if CALIBRATION_MM_PER_PIXEL:
                                    length_mm = length_px * CALIBRATION_MM_PER_PIXEL
                                    unit = "mm"
                                    
                                    # Verificar si está en el rango
                                    within_range = MIN_LENGTH_MM <= length_mm <= MAX_LENGTH_MM
                                    contour_color = (0, 255, 0) if within_range else (0, 0, 255)
                                    
                                    if within_range:
                                        within_range_count += 1
                                    else:
                                        outside_range_count += 1
                                else:
                                    length_mm = length_px
                                    unit = "px"
                                    contour_color = (255, 0, 0)
                                
                                # Dibujar el rectángulo
                                cv2.drawContours(roi_frame, [box], 0, contour_color, 2)
                                
                                # Mostrar longitud
                                M = cv2.moments(cnt)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    text_color = (255, 255, 255)
                                    cv2.putText(roi_frame, f"{length_mm:.1f} {unit}", (cx - 30, cy), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                            
                            # Actualizar contadores en la interfaz de manera más eficiente
                            counters = (len(filtered_contours), within_range_count, outside_range_count)
                            self.root.after_idle(lambda c=counters: self.update_counters(*c))
                
                # Dibujar línea de calibración si está en modo calibración
                if self.calibration_mode and self.cal_start_x is not None and self.cal_end_x is not None:
                    cv2.line(display_frame, (self.cal_start_x, self.cal_start_y), 
                            (self.cal_end_x, self.cal_end_y), (0, 255, 0), 2)
                    # Calcular y mostrar la longitud en píxeles
                    pixel_length = np.sqrt((self.cal_end_x - self.cal_start_x)**2 + 
                                        (self.cal_end_y - self.cal_start_y)**2)
                    cv2.putText(display_frame, f"Longitud: {pixel_length:.1f} px", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Mostrar información de calibración
                if CALIBRATION_MM_PER_PIXEL:
                    cv2.putText(display_frame, f"Calibración: {CALIBRATION_MM_PER_PIXEL:.4f} mm/px", 
                                (display_frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "Sin calibrar", (display_frame.shape[1] - 300, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Mostrar rectángulo siendo dibujado (para selección de ROI)
                if self.drawing and self.start_x is not None and self.end_x is not None:
                    cv2.rectangle(display_frame, (self.start_x, self.start_y), 
                                (self.end_x, self.end_y), (0, 255, 255), 2)
                
                # Convertir para mostrar en Tkinter
                cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Actualizar panel de cámara
                self.root.after_idle(lambda i=imgtk: self.update_camera_panel(i))
                
                # Actualizar panel de imagen binarizada si hay ROI
                if self.binary_frame is not None:
                    bin_img = Image.fromarray(self.binary_frame)
                    bin_imgtk = ImageTk.PhotoImage(image=bin_img)
                    self.root.after_idle(lambda i=bin_imgtk: self.update_binary_panel(i))
                
                # Reducir tasa de refresco para no sobrecargar el sistema
                time.sleep(0.05)  # ~20 FPS en lugar de 30 FPS
        
        except Exception as e:
            print(f"Error en video_loop: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def update_counters(self, total, within, outside):
        self.total_objects.set(f"Total objetos: {total}")
        self.within_range.set(f"Dentro de rango: {within}")
        self.outside_range.set(f"Fuera de rango: {outside}")
    
    def update_threshold(self):
        pass  # El cambio de threshold se aplica automáticamente en el ciclo de video
    
    def update_range(self, event=None):
        global MIN_LENGTH_MM, MAX_LENGTH_MM
        
        # Asegurar que mínimo <= máximo
        if self.min_length_var.get() > self.max_length_var.get():
            self.min_length_var.set(self.max_length_var.get())
        
        # Actualizar variables globales
        MIN_LENGTH_MM = self.min_length_var.get()
        MAX_LENGTH_MM = self.max_length_var.get()
        
        # Actualizar etiqueta de rango
        self.length_range_label.set(f"Rango: {MIN_LENGTH_MM}-{MAX_LENGTH_MM} mm")
    
    def on_mouse_down(self, event):
        if self.calibration_mode:
            self.cal_start_x = event.x
            self.cal_start_y = event.y
            self.cal_end_x = event.x
            self.cal_end_y = event.y
            self.cal_drawing = True
        else:
            self.start_x = event.x
            self.start_y = event.y
            self.end_x = event.x
            self.end_y = event.y
            self.drawing = True
    
    def on_mouse_move(self, event):
        if self.calibration_mode and self.cal_drawing:
            self.cal_end_x = event.x
            self.cal_end_y = event.y
        elif self.drawing:
            self.end_x = event.x
            self.end_y = event.y
    
    def on_mouse_up(self, event):
        if self.calibration_mode and self.cal_drawing:
            self.cal_end_x = event.x
            self.cal_end_y = event.y
            self.cal_drawing = False
            
            # Si estamos calibrando, calculamos la distancia en píxeles
            pixel_length = np.sqrt((self.cal_end_x - self.cal_start_x)**2 + 
                                  (self.cal_end_y - self.cal_start_y)**2)
            
            # Pedimos al usuario la longitud real
            self.get_calibration_length(pixel_length)
        
        elif self.drawing:
            self.end_x = event.x
            self.end_y = event.y
            self.drawing = False
            
            # Si estamos seleccionando ROI, actualizamos
            if not self.calibration_mode:
                # Convertir coordenadas de la imagen
                x1 = min(self.start_x, self.end_x)
                y1 = min(self.start_y, self.end_y)
                x2 = max(self.start_x, self.end_x)
                y2 = max(self.start_y, self.end_y)
                
                global ROI
                ROI = (x1, y1, x2, y2)
                self.roi_label.set(f"Área: {x1},{y1} a {x2},{y2}")
                self.status_var.set("Área de medición seleccionada")
    
    def get_calibration_length(self, pixel_length):
        # Ventana para ingresar la longitud real
        cal_dialog = tk.Toplevel(self.root)
        cal_dialog.title("Calibración")
        cal_dialog.geometry("300x150")
        cal_dialog.transient(self.root)
        cal_dialog.grab_set()
        
        ttk.Label(cal_dialog, text=f"Longitud en píxeles: {pixel_length:.1f}").pack(pady=5)
        ttk.Label(cal_dialog, text="Ingrese la longitud real en milímetros:").pack(pady=5)
        
        length_var = tk.DoubleVar(value=10.0)
        
        # Slider para ingresar valor
        scale = ttk.Scale(cal_dialog, from_=1, to=100, variable=length_var,
                        orient=tk.HORIZONTAL, length=200)
        scale.pack(pady=5)
        
        value_label = ttk.Label(cal_dialog, text="10.0 mm")
        value_label.pack(pady=5)
        
        # Actualizar label del slider
        def update_value(event):
            value_label.config(text=f"{length_var.get():.1f} mm")
        
        scale.bind("<Motion>", update_value)
        
        def confirm():
            global CALIBRATION_MM_PER_PIXEL
            real_length_mm = length_var.get()
            CALIBRATION_MM_PER_PIXEL = real_length_mm / pixel_length
            self.calibration_label.set(f"Calibración: {CALIBRATION_MM_PER_PIXEL:.4f} mm/pixel")
            self.status_var.set(f"Calibración completada: {CALIBRATION_MM_PER_PIXEL:.4f} mm/pixel")
            self.calibration_mode = False
            cal_dialog.destroy()
        
        ttk.Button(cal_dialog, text="Confirmar", command=confirm).pack(pady=10)
    
    def calibrate_camera(self):
        self.calibration_mode = True
        self.status_var.set("Modo calibración: Dibuje una línea sobre un objeto de longitud conocida")
        
        # Mostrar instrucciones
        messagebox.showinfo("Calibración", 
                           "1. Coloque un objeto de longitud conocida (por ejemplo, una regla) frente a la cámara.\n"
                           "2. Dibuje una línea sobre el objeto presionando y arrastrando el mouse.\n"
                           "3. Ingrese la longitud real en milímetros cuando se le solicite.")
    
    def select_roi(self):
        self.calibration_mode = False
        self.status_var.set("Seleccione el área de medición dibujando un rectángulo")
        
        # Mostrar instrucciones
        messagebox.showinfo("Selección de Área", 
                           "Dibuje un rectángulo en la imagen para definir el área de medición.")
    


    def update_camera_panel(self, imgtk):
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)
        
    def update_binary_panel(self, imgtk):
        self.binary_panel.imgtk = imgtk
        self.binary_panel.config(image=imgtk)



    def reset_roi(self):
        global ROI
        ROI = None
        self.roi_label.set("Sin área seleccionada")
        self.status_var.set("Área de medición restablecida")
        self.binary_frame = None  # Limpiar imagen binarizada
    
    def save_params_to_file(self):
        global CALIBRATION_MM_PER_PIXEL, MIN_LENGTH_MM, MAX_LENGTH_MM, ROI
        
        try:
            with open("measurement_params.txt", "w") as f:
                f.write(f"CALIBRATION_MM_PER_PIXEL={CALIBRATION_MM_PER_PIXEL}\n")
                f.write(f"MIN_LENGTH_MM={MIN_LENGTH_MM}\n")
                f.write(f"MAX_LENGTH_MM={MAX_LENGTH_MM}\n")
                if ROI:
                    f.write(f"ROI={ROI[0]},{ROI[1]},{ROI[2]},{ROI[3]}\n")
            
            self.status_var.set("Configuración guardada en 'measurement_params.txt'")
            messagebox.showinfo("Guardar Configuración", "Configuración guardada con éxito.")
        except Exception as e:
            self.status_var.set(f"Error al guardar configuración: {str(e)}")
            messagebox.showerror("Error", f"Error al guardar configuración: {str(e)}")
    
    def load_params_from_file(self):
        global CALIBRATION_MM_PER_PIXEL, MIN_LENGTH_MM, MAX_LENGTH_MM, ROI
        
        try:
            if not os.path.exists("measurement_params.txt"):
                self.status_var.set("No se encontró archivo de configuración")
                return
            
            with open("measurement_params.txt", "r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        if key == "CALIBRATION_MM_PER_PIXEL" and value != "None":
                            CALIBRATION_MM_PER_PIXEL = float(value)
                            self.calibration_label.set(f"Calibración: {CALIBRATION_MM_PER_PIXEL:.4f} mm/pixel")
                        elif key == "MIN_LENGTH_MM":
                            MIN_LENGTH_MM = int(float(value))
                            self.min_length_var.set(MIN_LENGTH_MM)
                        elif key == "MAX_LENGTH_MM":
                            MAX_LENGTH_MM = int(float(value))
                            self.max_length_var.set(MAX_LENGTH_MM)
                        elif key == "ROI" and value != "None":
                            coords = value.split(",")
                            if len(coords) == 4:
                                ROI = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
                                self.roi_label.set(f"Área: {ROI[0]},{ROI[1]} a {ROI[2]},{ROI[3]}")
            
            # Actualizar etiqueta de rango
            self.length_range_label.set(f"Rango: {MIN_LENGTH_MM}-{MAX_LENGTH_MM} mm")
            self.status_var.set("Configuración cargada desde 'measurement_params.txt'")
            messagebox.showinfo("Cargar Configuración", "Configuración cargada con éxito.")
        except Exception as e:
            self.status_var.set(f"Error al cargar configuración: {str(e)}")
            messagebox.showerror("Error", f"Error al cargar configuración: {str(e)}")
    
    def on_closing(self):
        self.thread_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.root.destroy()

# Fuera de la clase MeasurementApp
if __name__ == "__main__":
    root = tk.Tk()
    app = MeasurementApp(root)
    root.mainloop()