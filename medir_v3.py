import cv2
import numpy as np
import time

# Configuración inicial
WEBCAM_ID = 1  # ID de la webcam (normalmente 0 para la webcam por defecto)
CALIBRATION_MM_PER_PIXEL = None  # Valor de calibración (se establecerá durante la ejecución)
ROI = None  # Región de interés (Region Of Interest)
MIN_LENGTH_MM = 0  # Longitud mínima del rango aceptable (en mm)
MAX_LENGTH_MM = 100  # Longitud máxima del rango aceptable (en mm)

def calibrate_camera():
    """
    Función para calibrar la cámara usando un objeto de referencia de tamaño conocido.
    Retorna el factor de conversión mm/pixel.
    """
    global CALIBRATION_MM_PER_PIXEL
    
    cap = cv2.VideoCapture(WEBCAM_ID)
    
    # Verificar que la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return None
    
    print("=== MODO DE CALIBRACIÓN ===")
    print("1. Coloque un objeto de longitud conocida (por ejemplo, una regla) frente a la cámara.")
    print("2. Dibuje una línea sobre el objeto presionando y arrastrando el mouse.")
    print("3. Ingrese la longitud real en milímetros cuando se le solicite.")
    
    # Variables para el dibujo de la línea
    start_point = None
    end_point = None
    drawing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal start_point, end_point, drawing
        
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            drawing = False
    
    cv2.namedWindow("Calibración")
    cv2.setMouseCallback("Calibración", mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break
        
        # Dibujar la línea si se están seleccionando puntos
        if start_point and end_point:
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            # Calcular y mostrar la longitud en píxeles
            pixel_length = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            cv2.putText(frame, f"Longitud: {pixel_length:.1f} px", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Calibración", frame)
        
        key = cv2.waitKey(1) & 0xFF
        # Presionar Enter para confirmar la línea dibujada
        if key == 13 and start_point and end_point:  # Enter key
            pixel_length = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            break
        # Presionar ESC para cancelar
        elif key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    # Crear una ventana para ingresar la longitud real
    input_value = ""
    def get_input(event, x, y, flags, param):
        nonlocal input_value
        if event == cv2.EVENT_KEYDOWN:
            if ord('0') <= event <= ord('9') or event == ord('.'):
                input_value += chr(event)
            elif event == 8:  # Backspace
                input_value = input_value[:-1]
    
    input_window = np.zeros((200, 400, 3), np.uint8)
    cv2.namedWindow("Ingrese la longitud real (mm)")
    cv2.putText(input_window, "Ingrese la longitud real en mm", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(input_window, "y presione Enter para confirmar", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Crear un trackbar para introducir el valor (de 1 a 100 mm)
    def update_value(x):
        nonlocal input_value
        input_value = str(x)
    
    cv2.createTrackbar("mm", "Ingrese la longitud real (mm)", 10, 100, update_value)
    input_value = "10"  # Valor inicial del trackbar
    
    while True:
        # Crear una copia para actualizar
        display_window = input_window.copy()
        cv2.putText(display_window, f"Valor: {input_value} mm", (150, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Ingrese la longitud real (mm)", display_window)
        key = cv2.waitKey(1) & 0xFF
        
        # Actualizar el valor desde el trackbar
        input_value = str(cv2.getTrackbarPos("mm", "Ingrese la longitud real (mm)"))
        
        # Procesar teclas numéricas y punto
        if key >= ord('0') and key <= ord('9'):
            input_value += chr(key)
        elif key == ord('.') and '.' not in input_value:
            input_value += '.'
        elif key == 8:  # Backspace
            input_value = input_value[:-1]
        elif key == 13:  # Enter
            break
    
    cv2.destroyWindow("Ingrese la longitud real (mm)")
    
    try:
        real_length_mm = float(input_value)
        # Calcular el factor de conversión
        CALIBRATION_MM_PER_PIXEL = real_length_mm / pixel_length
        print(f"Calibración completada: {CALIBRATION_MM_PER_PIXEL:.4f} mm/pixel")
    except ValueError:
        print("Valor inválido. Usando valor predeterminado de 10 mm.")
        real_length_mm = 10.0
        CALIBRATION_MM_PER_PIXEL = real_length_mm / pixel_length
    
    cap.release()
    cv2.destroyAllWindows()
    return CALIBRATION_MM_PER_PIXEL

def select_roi():
    """
    Función para seleccionar la región de interés (ROI) 
    donde se medirán los objetos.
    """
    global ROI
    
    cap = cv2.VideoCapture(WEBCAM_ID)
    
    # Verificar que la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return None
    
    print("=== SELECCIÓN DE ÁREA DE MEDICIÓN ===")
    print("1. Dibuje un rectángulo en la pantalla presionando y arrastrando el mouse.")
    print("2. Presione Enter para confirmar o ESC para cancelar.")
    
    # Variables para el dibujo del rectángulo
    start_point = None
    end_point = None
    drawing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal start_point, end_point, drawing
        
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            drawing = False
    
    cv2.namedWindow("Selección de Área")
    cv2.setMouseCallback("Selección de Área", mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break
        
        # Dibujar el rectángulo si se están seleccionando puntos
        if start_point and end_point:
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
        
        cv2.imshow("Selección de Área", frame)
        
        key = cv2.waitKey(1) & 0xFF
        # Presionar Enter para confirmar el área
        if key == 13 and start_point and end_point:  # Enter key
            x1 = min(start_point[0], end_point[0])
            y1 = min(start_point[1], end_point[1])
            x2 = max(start_point[0], end_point[0])
            y2 = max(start_point[1], end_point[1])
            ROI = (x1, y1, x2, y2)
            break
        # Presionar ESC para cancelar
        elif key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    print(f"Área seleccionada: {ROI}")
    cap.release()
    cv2.destroyAllWindows()
    return ROI

def set_length_range():
    """
    Función para configurar el rango de longitudes aceptables.
    """
    global MIN_LENGTH_MM, MAX_LENGTH_MM
    
    print("=== CONFIGURACIÓN DE RANGO DE LONGITUD ===")
    
    # Crear ventana para el rango
    range_window = np.zeros((250, 500, 3), np.uint8)
    cv2.namedWindow("Configuración de Rango")
    
    # Crear trackbars para mínimo y máximo
    def update_min(x):
        global MIN_LENGTH_MM
        MIN_LENGTH_MM = x
    
    def update_max(x):
        global MAX_LENGTH_MM
        MAX_LENGTH_MM = x
    
    # Crear trackbars con valores iniciales
    cv2.createTrackbar("Mínimo (mm)", "Configuración de Rango", MIN_LENGTH_MM, 100, update_min)
    cv2.createTrackbar("Máximo (mm)", "Configuración de Rango", MAX_LENGTH_MM, 100, update_max)
    
    while True:
        # Actualizar valores desde los trackbars
        MIN_LENGTH_MM = cv2.getTrackbarPos("Mínimo (mm)", "Configuración de Rango")
        MAX_LENGTH_MM = cv2.getTrackbarPos("Máximo (mm)", "Configuración de Rango")
        
        # Asegurar que mínimo <= máximo
        if MIN_LENGTH_MM > MAX_LENGTH_MM:
            MIN_LENGTH_MM = MAX_LENGTH_MM
            cv2.setTrackbarPos("Mínimo (mm)", "Configuración de Rango", MIN_LENGTH_MM)
        
        # Crear una copia para actualizar
        display_window = range_window.copy()
        
        # Mostrar valores actuales
        cv2.putText(display_window, "Configuración del Rango de Medición", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_window, f"Longitud mínima: {MIN_LENGTH_MM} mm", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_window, f"Longitud máxima: {MAX_LENGTH_MM} mm", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_window, "Objetos dentro de este rango se mostrarán en verde", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_window, "Objetos fuera de este rango se mostrarán en rojo", (10, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display_window, "Presione Enter para confirmar o ESC para cancelar", (10, 220), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Configuración de Rango", display_window)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter para confirmar
            break
        elif key == 27:  # ESC para cancelar sin cambios
            return
    
    cv2.destroyWindow("Configuración de Rango")
    print(f"Rango configurado: {MIN_LENGTH_MM} - {MAX_LENGTH_MM} mm")

def measure_objects():
    """
    Función principal para medir objetos dentro del ROI seleccionado
    """
    global CALIBRATION_MM_PER_PIXEL, ROI, MIN_LENGTH_MM, MAX_LENGTH_MM
    
    # Abrir la webcam
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return
    
    print("=== MIDIENDO OBJETOS ===")
    print("Presione 'ESC' para salir.")
    print("Presione 'c' para recalibrar.")
    print("Presione 'r' para seleccionar una nueva área de medición.")
    print("Presione 'l' para configurar el rango de longitud.")
    
    # Umbral para binarización
    threshold_value = 128
    
    # Crear ventana con trackbar para ajustar umbral
    cv2.namedWindow("Medición de Objetos")
    cv2.createTrackbar("Umbral", "Medición de Objetos", threshold_value, 255, lambda x: None)
    
    # Parámetros para filtrado de objetos
    min_area = 50  # Área mínima en píxeles cuadrados
    cv2.createTrackbar("Área mínima", "Medición de Objetos", min_area, 500, lambda x: None)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break
        
        # Dibujar el ROI si está definido
        if ROI:
            x1, y1, x2, y2 = ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extraer la región de interés
            roi_frame = frame[y1:y2, x1:x2]
            
            # Leer el valor actual de los trackbars
            threshold_value = cv2.getTrackbarPos("Umbral", "Medición de Objetos")
            min_area = cv2.getTrackbarPos("Área mínima", "Medición de Objetos")
            
            # Convertir a escala de grises y aplicar umbral
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
            
            # Operaciones morfológicas para mejorar la segmentación
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos muy pequeños (ruido)
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            # Contadores para objetos dentro y fuera de rango
            within_range_count = 0
            outside_range_count = 0
            
            # Dibujar contornos y medir objetos
            for i, cnt in enumerate(filtered_contours):
                # Obtener rectángulo rotado mínimo
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                
                # Obtener dimensiones (ancho y alto)
                width, height = rect[1]
                
                # Usar el lado más largo como longitud
                length_px = max(width, height)
                
                # Convertir a milímetros si está calibrado
                if CALIBRATION_MM_PER_PIXEL:
                    length_mm = length_px * CALIBRATION_MM_PER_PIXEL
                    unit = "mm"
                    
                    # Determinar si está dentro del rango aceptable
                    within_range = MIN_LENGTH_MM <= length_mm <= MAX_LENGTH_MM
                    
                    # Color del contorno según el rango
                    contour_color = (0, 255, 0) if within_range else (0, 0, 255)  # Verde si está dentro, rojo si está fuera
                    
                    # Actualizar contadores
                    if within_range:
                        within_range_count += 1
                    else:
                        outside_range_count += 1
                else:
                    length_mm = length_px
                    unit = "px"
                    contour_color = (255, 0, 0)  # Azul si no está calibrado
                
                # Dibujar el rectángulo rotado con el color correspondiente
                cv2.drawContours(roi_frame, [box], 0, contour_color, 2)
                
                # Calcular el centro del contorno para colocar el texto
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Mostrar la longitud
                    text_color = (255, 255, 255)  # Texto siempre blanco para legibilidad
                    cv2.putText(roi_frame, f"{length_mm:.1f} {unit}", (cx - 30, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            # Mostrar la cantidad de objetos detectados y su clasificación
            cv2.putText(frame, f"Total objetos: {len(filtered_contours)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if CALIBRATION_MM_PER_PIXEL:
                cv2.putText(frame, f"Dentro de rango: {within_range_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Fuera de rango: {outside_range_count}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Mostrar el rango configurado
                cv2.putText(frame, f"Rango: {MIN_LENGTH_MM}-{MAX_LENGTH_MM} mm", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Sin área seleccionada. Presione 'r' para seleccionar.", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar información de calibración
        if CALIBRATION_MM_PER_PIXEL:
            cv2.putText(frame, f"Calibración: {CALIBRATION_MM_PER_PIXEL:.4f} mm/px", (frame.shape[1] - 300, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Sin calibrar. Presione 'c' para calibrar.", (frame.shape[1] - 300, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar imagen con umbral si hay ROI seleccionado
        if ROI:
            cv2.imshow("Umbral", binary)
        
        # Mostrar el frame principal
        cv2.imshow("Medición de Objetos", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para salir
            break
        elif key == ord('c'):  # 'c' para calibrar
            cv2.destroyAllWindows()
            CALIBRATION_MM_PER_PIXEL = calibrate_camera()
            # Reiniciar la webcam después de la calibración
            cap.release()
            cap = cv2.VideoCapture(WEBCAM_ID)
            cv2.namedWindow("Medición de Objetos")
            cv2.createTrackbar("Umbral", "Medición de Objetos", threshold_value, 255, lambda x: None)
            cv2.createTrackbar("Área mínima", "Medición de Objetos", min_area, 500, lambda x: None)
        elif key == ord('r'):  # 'r' para seleccionar nueva ROI
            cv2.destroyAllWindows()
            ROI = select_roi()
            # Reiniciar la webcam después de seleccionar ROI
            cap.release()
            cap = cv2.VideoCapture(WEBCAM_ID)
            cv2.namedWindow("Medición de Objetos")
            cv2.createTrackbar("Umbral", "Medición de Objetos", threshold_value, 255, lambda x: None)
            cv2.createTrackbar("Área mínima", "Medición de Objetos", min_area, 500, lambda x: None)
        elif key == ord('l'):  # 'l' para configurar rango de longitud
            # Guardar las ventanas actuales
            cv2.destroyWindow("Medición de Objetos")
            if ROI:
                cv2.destroyWindow("Umbral")
            
            set_length_range()
            
            # Recrear las ventanas
            cv2.namedWindow("Medición de Objetos")
            cv2.createTrackbar("Umbral", "Medición de Objetos", threshold_value, 255, lambda x: None)
            cv2.createTrackbar("Área mínima", "Medición de Objetos", min_area, 500, lambda x: None)
            if ROI:
                cv2.imshow("Umbral", binary)
    
    cap.release()
    cv2.destroyAllWindows()

def save_params_to_file():
    """
    Guarda los parámetros actuales en un archivo
    """
    global CALIBRATION_MM_PER_PIXEL, MIN_LENGTH_MM, MAX_LENGTH_MM
    
    try:
        with open("measurement_params.txt", "w") as f:
            f.write(f"CALIBRATION_MM_PER_PIXEL={CALIBRATION_MM_PER_PIXEL}\n")
            f.write(f"MIN_LENGTH_MM={MIN_LENGTH_MM}\n")
            f.write(f"MAX_LENGTH_MM={MAX_LENGTH_MM}\n")
        print("Parámetros guardados en 'measurement_params.txt'")
    except Exception as e:
        print(f"Error al guardar parámetros: {e}")

def load_params_from_file():
    """
    Carga los parámetros desde un archivo
    """
    global CALIBRATION_MM_PER_PIXEL, MIN_LENGTH_MM, MAX_LENGTH_MM
    
    try:
        with open("measurement_params.txt", "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    if key == "CALIBRATION_MM_PER_PIXEL" and value != "None":
                        CALIBRATION_MM_PER_PIXEL = float(value)
                    elif key == "MIN_LENGTH_MM":
                        MIN_LENGTH_MM = int(float(value))
                    elif key == "MAX_LENGTH_MM":
                        MAX_LENGTH_MM = int(float(value))
        print("Parámetros cargados desde 'measurement_params.txt'")
    except FileNotFoundError:
        print("Archivo de parámetros no encontrado. Se usarán valores predeterminados.")
    except Exception as e:
        print(f"Error al cargar parámetros: {e}")

if __name__ == "__main__":
    print("Iniciando sistema de medición de objetos")
    
    # Intentar cargar configuración desde archivo
    load_params_from_file()
    
    # Mostrar menú principal
    while True:
        print("\n=== SISTEMA DE MEDICIÓN DE OBJETOS PEQUEÑOS ===")
        print("1. Calibrar cámara")
        print("2. Seleccionar área de medición")
        print("3. Configurar rango de longitud")
        print("4. Iniciar medición")
        print("5. Guardar configuración")
        print("6. Cargar configuración")
        print("0. Salir")
        
        option = input("Seleccione una opción: ")
        
        if option == "1":
            CALIBRATION_MM_PER_PIXEL = calibrate_camera()
        elif option == "2":
            ROI = select_roi()
        elif option == "3":
            set_length_range()
        elif option == "4":
            if CALIBRATION_MM_PER_PIXEL is None:
                print("ADVERTENCIA: No se ha calibrado la cámara. Las mediciones se mostrarán en píxeles.")
            if ROI is None:
                print("ADVERTENCIA: No se ha seleccionado un área de medición.")
                if input("¿Desea seleccionar un área ahora? (s/n): ").lower() == "s":
                    ROI = select_roi()
            measure_objects()
        elif option == "5":
            save_params_to_file()
        elif option == "6":
            load_params_from_file()
        elif option == "0":
            print("Saliendo del sistema...")
            break
        else:
            print("Opción no válida. Intente de nuevo.")