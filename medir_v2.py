import cv2
import numpy as np
import time

# Configuración inicial
WEBCAM_ID = 1  # ID de la webcam (normalmente 0 para la webcam por defecto)
CALIBRATION_MM_PER_PIXEL = None  # Valor de calibración (se establecerá durante la ejecución)
ROI = None  # Región de interés (Region Of Interest)

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

def measure_objects():
    """
    Función principal para medir objetos dentro del ROI seleccionado
    """
    global CALIBRATION_MM_PER_PIXEL, ROI
    
    # Abrir la webcam
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return
    
    print("=== MIDIENDO OBJETOS ===")
    print("Presione 'ESC' para salir.")
    print("Presione 'c' para recalibrar.")
    print("Presione 'r' para seleccionar una nueva área de medición.")
    
    # Umbral para binarización
    threshold_value = 128
    
    # Crear ventana con trackbar para ajustar umbral
    cv2.namedWindow("Medición de Objetos")
    cv2.createTrackbar("Umbral", "Medición de Objetos", threshold_value, 255, lambda x: None)
    
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
            
            # Leer el valor actual del trackbar
            threshold_value = cv2.getTrackbarPos("Umbral", "Medición de Objetos")
            
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
            min_area = 50  # Ajustar según el tamaño de los objetos
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            # Dibujar contornos y medir objetos
            for i, cnt in enumerate(filtered_contours):
                # Obtener rectángulo rotado mínimo
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                
                # Dibujar el rectángulo rotado
                cv2.drawContours(roi_frame, [box], 0, (0, 0, 255), 2)
                
                # Obtener dimensiones (ancho y alto)
                width, height = rect[1]
                
                # Usar el lado más largo como longitud
                length_px = max(width, height)
                
                # Convertir a milímetros si está calibrado
                if CALIBRATION_MM_PER_PIXEL:
                    length_mm = length_px * CALIBRATION_MM_PER_PIXEL
                    unit = "mm"
                else:
                    length_mm = length_px
                    unit = "px"
                
                # Calcular el centro del contorno para colocar el texto
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Mostrar la longitud
                    cv2.putText(roi_frame, f"{length_mm:.1f} {unit}", (cx - 30, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Mostrar la cantidad de objetos detectados
            cv2.putText(frame, f"Objetos: {len(filtered_contours)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Sin área seleccionada. Presione 'r' para seleccionar.", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar información de calibración
        if CALIBRATION_MM_PER_PIXEL:
            cv2.putText(frame, f"Calibración: {CALIBRATION_MM_PER_PIXEL:.4f} mm/px", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Sin calibrar. Presione 'c' para calibrar.", (10, 30), 
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
        elif key == ord('r'):  # 'r' para seleccionar nueva ROI
            cv2.destroyAllWindows()
            ROI = select_roi()
            # Reiniciar la webcam después de seleccionar ROI
            cap.release()
            cap = cv2.VideoCapture(WEBCAM_ID)
            cv2.namedWindow("Medición de Objetos")
            cv2.createTrackbar("Umbral", "Medición de Objetos", threshold_value, 255, lambda x: None)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Iniciando sistema de medición de objetos")
    print("1. Primero se realizará la calibración")
    print("2. Luego se seleccionará el área de medición")
    print("3. Finalmente se iniciará la medición de objetos")
    
    # Realizar calibración inicial
    CALIBRATION_MM_PER_PIXEL = calibrate_camera()
    
    # Seleccionar área de medición
    ROI = select_roi()
    
    # Iniciar medición
    measure_objects()