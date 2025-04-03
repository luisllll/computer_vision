import cv2
import numpy as np
import time
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

# Configuración inicial
WEBCAM_ID = 1  # ID de la webcam (normalmente 0 para la webcam por defecto)
CALIBRATION_MM_PER_PIXEL = None  # Valor de calibración (se establecerá durante la ejecución)
ROI = None  # Región de interés (Region Of Interest)

# Inicialización de modelos
def initialize_sam():
    """
    Inicializa el modelo SAM para segmentación
    """
    print("Inicializando modelo SAM...")
    
    # Cargar modelo SAM desde Hugging Face (versión más ligera)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Usar SAM-base en lugar de SAM-huge (mucho más pequeño y rápido)
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    print(f"Modelo SAM cargado exitosamente. Usando dispositivo: {device}")
    
    return sam_model, sam_processor, device

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
            frame_copy = frame.copy()
            cv2.line(frame_copy, start_point, end_point, (0, 255, 0), 2)
            
            # Calcular y mostrar la longitud en píxeles
            pixel_length = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            cv2.putText(frame_copy, f"Longitud: {pixel_length:.1f} px", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Calibración", frame_copy)
        else:
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
    input_window = np.zeros((200, 400, 3), np.uint8)
    cv2.namedWindow("Ingrese la longitud real (mm)")
    cv2.putText(input_window, "Ingrese la longitud real en mm", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(input_window, "y presione Enter para confirmar", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Crear un trackbar para introducir el valor (de 1 a 300 mm)
    def update_value(x):
        nonlocal input_value
        input_value = str(x)
    
    cv2.createTrackbar("mm", "Ingrese la longitud real (mm)", 10, 300, update_value)
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
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("Selección de Área", frame_copy)
        else:
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

def export_measurements_to_csv(measurements, filename="mediciones.csv"):
    """
    Exporta las mediciones a un archivo CSV
    """
    import csv
    from datetime import datetime
    
    # Nombre de archivo con fecha y hora
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mediciones_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Encabezados
        headers = ['ID', 'Longitud (mm)', 'Área (mm²)', 'Timestamp']
        writer.writerow(headers)
        
        # Datos
        for i, m in enumerate(measurements):
            writer.writerow([
                i+1, 
                m.get('length', 'N/A'), 
                m.get('area', 'N/A'), 
                m.get('timestamp', 'N/A')
            ])
    
    print(f"Mediciones exportadas a {filename}")

def measure_objects_with_sam_direct(sam_model, sam_processor, device):
    """
    Función principal para medir objetos
    utilizando SAM directamente con clicks del usuario
    """
    global CALIBRATION_MM_PER_PIXEL, ROI
    
    # Abrir la webcam
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return
    
    print("=== MIDIENDO OBJETOS CON SAM DIRECTO ===")
    print("Presione 'ESC' para salir.")
    print("Presione 'c' para recalibrar.")
    print("Presione 'r' para seleccionar una nueva área de medición.")
    print("Presione 'e' para exportar mediciones a CSV.")
    print("Haga clic en los objetos para medirlos con SAM.")
    
    # Crear ventana principal
    cv2.namedWindow("Medición de Objetos")
    
    # Lista para almacenar todas las mediciones
    all_measurements_history = []
    
    # Variables para el manejo de clicks
    clicked_points = []
    click_to_process = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_points, click_to_process
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Ajustar coordenadas si hay ROI
            if ROI:
                x1, y1, x2, y2 = ROI
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Convertir coordenadas globales a coordenadas dentro del ROI
                    roi_x = x - x1
                    roi_y = y - y1
                    clicked_points.append((roi_x, roi_y))
                    click_to_process = True
            else:
                clicked_points.append((x, y))
                click_to_process = True
    
    cv2.setMouseCallback("Medición de Objetos", mouse_callback)
    
    # Objetos actuales segmentados
    current_segmentations = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break
        
        # Dibujar el ROI si está definido
        roi_frame = None
        if ROI:
            x1, y1, x2, y2 = ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extraer la región de interés
            roi_frame = frame[y1:y2, x1:x2].copy()
        else:
            roi_frame = frame.copy()
        
        # Procesar clicks si hay alguno
        if click_to_process and roi_frame is not None:
            # Solo procesar el último click
            last_click = clicked_points[-1]
            
            try:
                # Convertir a formato PIL para SAM
                roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                roi_pil = Image.fromarray(roi_rgb)
                
                # Preparar punto de entrada para SAM
                input_points = [[[last_click[0], last_click[1]]]]
                
                # Procesar la imagen con SAM
                inputs = sam_processor(
                    roi_pil, 
                    input_points=input_points, 
                    return_tensors="pt"
                ).to(device)
                
                # Generar segmentación
                with torch.no_grad():
                    outputs = sam_model(**inputs)
                
                # Post-procesar las máscaras
                masks = sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )
                
                # Obtener la mejor máscara
                best_mask_idx = torch.argmax(outputs.iou_scores)
                mask = masks[0][best_mask_idx].numpy()
                
                # Convertir máscara a formato OpenCV
                mask_cv = (mask * 255).astype(np.uint8)
                
                # Encontrar contornos de la máscara
                contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Calcular el rectángulo mínimo rotado
                    largest_contour = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(largest_contour)
                    box_points = cv2.boxPoints(rect)
                    box_points = np.array(box_points, dtype=np.int32)
                    
                    # Calcular dimensiones
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
                    
                    # Calcular área
                    area_px = cv2.contourArea(largest_contour)
                    area_mm2 = area_px * (CALIBRATION_MM_PER_PIXEL ** 2) if CALIBRATION_MM_PER_PIXEL else area_px
                    
                    # Crear una máscara coloreada
                    colored_mask = np.zeros_like(roi_frame)
                    colored_mask[mask] = [0, 255, 0]
                    
                    # Guardar la segmentación actual
                    current_segmentations.append({
                        'contour': largest_contour,
                        'box': box_points,
                        'mask': colored_mask,
                        'center': last_click,
                        'length_px': length_px,
                        'length_mm': length_mm,
                        'area_px': area_px,
                        'area_mm2': area_mm2,
                        'unit': unit
                    })
                    
                    # Añadir la medición al historial
                    all_measurements_history.append({
                        'length': length_mm,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'area': area_mm2
                    })
                    
                    print(f"Objeto medido: {length_mm:.2f} {unit}, Área: {area_mm2:.2f} {unit}²")
            
            except Exception as e:
                print(f"Error procesando segmentación: {e}")
            
            # Reset flag
            click_to_process = False
        
        # Dibujar todas las segmentaciones actuales
        if ROI and roi_frame is not None:
            x1, y1, x2, y2 = ROI
            
            # Crear copia del ROI frame para dibujar
            display_roi = roi_frame.copy()
            
            # Crear una máscara combinada
            combined_mask = np.zeros_like(roi_frame)
            
            for i, seg in enumerate(current_segmentations):
                # Dibujar contorno
                cv2.drawContours(display_roi, [seg['box']], 0, (0, 0, 255), 2)
                
                # Añadir a la máscara combinada
                combined_mask = cv2.add(combined_mask, seg['mask'])
                
                # Dibujar información
                center_x, center_y = seg['center']
                
                # Mostrar medidas
                cv2.putText(display_roi, f"{seg['length_mm']:.2f} {seg['unit']}", 
                            (center_x, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.putText(display_roi, f"Area: {seg['area_mm2']:.2f} {seg['unit']}²", 
                            (center_x, center_y + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Superponer la máscara combinada
            cv2.addWeighted(display_roi, 0.7, combined_mask, 0.3, 0, display_roi)
            
            # Reemplazar la región en el frame original
            frame[y1:y2, x1:x2] = display_roi
            
            # Mostrar visualización de segmentación
            if combined_mask.any():
                cv2.imshow("Segmentación", combined_mask)
        
        # Mostrar información de calibración
        if CALIBRATION_MM_PER_PIXEL:
            cv2.putText(frame, f"Calibración: {CALIBRATION_MM_PER_PIXEL:.4f} mm/px", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Sin calibrar. Presione 'c' para calibrar.", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar cantidad de objetos medidos
        cv2.putText(frame, f"Objetos: {len(current_segmentations)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Instrucciones
        cv2.putText(frame, "Haga clic en objetos para medir", (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "d: borrar última medición", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Mostrar el frame principal
        cv2.imshow("Medición de Objetos", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para salir
            # Preguntar si desea guardar las mediciones
            if all_measurements_history:
                save_window = np.zeros((200, 400, 3), np.uint8)
                cv2.putText(save_window, "¿Guardar mediciones? (s/n)", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Guardar", save_window)
                
                save_key = cv2.waitKey(0) & 0xFF
                if save_key == ord('s'):
                    export_measurements_to_csv(all_measurements_history)
            break
        elif key == ord('c'):  # 'c' para calibrar
            cv2.destroyAllWindows()
            CALIBRATION_MM_PER_PIXEL = calibrate_camera()
            # Reiniciar la webcam después de la calibración
            cap.release()
            cap = cv2.VideoCapture(WEBCAM_ID)
            cv2.namedWindow("Medición de Objetos")
            cv2.setMouseCallback("Medición de Objetos", mouse_callback)
            current_segmentations = []  # Reiniciar segmentaciones
        elif key == ord('r'):  # 'r' para seleccionar nueva ROI
            cv2.destroyAllWindows()
            ROI = select_roi()
            # Reiniciar la webcam después de seleccionar ROI
            cap.release()
            cap = cv2.VideoCapture(WEBCAM_ID)
            cv2.namedWindow("Medición de Objetos")
            cv2.setMouseCallback("Medición de Objetos", mouse_callback)
            current_segmentations = []  # Reiniciar segmentaciones
        elif key == ord('e'):  # 'e' para exportar mediciones
            if all_measurements_history:
                export_measurements_to_csv(all_measurements_history)
                print("Mediciones exportadas a CSV.")
        elif key == ord('d'):  # 'd' para eliminar última medición
            if current_segmentations:
                current_segmentations.pop()
                if all_measurements_history:
                    all_measurements_history.pop()
                print("Última medición eliminada")
        elif key == ord('x'):  # 'x' para limpiar todas las mediciones
            current_segmentations = []
            print("Todas las mediciones eliminadas de la pantalla")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("""
===========================================
     SISTEMA DE MEDICIÓN DIRECTO CON SAM
===========================================

Este sistema utiliza exclusivamente SAM (Segment Anything Model) 
para segmentar y medir objetos a partir de clicks del usuario.

Flujo de trabajo:
1. Calibre la cámara usando un objeto de tamaño conocido
2. Seleccione una región de interés (opcional pero recomendado)
3. Haga clic en los objetos que desea medir

Comandos durante la ejecución:
- ESC: Salir
- c: Calibrar cámara
- r: Seleccionar región de interés (ROI)
- e: Exportar mediciones a CSV
- d: Borrar última medición
- x: Borrar todas las mediciones de la pantalla

Requisitos:
pip install opencv-python numpy torch transformers pillow
    """)
    
    # Iniciar el sistema
    try:
        # Inicializar modelos
        print("Inicializando modelo SAM (esto puede tardar unos momentos)...")
        sam_model, sam_processor, device = initialize_sam()
        
        print("\n1. Primero se realizará la calibración")
        print("2. Luego se seleccionará el área de medición (opcional)")
        print("3. Finalmente podrá medir objetos haciendo clic sobre ellos")
        
        # Realizar calibración inicial
        CALIBRATION_MM_PER_PIXEL = calibrate_camera()
        
        # Seleccionar área de medición
        ROI = select_roi()
        
        # Iniciar medición con SAM directo
        measure_objects_with_sam_direct(sam_model, sam_processor, device)
    
    except Exception as e:
        print(f"Error: {e}")
        print("Revise que las bibliotecas necesarias estén instaladas:")
        print("pip install opencv-python numpy torch transformers pillow")
        print("\nAsegúrese de tener conexión a Internet para descargar el modelo SAM de Hugging Face:")
        print("- SAM: facebook/sam-vit-base (se descargará automáticamente de Hugging Face)")