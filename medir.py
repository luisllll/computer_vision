import cv2
import numpy as np
import time
from datetime import datetime
import os

def medidor_almendras_simple():
    # Inicializar webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Crear directorio para resultados
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    
    # Parámetros para procesamiento de imagen
    min_area = 100       # Área mínima para considerar un objeto
    max_area = 10000     # Área máxima para considerar un objeto
    usar_watershed = True
    
    # Parámetros para filtro de color verde (HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Factores de calibración (mm por píxel)
    mm_por_pixel = 0.2   # Valor inicial por defecto
    calibracion_realizada = False
    objeto_referencia_mm = 20  # Tamaño en mm del objeto de referencia (ej. moneda)
    
    # Variables para visualización
    mostrar_mascara = False
    tiempo_inicio = time.time()
    calibracion_modo = False
    referencia_p1 = None
    referencia_p2 = None
    
    # Región de interés (ROI) para medir
    roi_definida = False
    seleccionando_roi = False
    roi_inicio = (0, 0)
    roi_fin = (0, 0)
    roi = None
    
    print("\n== MEDIDOR DE DIMENSIONES DE ALMENDRAS ==")
    print("Seleccione el área donde se encuentran las almendras (clic y arrastre)")
    print("\nControles:")
    print("  'q' - Salir")
    print("  'r' - Redefinir área de medición")
    print("  'w' - Activar/desactivar separación de objetos agrupados")
    print("  'm' - Mostrar/ocultar máscara de chroma")
    print("  'c' - Entrar/salir del modo calibración")
    print("  's' - Guardar captura")
    print("======================\n")
    
    # Callback para selección de ROI y puntos de calibración
    def mouse_callback(event, x, y, flags, param):
        nonlocal seleccionando_roi, roi_inicio, roi_fin, roi_definida, roi
        nonlocal calibracion_modo, referencia_p1, referencia_p2, mm_por_pixel, calibracion_realizada
        
        if calibracion_modo:
            # Modo de calibración - seleccionar puntos de referencia
            if event == cv2.EVENT_LBUTTONDOWN:
                if referencia_p1 is None:
                    referencia_p1 = (x, y)
                elif referencia_p2 is None:
                    referencia_p2 = (x, y)
                    
                    # Calcular distancia en píxeles
                    dist_pixels = np.sqrt((referencia_p2[0] - referencia_p1[0])**2 + 
                                         (referencia_p2[1] - referencia_p1[1])**2)
                    
                    # Calcular factor de calibración (mm por pixel)
                    if dist_pixels > 0:
                        mm_por_pixel = objeto_referencia_mm / dist_pixels
                        calibracion_realizada = True
                        print(f"Calibración realizada: {mm_por_pixel:.4f} mm/pixel")
                        
                    # Resetear puntos para permitir recalibración
                    referencia_p1 = None
                    referencia_p2 = None
                    calibracion_modo = False
                    
        else:
            # Modo normal - seleccionar ROI
            if event == cv2.EVENT_LBUTTONDOWN:
                seleccionando_roi = True
                roi_inicio = (x, y)
            
            elif event == cv2.EVENT_MOUSEMOVE and seleccionando_roi:
                roi_fin = (x, y)
            
            elif event == cv2.EVENT_LBUTTONUP:
                seleccionando_roi = False
                roi_fin = (x, y)
                x1, y1 = min(roi_inicio[0], roi_fin[0]), min(roi_inicio[1], roi_fin[1])
                x2, y2 = max(roi_inicio[0], roi_fin[0]), max(roi_inicio[1], roi_fin[1])
                
                if x2 - x1 > 20 and y2 - y1 > 20:
                    roi = (x1, y1, x2, y2)
                    roi_definida = True
    
    # Configurar ventana y callback del mouse
    cv2.namedWindow('Medidor de Almendras')
    cv2.setMouseCallback('Medidor de Almendras', mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer de la webcam")
            break
        
        tiempo_frame = time.time()
        frame_original = frame.copy()
        
        # Si estamos en modo calibración
        if calibracion_modo:
            # Mensaje de instrucción
            cv2.putText(frame, "MODO CALIBRACION: Seleccione dos puntos para el objeto de referencia", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Si ya se seleccionó el primer punto, mostrarlo
            if referencia_p1 is not None:
                cv2.circle(frame, referencia_p1, 5, (0, 0, 255), -1)
                
                # Si se está seleccionando el segundo punto, mostrar línea
                if referencia_p2 is None:
                    # Mostrar posición actual del mouse como si fuera el segundo punto
                    mouse_pos = (flags & cv2.EVENT_FLAG_LBUTTON)
                    cv2.line(frame, referencia_p1, (x, y), (0, 255, 0), 2)
            
            cv2.putText(frame, f"Tamaño de referencia: {objeto_referencia_mm} mm", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if calibracion_realizada:
                cv2.putText(frame, f"Calibración: {mm_por_pixel:.4f} mm/pixel", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            # Mostrar el frame con información de calibración
            cv2.imshow('Medidor de Almendras', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):  # Salir del modo calibración
                calibracion_modo = False
                referencia_p1 = None
                referencia_p2 = None
            
            continue
        
        # Fase 1: Selección de ROI si no está definida
        if not roi_definida:
            cv2.putText(frame, "Seleccione área de medición (clic y arrastre)", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Dibujar ROI mientras se selecciona
            if seleccionando_roi:
                x1, y1 = roi_inicio
                x2, y2 = roi_fin
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.imshow('Medidor de Almendras', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):  # Entrar en modo calibración
                calibracion_modo = True
            
            continue
        
        # Fase 2: Procesamiento de imagen y medición
        x1, y1, x2, y2 = roi
        roi_img = frame[y1:y2, x1:x2].copy()
        
        # PREPROCESAMIENTO PARA CHROMA VERDE
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_objects = cv2.bitwise_not(mask_green)
        
        # Eliminar ruido de la máscara
        kernel = np.ones((3,3), np.uint8)
        mask_clean = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Visualización de máscara o procesamiento normal
        if mostrar_mascara:
            mask_display = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
            roi_procesada = mask_display
        else:
            roi_procesada = roi_img.copy()
            bg_black = np.zeros_like(roi_procesada)
            roi_procesada = cv2.bitwise_and(roi_img, roi_img, mask=mask_clean)
        
        # Mejora de contraste y binarización
        gray = cv2.cvtColor(roi_procesada, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eq = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Tratamiento para separar almendras agrupadas
        if usar_watershed:
            # Transformación de distancia
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            
            # Umbral adaptativo
            _, markers = cv2.threshold(dist, 0.3, 1.0, cv2.THRESH_BINARY)
            markers = np.uint8(markers * 255)
            
            # Encontrar contornos de los marcadores
            contornos_marcadores, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Crear marcadores para watershed
            markers_watershed = np.zeros(gray.shape, dtype=np.int32)
            
            # Dibujar los marcadores
            for i, contorno in enumerate(contornos_marcadores):
                cv2.drawContours(markers_watershed, [contorno], -1, i+1, -1)
            
            # Asegurarse de tener un fondo
            background = cv2.dilate(binary, kernel, iterations=3)
            bg_contornos, _ = cv2.findContours(background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(markers_watershed, bg_contornos, -1, 1, 3)
            
            # Aplicar watershed
            cv2.watershed(roi_procesada, markers_watershed)
            
            # Encontrar contornos de objetos separados
            contornos = []
            for i in range(2, np.max(markers_watershed) + 1):
                mask = np.zeros_like(gray)
                mask[markers_watershed == i] = 255
                obj_contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contornos.extend(obj_contornos)
        else:
            # Método simple: encontrar contornos directamente
            contornos, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área y forma
        contornos_filtrados = []
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if min_area < area < max_area:
                # Filtrar por forma, almendras son ovaladas
                x, y, w, h = cv2.boundingRect(contorno)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Proporción típica de almendras
                if 0.3 < aspect_ratio < 3.0:
                    contornos_filtrados.append(contorno)
        
        # Crear imagen para visualización
        roi_visualizacion = roi_img.copy()
        
        # Variables para calcular promedios
        total_largo = 0
        total_ancho = 0
        total_area = 0
        
        # Procesar cada contorno y calcular dimensiones
        for i, contorno in enumerate(contornos_filtrados):
            # Obtener momentos del contorno
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            
            # Calcular área en mm²
            area_pixels = cv2.contourArea(contorno)
            area_mm2 = area_pixels * (mm_por_pixel ** 2)
            
            # Encontrar eje principal (elipse que mejor se ajusta)
            if len(contorno) >= 5:  # Necesitamos al menos 5 puntos para ajustar una elipse
                ellipse = cv2.fitEllipse(contorno)
                center, axes, angle = ellipse
                
                # Longitud del eje mayor y menor en mm
                eje_mayor_mm = max(axes) * mm_por_pixel
                eje_menor_mm = min(axes) * mm_por_pixel
                
                # Dibujar elipse
                cv2.ellipse(roi_visualizacion, ellipse, (0, 255, 255), 2)
                
                # Dibujar ejes principales
                angle_rad = np.deg2rad(angle)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                
                # Eje mayor
                endpoint1 = (int(cx + cos_a * axes[0]/2), int(cy + sin_a * axes[0]/2))
                endpoint2 = (int(cx - cos_a * axes[0]/2), int(cy - sin_a * axes[0]/2))
                cv2.line(roi_visualizacion, endpoint1, endpoint2, (0, 0, 255), 2)
                
                # Eje menor
                endpoint3 = (int(cx - sin_a * axes[1]/2), int(cy + cos_a * axes[1]/2))
                endpoint4 = (int(cx + sin_a * axes[1]/2), int(cy - cos_a * axes[1]/2))
                cv2.line(roi_visualizacion, endpoint3, endpoint4, (255, 0, 0), 2)
            else:
                # Si no podemos ajustar una elipse, usar rectángulo rotado
                rect = cv2.minAreaRect(contorno)
                eje_mayor_mm = max(rect[1][0], rect[1][1]) * mm_por_pixel
                eje_menor_mm = min(rect[1][0], rect[1][1]) * mm_por_pixel
                
                # Dibujar rectángulo rotado
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(roi_visualizacion, [box], 0, (0, 255, 255), 2)
            
            # Acumular para promedios
            total_largo += eje_mayor_mm
            total_ancho += eje_menor_mm
            total_area += area_mm2
            
            # Dibujar contorno y centro
            cv2.drawContours(roi_visualizacion, [contorno], -1, (0, 255, 0), 2)
            cv2.circle(roi_visualizacion, (cx, cy), 5, (255, 0, 0), -1)
            
            # Mostrar mediciones en la imagen
            tx, ty = cx + 10, cy
            cv2.putText(roi_visualizacion, f"#{i+1}", (cx-10, cy-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(roi_visualizacion, f"{eje_mayor_mm:.1f}x{eje_menor_mm:.1f}mm", (tx, ty),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(roi_visualizacion, f"{eje_mayor_mm:.1f}x{eje_menor_mm:.1f}mm", (tx, ty),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            cv2.putText(roi_visualizacion, f"A: {area_mm2:.1f}mm²", (tx, ty+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(roi_visualizacion, f"A: {area_mm2:.1f}mm²", (tx, ty+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Insertar ROI procesada en el frame original
        frame[y1:y2, x1:x2] = roi_visualizacion
        
        # Dibujar rectángulo ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Mostrar cantidad de almendras detectadas
        num_almendras = len(contornos_filtrados)
        cv2.putText(frame, f"Almendras: {num_almendras}", (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, f"Almendras: {num_almendras}", (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Panel de información
        alto, ancho = frame.shape[:2]
        panel_alto = 120
        panel = np.zeros((panel_alto, ancho, 3), dtype=np.uint8)
        panel[:, :] = (45, 45, 45)  # Fondo gris oscuro
        
        # Línea divisoria
        cv2.line(panel, (0, 0), (ancho, 0), (200, 200, 200), 2)
        
        # Título y estado de calibración
        if calibracion_realizada:
            cv2.putText(panel, "MEDIDOR DE ALMENDRAS", (ancho//2 - 160, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(panel, f"Calibración: {mm_por_pixel:.4f} mm/pixel", 
                       (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(panel, "MEDIDOR DE ALMENDRAS (SIN CALIBRAR)", (ancho//2 - 220, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            cv2.putText(panel, "Pulse 'c' para calibrar", 
                       (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
        
        # Estado y modos
        modo_sep = "Separación: ACTIVADA" if usar_watershed else "Separación: DESACTIVADA"
        modo_vis = "Modo: MÁSCARA" if mostrar_mascara else "Modo: NORMAL"
        
        cv2.putText(panel, modo_sep, (ancho - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Si hay almendras detectadas, mostrar estadísticas
        if num_almendras > 0:
            # Calcular promedios
            largo_promedio = total_largo / num_almendras
            ancho_promedio = total_ancho / num_almendras
            area_promedio = total_area / num_almendras
            
            # Mostrar promedios
            cv2.putText(panel, f"Promedio - Largo: {largo_promedio:.2f} mm, Ancho: {ancho_promedio:.2f} mm, Área: {area_promedio:.2f} mm²",
                       (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Tiempo y FPS
        fps = 1.0 / max(0.001, time.time() - tiempo_frame)
        tiempo_actual = time.time() - tiempo_inicio
        mins = int(tiempo_actual // 60)
        segs = int(tiempo_actual % 60)
        
        cv2.putText(panel, f"Tiempo: {mins:02d}:{segs:02d} | FPS: {fps:.1f}", (ancho - 280, panel_alto - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Combinar frame y panel
        resultado = np.vstack([frame, panel])
        
        # Mostrar resultado
        cv2.imshow('Medidor de Almendras', resultado)
        
        # Procesar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            roi_definida = False
            print("Redefiniendo área de medición...")
        elif key == ord('w'):
            usar_watershed = not usar_watershed
            print(f"Separación de objetos agrupados: {'activada' if usar_watershed else 'desactivada'}")
        elif key == ord('m'):
            mostrar_mascara = not mostrar_mascara
            print(f"Mostrar máscara: {'activado' if mostrar_mascara else 'desactivado'}")
        elif key == ord('c'):
            calibracion_modo = True
            print("Modo calibración activado. Seleccione dos puntos en un objeto de referencia conocido.")
        elif key == ord('s'):
            # Guardar captura con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_img = f"resultados/almendras_{timestamp}.jpg"
            cv2.imwrite(filename_img, resultado)
            print(f"Captura guardada: {filename_img}")
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        medidor_almendras_simple()
    except Exception as e:
        print(f"Error: {e}")
        print("Comprueba que tienes OpenCV instalado: pip install opencv-python numpy")