import cv2
import numpy as np
import time
from datetime import datetime
import os

def contador_almendras_chroma():
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
    max_area = 5000      # Área máxima para considerar un objeto
    usar_watershed = True
    
    # Parámetros para filtro de color verde (HSV)
    # Valores predeterminados para un rango de verde típico de chroma
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Variables para estadísticas
    total_objetos = 0
    conteo_maximo = 0
    tiempo_inicio = time.time()
    
    # Controles de visualización
    mostrar_mascara = False
    
    # Región de interés (ROI) para contar
    roi_definida = False
    seleccionando_roi = False
    roi_inicio = (0, 0)
    roi_fin = (0, 0)
    roi = None
    
    print("\n== CONTADOR DE ALMENDRAS (CHROMA) ==")
    print("Seleccione el área donde se encuentran las almendras (clic y arrastre)")
    print("\nControles:")
    print("  'q' - Salir")
    print("  'r' - Redefinir área de conteo")
    print("  'w' - Activar/desactivar separación de objetos agrupados")
    print("  'm' - Mostrar/ocultar máscara de chroma")
    print("  'c' - Reiniciar contador")
    print("======================\n")
    
    # Callback para selección de ROI
    def mouse_callback(event, x, y, flags, param):
        nonlocal seleccionando_roi, roi_inicio, roi_fin, roi_definida, roi
        
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
    cv2.namedWindow('Contador de Almendras')
    cv2.setMouseCallback('Contador de Almendras', mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer de la webcam")
            break
        
        tiempo_frame = time.time()
        
        # Fase 1: Selección de ROI
        if not roi_definida:
            cv2.putText(frame, "Seleccione área de conteo (clic y arrastre)", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Dibujar ROI mientras se selecciona
            if seleccionando_roi:
                x1, y1 = roi_inicio
                x2, y2 = roi_fin
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.imshow('Contador de Almendras', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            continue
        
        # Fase 2: Procesamiento de imagen y conteo
        # Extraer región de interés
        x1, y1, x2, y2 = roi
        roi_img = frame[y1:y2, x1:x2].copy()
        
        # PREPROCESAMIENTO MEJORADO PARA CHROMA VERDE:
        
        # 1. Convertir a HSV para mejor separación de color
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        
        # 2. Crear máscara para el fondo verde
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # 3. Invertir la máscara para obtener los objetos (no el fondo)
        mask_objects = cv2.bitwise_not(mask_green)
        
        # 4. Eliminar ruido de la máscara
        kernel = np.ones((3,3), np.uint8)
        mask_clean = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 5. Aplicar la máscara a la imagen original si se desea visualizar
        if mostrar_mascara:
            # Convertir máscara a 3 canales para visualización
            mask_display = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
            roi_procesada = mask_display
        else:
            # Extraer solo los objetos (almendras) del fondo verde
            roi_procesada = roi_img.copy()
            # Crear fondo negro
            bg_black = np.zeros_like(roi_procesada)
            # Colocar objetos sobre fondo negro
            roi_procesada = cv2.bitwise_and(roi_img, roi_img, mask=mask_clean)
        
        # 6. Mejorar contraste para detección de bordes
        gray = cv2.cvtColor(roi_procesada, cv2.COLOR_BGR2GRAY)
        
        # 7. Aplicar ecualización de histograma adaptativa para mejorar contraste local
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eq = clahe.apply(gray)
        
        # 8. Usar umbral adaptativo para mejor binarización
        binary = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Tratamiento para separar almendras agrupadas
        if usar_watershed:
            # Transformación de distancia para separar objetos agrupados
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            
            # Umbral adaptativo basado en la forma típica de almendras
            # (Almendras tienden a ser ovaladas, así que usamos un umbral que preserve esa forma)
            _, markers = cv2.threshold(dist, 0.3, 1.0, cv2.THRESH_BINARY)
            markers = np.uint8(markers * 255)
            
            # Dilatar marcadores levemente para asegurar al menos un marcador por objeto
            markers = cv2.dilate(markers, kernel, iterations=1)
            
            # Crear marcadores para watershed
            contornos_marcadores, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            markers_watershed = np.zeros(gray.shape, dtype=np.int32)
            
            # Dibujar los marcadores
            for i, contorno in enumerate(contornos_marcadores):
                cv2.drawContours(markers_watershed, [contorno], -1, i+1, -1)
            
            # Asegurarse de tener un fondo
            background = cv2.dilate(binary, kernel, iterations=3)
            bg_contornos, _ = cv2.findContours(background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in bg_contornos:
                cv2.drawContours(markers_watershed, [c], -1, 1, 3)
            
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
                # Calcular proporción de aspecto para identificar formas similares a almendras
                x, y, w, h = cv2.boundingRect(contorno)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Almendras típicamente tienen proporción de aspecto entre 0.5 y 2.0
                if 0.3 < aspect_ratio < 3.0:
                    contornos_filtrados.append(contorno)
        
        # Actualizar conteo
        total_objetos = len(contornos_filtrados)
        conteo_maximo = max(conteo_maximo, total_objetos)
        
        # Crear imagen para visualización final
        roi_visualizacion = roi_img.copy()
        
        # Dibujar contornos y enumerar objetos
        for i, contorno in enumerate(contornos_filtrados):
            # Obtener centro aproximado
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            # Dibujar contorno y centro
            cv2.drawContours(roi_visualizacion, [contorno], -1, (0, 255, 0), 2)
            cv2.circle(roi_visualizacion, (cX, cY), 5, (255, 0, 0), -1)
            
            # Enumerar objeto
            cv2.putText(roi_visualizacion, str(i+1), (cX - 10, cY - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(roi_visualizacion, str(i+1), (cX - 10, cY - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Insertar ROI procesada en el frame original
        frame[y1:y2, x1:x2] = roi_visualizacion
        
        # Dibujar rectángulo ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Mostrar texto con conteo actual
        cv2.putText(frame, f"Almendras: {total_objetos}", (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, f"Almendras: {total_objetos}", (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Panel de información
        alto, ancho = frame.shape[:2]
        panel_alto = 100
        panel = np.zeros((panel_alto, ancho, 3), dtype=np.uint8)
        panel[:, :] = (45, 45, 45)  # Fondo gris oscuro
        
        # Línea divisoria
        cv2.line(panel, (0, 0), (ancho, 0), (200, 200, 200), 2)
        
        # Texto informativo
        cv2.putText(panel, "CONTADOR DE ALMENDRAS", (ancho//2 - 160, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Conteo actual y máximo
        cv2.putText(panel, f"Conteo actual: {total_objetos}", (30, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
        cv2.putText(panel, f"Conteo máximo: {conteo_maximo}", (ancho//2 + 30, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        # Modo de separación y visualización
        modo_sep = "Separación: ACTIVADA" if usar_watershed else "Separación: DESACTIVADA"
        modo_vis = "Modo: MÁSCARA" if mostrar_mascara else "Modo: NORMAL"
        
        cv2.putText(panel, modo_sep, (30, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(panel, modo_vis, (250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Tiempo y FPS
        fps = 1.0 / max(0.001, time.time() - tiempo_frame)
        tiempo_actual = time.time() - tiempo_inicio
        mins = int(tiempo_actual // 60)
        segs = int(tiempo_actual % 60)
        
        cv2.putText(panel, f"Tiempo: {mins:02d}:{segs:02d} | FPS: {fps:.1f}", (ancho - 280, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Combinar frame y panel
        resultado = np.vstack([frame, panel])
        
        # Mostrar resultado
        cv2.imshow('Contador de Almendras', resultado)
        
        # Procesar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            roi_definida = False
            print("Redefiniendo área de conteo...")
        elif key == ord('w'):
            usar_watershed = not usar_watershed
            print(f"Separación de objetos agrupados: {'activada' if usar_watershed else 'desactivada'}")
        elif key == ord('m'):
            mostrar_mascara = not mostrar_mascara
            print(f"Mostrar máscara: {'activado' if mostrar_mascara else 'desactivado'}")
        elif key == ord('c'):
            conteo_maximo = 0
            tiempo_inicio = time.time()
            print("Contador reiniciado")
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        contador_almendras_chroma()
    except Exception as e:
        print(f"Error: {e}")
        print("Comprueba que tienes OpenCV instalado: pip install opencv-python numpy")