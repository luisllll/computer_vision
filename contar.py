import cv2
import numpy as np
import time
from datetime import datetime
import os

def contar_almendras():
    # Inicializar webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Crear directorio para resultados
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    
    # Región de interés (ROI) para contar
    roi_definida = False
    seleccionando_roi = False
    roi_inicio = (0, 0)
    roi_fin = (0, 0)
    roi = None
    
    # Variables para estadísticas
    total_objetos = 0
    conteo_maximo = 0
    tiempo_inicio = time.time()
    
    # Parámetros optimizados para almendras
    min_area = 100
    max_area = 10000
    excentricidad_min = 0.5  # Para reconocer formas ovaladas
    
    print("\n== CONTADOR AVANZADO DE ALMENDRAS ==")
    print("Seleccione el área donde se encuentran las almendras (clic y arrastre)")
    print("\nControles:")
    print("  'q' - Salir")
    print("  'r' - Redefinir área de conteo")
    print("  's' - Guardar captura")
    print("  'c' - Reiniciar contador")
    print("===================================\n")
    
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
            
            if seleccionando_roi:
                x1, y1 = roi_inicio
                x2, y2 = roi_fin
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.imshow('Contador de Almendras', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            continue
        
        # Fase 2: Procesamiento y conteo
        x1, y1, x2, y2 = roi
        roi_img = frame[y1:y2, x1:x2].copy()
        
        # Preprocesamiento mejorado
        # 1. Normalización de iluminación
        lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        roi_normalizado = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 2. Conversión a escala de grises y desenfoque
        gray = cv2.cvtColor(roi_normalizado, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Binarización adaptativa (funciona mejor para condiciones variables)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 21, 5)
        
        # 4. Operaciones morfológicas para limpiar ruido
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Implementación avanzada de watershed para separar objetos agrupados
        # 1. Transformación de distancia (para encontrar centros)
        dist = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        
        # 2. Normalización para visualización uniforme
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        
        # 3. Umbralización para encontrar marcadores de objetos
        dist_thresh = 0.4  # Valor optimizado para almendras
        _, sure_fg = cv2.threshold(dist, dist_thresh, 1.0, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg * 255)
        
        # 4. Encontrar zonas desconocidas
        sure_bg = cv2.dilate(closing, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 5. Etiquetado de componentes
        _, markers = cv2.connectedComponents(sure_fg)
        
        # 6. Añadir 1 a todos para reservar 0 para watershed
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 7. Aplicar watershed
        markers = cv2.watershed(roi_normalizado, markers)
        
        # Preparar imagen de visualización
        roi_visualizacion = roi_img.copy()
        
        # Crear una máscara para los objetos encontrados
        objetos_mask = np.zeros_like(gray)
        objetos_mask[markers > 1] = 255
        
        # Encontrar contornos de objetos 
        contornos, _ = cv2.findContours(objetos_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos y analizar formas
        almendras_detectadas = []
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            
            # Filtrar por área
            if min_area <= area <= max_area:
                # Análisis de forma para identificar almendras
                # Las almendras son generalmente ovaladas
                
                # Encontrar el rectángulo rotado mínimo
                rect = cv2.minAreaRect(contorno)
                ancho, alto = rect[1]
                
                # Evitar división por cero
                if min(ancho, alto) > 0:
                    # Calcular relación de aspecto (largo/ancho)
                    aspect_ratio = max(ancho, alto) / min(ancho, alto)
                    
                    # Calcular excentricidad aproximada (0=círculo, 1=línea)
                    excentricidad = np.sqrt(1 - (min(ancho, alto) / max(ancho, alto))**2)
                    
                    # Las almendras típicamente tienen excentricidad entre 0.5 y 0.9
                    # y relación de aspecto entre 1.5 y 3.0
                    if excentricidad >= excentricidad_min and 1.5 <= aspect_ratio <= 4.0:
                        almendras_detectadas.append(contorno)
        
        # Actualizar conteo
        total_objetos = len(almendras_detectadas)
        conteo_maximo = max(conteo_maximo, total_objetos)
        
        # Dibujar y numerar almendras detectadas
        for i, contorno in enumerate(almendras_detectadas):
            # Calcular el centro
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            # Dibujar contorno con color
            cv2.drawContours(roi_visualizacion, [contorno], -1, (0, 255, 0), 2)
            
            # Dibujar círculo en el centro
            cv2.circle(roi_visualizacion, (cX, cY), 5, (255, 0, 0), -1)
            
            # Numerar cada almendra
            cv2.putText(roi_visualizacion, str(i+1), (cX - 10, cY - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(roi_visualizacion, str(i+1), (cX - 10, cY - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Colocar la ROI procesada en el frame
        frame[y1:y2, x1:x2] = roi_visualizacion
        
        # Dibujar rectángulo de la ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Mostrar conteo en la parte superior
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
        
        # Título
        cv2.putText(panel, "CONTADOR DE ALMENDRAS", (ancho//2 - 140, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Conteo actual y máximo
        cv2.putText(panel, f"Conteo actual: {total_objetos}", (30, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
        cv2.putText(panel, f"Conteo máximo: {conteo_maximo}", (ancho//2 + 30, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        # Tiempo y FPS
        tiempo_actual = time.time() - tiempo_inicio
        mins = int(tiempo_actual // 60)
        segs = int(tiempo_actual % 60)
        fps = 1.0 / max(0.001, time.time() - tiempo_frame)
        
        tiempo_texto = f"Tiempo: {mins:02d}:{segs:02d} | FPS: {fps:.1f}"
        cv2.putText(panel, tiempo_texto, (ancho - 280, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Combinar frame y panel
        resultado = np.vstack([frame, panel])
        
        # Mostrar resultado
        cv2.imshow('Contador de Almendras', resultado)
        
        # Procesar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Salir
            break
        elif key == ord('r'):  # Redefinir ROI
            roi_definida = False
            print("Redefiniendo área de conteo...")
        elif key == ord('s'):  # Guardar captura
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resultados/almendras_{timestamp}.jpg"
            cv2.imwrite(filename, resultado)
            print(f"Captura guardada como: {filename}")
        elif key == ord('c'):  # Reiniciar contador
            conteo_maximo = 0
            tiempo_inicio = time.time()
            print("Contador reiniciado")
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        contar_almendras()
    except Exception as e:
        print(f"Error: {e}")
        print("Comprueba que tienes instalado OpenCV: pip install opencv-python numpy")