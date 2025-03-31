Contador de Almendras
Este software permite contar automáticamente almendras (u objetos similares) a través de una webcam, con especial énfasis en la detección precisa incluso cuando las almendras están agrupadas o se tocan entre sí.
Características

Detección precisa: Cuenta almendras individuales aunque estén en contacto.
Optimizado para chroma: Funciona especialmente bien con almendras sobre fondo verde.
Interfaz simple: Fácil de usar con controles intuitivos.
Visualización clara: Muestra contornos numerados para verificación visual.
Estadísticas básicas: Seguimiento de conteo actual y máximo.

Requisitos

Python 3.6 o superior
OpenCV (cv2)
NumPy

Puedes instalar las dependencias con:
bashCopiarpip install opencv-python numpy
Cómo usar
1. Ejecución
Ejecuta el script principal:
bashCopiarpython contador_almendras.py
2. Configuración inicial

Coloca las almendras sobre un fondo verde (tipo chroma).
Al iniciar el programa, selecciona el área de interés haciendo clic y arrastrando.

3. Controles
TeclaFunciónqSalir del programarRedefinir área de conteowActivar/desactivar separación de objetos agrupadosmMostrar/ocultar máscara de chromacReiniciar contador
4. Consejos para mejores resultados

Iluminación: Asegúrate de tener buena iluminación uniforme.
Fondo: Utiliza un fondo verde mate (sin brillos).
Distribución: Separa ligeramente las almendras para mejor detección.
Activar separación: Si las almendras están muy juntas, asegúrate de que el modo de separación esté activado (tecla 'w').

Cómo funciona
El programa implementa un algoritmo de visión por computadora en varias etapas:

Preprocesamiento:

Detecta y elimina el fondo verde mediante filtrado HSV
Mejora el contraste local con ecualización de histograma adaptativa


Segmentación:

Utiliza umbralización adaptativa para identificar objetos
Aplica operaciones morfológicas para limpiar la imagen


Separación de objetos:

Implementa el algoritmo watershed para separar objetos que se tocan
Utiliza transformación de distancia para identificar centros de objetos


Filtrado:

Descarta objetos demasiado pequeños o grandes
Filtra por proporción de aspecto para identificar formas de almendras


Visualización y conteo:

Dibuja contornos y enumera cada objeto detectado
Muestra estadísticas de conteo


----------------------------------------------------


Funcionamiento en profundidad del Contador de Almendras
Voy a explicarte en detalle cómo funciona el sistema de conteo de almendras y los hiperparámetros clave que afectan su rendimiento:
1. Detección del fondo chroma (filtrado HSV)
El sistema utiliza el espacio de color HSV (Hue, Saturation, Value) porque separa el tono del color (H) de su intensidad (V), lo que hace más robusto el sistema ante cambios de iluminación.
pythonCopiarhsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_objects = cv2.bitwise_not(mask_green)
Hiperparámetros clave:

lower_green y upper_green: Estos arrays definen el rango de color verde en el espacio HSV que se considera como "chroma".

lower_green = np.array([35, 50, 50]): El límite inferior (verde amarillento con saturación y brillo moderados)
upper_green = np.array([85, 255, 255]): El límite superior (verde azulado con máxima saturación y brillo)

Si el fondo verde no se detecta correctamente, estos valores deberían ajustarse. Un valor H (primer elemento) entre 35-85 cubre la mayoría de tonos verdes.

2. Limpieza de la máscara (operaciones morfológicas)
Para eliminar ruido y pequeños huecos en la máscara:
pythonCopiarkernel = np.ones((3,3), np.uint8)
mask_clean = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, kernel, iterations=1)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
Hiperparámetros:

kernel: Una matriz 3x3 de unos que define el "área de influencia" de las operaciones morfológicas.

Un kernel más grande (5x5) eliminaría más ruido pero podría fusionar objetos cercanos.
Un kernel más pequeño preservaría más detalles pero retendría más ruido.


iterations: Número de veces que se aplica la operación.

Valores mayores intensifican el efecto pero pueden distorsionar la forma de las almendras.



3. Mejora de contraste (CLAHE)
La ecualización de histograma adaptativa con limitación de contraste (CLAHE) mejora la visibilidad de bordes:
pythonCopiarclahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray)
Hiperparámetros:

clipLimit=2.0: Limita la amplificación del contraste para evitar ruido excesivo.

Valores mayores dan más contraste pero pueden amplificar ruido.
Valores menores dan menos contraste pero son más suaves.


tileGridSize=(8,8): Divide la imagen en 8x8 regiones para ecualización local.

Mallas más finas (16x16) se adaptan mejor a variaciones locales.
Mallas más gruesas (4x4) dan un resultado más uniforme.



4. Binarización adaptativa
Convierte la imagen a blanco y negro de forma que se adapta a diferentes regiones:
pythonCopiarbinary = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)
Hiperparámetros:

cv2.ADAPTIVE_THRESH_GAUSSIAN_C: Usa una ponderación gaussiana para los píxeles vecinos.
11: Tamaño de la vecindad para calcular el umbral (debe ser impar).

Valores mayores son más robustos a variaciones pero pueden perder detalles.
Valores menores preservan detalles pero son más susceptibles al ruido.


2: Constante que se resta del valor calculado (ajusta la sensibilidad).

Valores mayores generan menos píxeles blancos (objetos más pequeños).
Valores menores generan más píxeles blancos (objetos más grandes).



5. Separación de objetos agrupados (Watershed)
El algoritmo Watershed es clave para separar almendras que se tocan:
pythonCopiar# Transformación de distancia
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

# Umbral para marcadores
_, markers = cv2.threshold(dist, 0.3, 1.0, cv2.THRESH_BINARY)
Hiperparámetros cruciales:

cv2.DIST_L2: Utiliza distancia euclidiana (la más natural para formas orgánicas).
5: Tamaño de máscara para la transformación de distancia.

Valores mayores dan transformaciones más suaves pero menos precisas.
Valores menores son más precisos pero más sensibles al ruido.


0.3: Umbral de distancia normalizada.

Este es quizás el parámetro más importante para separar objetos agrupados
Valores más bajos (0.1-0.2) crean más marcadores, separando más agresivamente.
Valores más altos (0.4-0.5) generan menos marcadores, manteniendo unidos objetos que se tocan ligeramente.
El valor óptimo depende del tamaño y forma de las almendras.



6. Filtrado de contornos por área y forma
pythonCopiarif min_area < area < max_area:
    # Calcular proporción de aspecto para identificar almendras
    x, y, w, h = cv2.boundingRect(contorno)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Almendras típicamente tienen proporción entre 0.3 y 3.0
    if 0.3 < aspect_ratio < 3.0:
        contornos_filtrados.append(contorno)
Hiperparámetros:

min_area = 100: Área mínima en píxeles para considerar un objeto.

Depende de la distancia de la cámara a las almendras y resolución.
Objetos con menos píxeles que este valor se descartan como ruido.


max_area = 5000: Área máxima en píxeles.

Objetos más grandes que este valor se descartan (podrían ser grupos no separados).


aspect_ratio: Proporción ancho/alto.

El rango 0.3 < aspect_ratio < 3.0 permite identificar formas ovaladas en cualquier orientación.
Un rango más estricto como 0.5 < aspect_ratio < 2.0 sería más específico para almendras.



Influencia del chroma en el rendimiento
El fondo chroma verde:

Simplifica la segmentación inicial: Al poder aislar fácilmente los objetos del fondo, se reduce la complejidad del problema.
Aumenta la precisión de la transformación de distancia: Con objetos bien definidos contra el fondo, la transformación de distancia funciona mejor para encontrar centros de objetos.
Mejora la consistencia: Reduce la variabilidad introducida por fondos no uniformes o texturas que podrían confundirse con partes de almendras.

Posibles mejoras y ajustes

Calibración automática del rango de color: Detectar automáticamente el rango óptimo de verde en la primera ejecución.
Ajuste dinámico del umbral de watershed: Modificar el umbral (0.3) según la densidad de objetos en la imagen.
Aprendizaje automático: Incorporar un clasificador entrenado específicamente en reconocer formas de almendras.
Ajustes específicos por tamaño: Modificar parámetros como min_area y max_area automáticamente según la distancia focal y zoom.