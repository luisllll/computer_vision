# src/yolo_detector.py - Módulo de detección YOLO
import numpy as np

class YOLODetector:
    """
    Módulo para detección de objetos usando YOLO
    Versión corregida para Ultralytics 8.2.0+ que elimina warnings
    """
    def __init__(self):
        self.model = None
        self.model_name = None
        self.model_loaded = False
        self.confidence_threshold = 0.3
        
        # Configuración específica para almendras - MÁS PERMISIVA
        self.detection_config = {
            'verbose': False,
            'save': False,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'show': False,
            'stream_buffer': False,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,  # Detectar todas las clases
            'retina_masks': False,
            'show_boxes': False,  # CORREGIDO: evitar warning 'boxes' deprecated
            'show_labels': False,
            'show_conf': False
        }
        
        # Filtros más permisivos para almendras
        self.filters = {
            'min_area': 5,         # Reducido de 100
            'max_area': 1000000,      # Aumentado de 50000
            'min_aspect_ratio': 0.8,  # Más permisivo
            'max_aspect_ratio': 6.0,  # Más permisivo
            'min_width': 5,
            'min_height': 5
        }
        
        # Debug mode
        self.debug_mode = True
    
    def load_model(self, model_name):
        """
        Cargar modelo YOLO especificado
        
        Args:
            model_name (str): Nombre del modelo (ej: 'yolov8s.pt')
        
        Returns:
            bool: True si se cargó exitosamente
        """
        try:
            print(f"📥 Cargando modelo {model_name}...")
            
            # Importar ultralytics
            from ultralytics import YOLO
            
            # Verificar si el modelo ya está descargado
            self._check_model_availability(model_name)
            
            # Cargar modelo
            self.model = YOLO(model_name)
            self.model_name = model_name
            self.model_loaded = True
            
            # Configurar modelo
            self._configure_model()
            
            print(f"✅ Modelo {model_name} cargado exitosamente")
            
            # Mostrar información del modelo
            if hasattr(self.model, 'names'):
                print(f"   📊 Clases disponibles: {len(self.model.names)}")
                class_names = list(self.model.names.values())
                if len(class_names) > 5:
                    print(f"   🏷️  Clases: {class_names[:3]}... (+{len(class_names)-3} más)")
                else:
                    print(f"   🏷️  Clases: {class_names}")
            
            return True
            
        except ImportError:
            print("❌ Error: ultralytics no está instalado")
            print("   💡 Instala con: pip install ultralytics")
            return False
            
        except Exception as e:
            print(f"❌ Error cargando modelo {model_name}: {e}")
            self.model_loaded = False
            return False
    
    def _check_model_availability(self, model_name):
        """Verificar disponibilidad del modelo"""
        try:
            from pathlib import Path
            
            # Verificar en caché local
            weights_dir = Path.home() / '.ultralytics' / 'weights'
            model_path = weights_dir / model_name
            
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024*1024)
                print(f"   ✅ Modelo encontrado localmente ({size_mb:.1f} MB)")
            else:
                print(f"   📥 Descargando modelo {model_name} (primera vez)...")
                print("   ⏳ Esto puede tomar unos minutos dependiendo de tu conexión...")
                
        except Exception as e:
            if self.debug_mode:
                print(f"   ⚠️ No se pudo verificar caché: {e}")
    
    def _configure_model(self):
        """Configurar parámetros del modelo"""
        if not self.model_loaded:
            return
        
        try:
            # Configurar umbral de confianza
            self.model.overrides['conf'] = self.confidence_threshold
            
            # Configurar para detección silenciosa
            self.model.overrides.update(self.detection_config)
            
            print(f"   🔧 Configurado - Confianza: {self.confidence_threshold}")
            
        except Exception as e:
            if self.debug_mode:
                print(f"   ⚠️ Error en configuración: {e}")
    
    def set_confidence_threshold(self, threshold):
        """
        Establecer umbral de confianza para detecciones
        
        Args:
            threshold (float): Umbral entre 0.0 y 1.0
        """
        old_threshold = self.confidence_threshold
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        
        if self.model_loaded:
            try:
                self.model.overrides['conf'] = self.confidence_threshold
                if self.debug_mode and abs(old_threshold - self.confidence_threshold) > 0.05:
                    print(f"🎯 Umbral actualizado: {old_threshold:.2f} → {self.confidence_threshold:.2f}")
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Error actualizando confianza: {e}")
    
    def detect_objects(self, image):
        """
        Detectar objetos en una imagen (MÉTODO PRINCIPAL CORREGIDO)
        
        Args:
            image (numpy.ndarray): Imagen de entrada en formato BGR
        
        Returns:
            list: Lista de detecciones con bbox, confianza y área
        """
        if not self.model_loaded:
            if self.debug_mode:
                print("⚠️ Modelo no cargado")
            return []
        
        try:
            # Ejecutar detección con configuración corregida
            results = self.model(
                image, 
                conf=self.confidence_threshold,
                verbose=False,          # Sin salida verbose
                show=False,            # No mostrar ventanas
                save=False,            # No guardar resultados
                stream=False           # No usar streaming
            )
            
            detections = []
            
            # Procesar resultados con método corregido
            for result in results:
                # Verificar si hay detecciones (método compatible)
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    # Verificar que hay datos válidos
                    if len(boxes) > 0:
                        detections.extend(self._extract_detections(boxes, image.shape))
            
            # Aplicar filtros personalizados
            filtered_detections = self._apply_filters(detections)
            
            # Debug información (solo si hay cambios significativos)
            if self.debug_mode and len(detections) > 0:
                if len(filtered_detections) != len(detections):
                    print(f"🔍 Detecciones: {len(detections)} → {len(filtered_detections)} (filtradas)")
                    
                    # Mostrar razones de filtrado si están muy filtradas
                    if len(filtered_detections) == 0 and len(detections) > 0:
                        self._debug_filtered_detections(detections)
            
            return filtered_detections
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Error en detección: {e}")
            return []
    
    def _extract_detections(self, boxes, image_shape):
        """
        Extraer detecciones de los boxes de YOLO (método seguro)
        """
        detections = []
        
        try:
            # Extraer coordenadas de manera compatible
            if hasattr(boxes, 'xyxy'):
                # Convertir a numpy si es tensor
                if hasattr(boxes.xyxy, 'cpu'):
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                else:
                    xyxy = np.array(boxes.xyxy)
                    conf = np.array(boxes.conf)
                
                # Extraer clase si está disponible
                if hasattr(boxes, 'cls') and boxes.cls is not None:
                    if hasattr(boxes.cls, 'cpu'):
                        cls = boxes.cls.cpu().numpy()
                    else:
                        cls = np.array(boxes.cls)
                else:
                    cls = np.zeros(len(xyxy))  # Clase por defecto
                
                # Procesar cada detección
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    confidence = float(conf[i])
                    class_id = int(cls[i]) if i < len(cls) else 0
                    
                    # Validar coordenadas básicas
                    if self._validate_coordinates(x1, y1, x2, y2, image_shape):
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'area': (x2 - x1) * (y2 - y1),
                            'class_id': class_id,
                            'class_name': self._get_class_name(class_id)
                        }
                        detections.append(detection)
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Error extrayendo detecciones: {e}")
        
        return detections
    
    def _validate_coordinates(self, x1, y1, x2, y2, image_shape):
        """
        Validar que las coordenadas del bounding box sean válidas
        """
        h, w = image_shape[:2]
        
        # Verificar que las coordenadas estén dentro de la imagen
        if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
            return False
        
        # Verificar que el bbox tenga área mínima
        if (x2 - x1) < 3 or (y2 - y1) < 3:
            return False
        
        # Verificar orden correcto de coordenadas
        if x2 <= x1 or y2 <= y1:
            return False
        
        return True
    
    def _get_class_name(self, class_id):
        """Obtener nombre de la clase"""
        if self.model_loaded and hasattr(self.model, 'names'):
            return self.model.names.get(class_id, f"class_{class_id}")
        return f"object_{class_id}"
    
    def _apply_filters(self, detections):
        """
        Aplicar filtros personalizados para almendras
        """
        filtered = []
        
        for detection in detections:
            if self._passes_all_filters(detection):
                filtered.append(detection)
        
        return filtered
    
    def _passes_all_filters(self, detection):
        """
        Verificar si una detección pasa todos los filtros
        """
        # Filtro por confianza (ya aplicado por YOLO, pero verificamos)
        if detection['confidence'] < self.confidence_threshold:
            return False
        
        # Extraer dimensiones
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = detection['area']
        
        # Filtro por área
        if area < self.filters['min_area'] or area > self.filters['max_area']:
            return False
        
        # Filtro por dimensiones mínimas
        if width < self.filters['min_width'] or height < self.filters['min_height']:
            return False
        
        # Filtro por aspect ratio
        if min(width, height) > 0:  # Evitar división por cero
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio < self.filters['min_aspect_ratio'] or aspect_ratio > self.filters['max_aspect_ratio']:
                return False
        else:
            return False  # Dimensiones inválidas
        
        return True
    
    def _debug_filtered_detections(self, detections):
        """
        Debug: mostrar por qué se filtran las detecciones
        """
        print(f"\n🔍 DEBUG: Analizando {len(detections)} detecciones filtradas:")
        
        for i, detection in enumerate(detections[:3]):  # Solo mostrar primeras 3
            conf = detection['confidence']
            area = detection['area']
            bbox = detection['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            if min(width, height) > 0:
                aspect_ratio = max(width, height) / min(width, height)
            else:
                aspect_ratio = 999
            
            print(f"  Det {i+1}: conf={conf:.2f}, área={area:.0f}, ratio={aspect_ratio:.1f}")
            
            # Verificar cada filtro
            reasons = []
            if conf < self.confidence_threshold:
                reasons.append(f"confianza<{self.confidence_threshold}")
            if area < self.filters['min_area']:
                reasons.append(f"área<{self.filters['min_area']}")
            if area > self.filters['max_area']:
                reasons.append(f"área>{self.filters['max_area']}")
            if aspect_ratio < self.filters['min_aspect_ratio'] or aspect_ratio > self.filters['max_aspect_ratio']:
                reasons.append(f"ratio fuera de {self.filters['min_aspect_ratio']}-{self.filters['max_aspect_ratio']}")
            if width < self.filters['min_width'] or height < self.filters['min_height']:
                reasons.append("dimensiones muy pequeñas")
            
            if reasons:
                print(f"    ❌ Filtrada: {', '.join(reasons)}")
            else:
                print(f"    ✅ Debería ser válida")
        
        print("💡 Sugerencias:")
        print("   - Reducir umbral de confianza si conf muy alta")
        print("   - Ajustar filtros de área si áreas válidas")
        print("   - Revisar aspect ratio si objetos no alargados")
    
    def get_model_info(self):
        """
        Obtener información del modelo cargado
        
        Returns:
            dict: Información del modelo
        """
        if not self.model_loaded:
            return {'loaded': False}
        
        info = {
            'loaded': True,
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'filters': self.filters.copy()
        }
        
        # Añadir información del modelo si está disponible
        if hasattr(self.model, 'names'):
            info.update({
                'num_classes': len(self.model.names),
                'class_names': list(self.model.names.values())
            })
        
        return info
    
    def set_filters(self, **kwargs):
        """
        Configurar filtros de detección
        
        Args:
            **kwargs: Parámetros de filtros
        """
        for key, value in kwargs.items():
            if key in self.filters:
                old_value = self.filters[key]
                self.filters[key] = value
                if self.debug_mode:
                    print(f"🔧 Filtro {key}: {old_value} → {value}")
        
        if self.debug_mode and kwargs:
            print("⚙️ Filtros actualizados")
    
    def set_debug_mode(self, enabled):
        """Activar/desactivar modo debug"""
        self.debug_mode = enabled
        print(f"🐛 Debug mode: {'ON' if enabled else 'OFF'}")
    
    def benchmark_model(self, test_image, iterations=10):
        """
        Hacer benchmark del modelo con una imagen de test
        
        Args:
            test_image (numpy.ndarray): Imagen para test
            iterations (int): Número de iteraciones para el benchmark
        
        Returns:
            dict: Resultados del benchmark
        """
        if not self.model_loaded:
            return {'error': 'Modelo no cargado'}
        
        import time
        
        print(f"📊 Iniciando benchmark con {iterations} iteraciones...")
        
        # Calentar modelo (3 iteraciones)
        for _ in range(3):
            self.detect_objects(test_image)
        
        # Medir rendimiento
        times = []
        detection_counts = []
        
        for i in range(iterations):
            start_time = time.time()
            detections = self.detect_objects(test_image)
            end_time = time.time()
            
            times.append(end_time - start_time)
            detection_counts.append(len(detections))
        
        # Calcular estadísticas
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        avg_detections = np.mean(detection_counts)
        
        results = {
            'model_name': self.model_name,
            'iterations': iterations,
            'avg_inference_time_ms': avg_time * 1000,
            'fps': fps,
            'avg_detections': avg_detections,
            'confidence_threshold': self.confidence_threshold,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000
        }
        
        print(f"📈 Benchmark {self.model_name}:")
        print(f"   ⏱️ Tiempo promedio: {avg_time*1000:.1f}ms")
        print(f"   🎬 FPS promedio: {fps:.1f}")
        print(f"   🎯 Detecciones promedio: {avg_detections:.1f}")
        print(f"   📊 Rango: {min(times)*1000:.1f}-{max(times)*1000:.1f}ms")
        
        return results
    
    def reset_filters_to_permissive(self):
        """Restablecer filtros a valores muy permisivos para debug"""
        self.filters = {
            'min_area': 25,
            'max_area': 50000,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 10.0,
            'min_width': 3,
            'min_height': 3
        }
        print("🔄 Filtros restablecidos a valores permisivos")
    
    def reset_filters_to_strict(self):
        """Restablecer filtros a valores estrictos para almendras"""
        self.filters = {
            'min_area': 200,
            'max_area': 5000,
            'min_aspect_ratio': 1.2,
            'max_aspect_ratio': 3.0,
            'min_width': 10,
            'min_height': 15
        }
        print("🔄 Filtros restablecidos a valores estrictos")


# Función de utilidad para testing
def test_yolo_detector():
    """Función de test para el detector YOLO"""
    import cv2
    
    print("🧪 Iniciando test completo del detector YOLO...")
    
    # Crear detector
    detector = YOLODetector()
    detector.set_debug_mode(True)
    
    # Test de carga de modelo
    print("\n1️⃣ Test de carga de modelo...")
    success = detector.load_model('yolov8s.pt')
    if not success:
        print("❌ No se pudo cargar el modelo")
        return
    
    # Mostrar información del modelo
    print("\n2️⃣ Información del modelo...")
    info = detector.get_model_info()
    print(f"📋 Info: {info}")
    
    # Crear imagen de test
    print("\n3️⃣ Test de detección...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test de detección
    detections = detector.detect_objects(test_image)
    print(f"✅ Test de detección completado - Detecciones: {len(detections)}")
    
    # Test de benchmark
    print("\n4️⃣ Test de rendimiento...")
    benchmark = detector.benchmark_model(test_image, iterations=5)
    
    print(f"\n✅ Test completo exitoso")
    print(f"📊 Resumen: {benchmark['fps']:.1f} FPS, {benchmark['avg_detections']:.1f} detecciones promedio")


if __name__ == "__main__":
    test_yolo_detector()