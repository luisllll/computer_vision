# src/ruler_calibrator.py - Módulo de calibración con regla
import numpy as np
import cv2
import math
from datetime import datetime

class RulerCalibrator:
    """
    Módulo para calibración de cámara usando regla como referencia
    Convierte mediciones en píxeles a unidades reales (mm)
    """
    def __init__(self):
        self.mm_per_pixel = None
        self.calibrated = False
        self.calibration_data = {}
        
        # Configuración para detección automática de regla
        self.ruler_detection_config = {
            'min_line_length': 80,
            'max_line_gap': 5,
            'threshold': 100,
            'min_ruler_length_pixels': 150,
            'canny_low': 50,
            'canny_high': 150
        }
        
        # Historial de calibraciones
        self.calibration_history = []
    
    def calibrate_with_line(self, start_point, end_point, ruler_length_mm):
        """
        Calibrar usando una línea dibujada manualmente
        
        Args:
            start_point (tuple): Punto inicial (x, y)
            end_point (tuple): Punto final (x, y)
            ruler_length_mm (float): Longitud real de la línea en milímetros
        
        Returns:
            bool: True si la calibración fue exitosa
        """
        try:
            # Calcular longitud en píxeles
            pixel_length = self._calculate_pixel_distance(start_point, end_point)
            
            # Validar longitud mínima
            if pixel_length < 10:
                print("❌ Error: Línea muy corta para calibración precisa")
                return False
            
            # Validar longitud real
            if ruler_length_mm <= 0:
                print("❌ Error: Longitud real debe ser positiva")
                return False
            
            # Calcular escala
            self.mm_per_pixel = ruler_length_mm / pixel_length
            self.calibrated = True
            
            # Guardar datos de calibración
            self.calibration_data = {
                'start_point': start_point,
                'end_point': end_point,
                'pixel_length': pixel_length,
                'ruler_length_mm': ruler_length_mm,
                'mm_per_pixel': self.mm_per_pixel,
                'timestamp': datetime.now().isoformat(),
                'method': 'manual_line'
            }
            
            # Añadir al historial
            self.calibration_history.append(self.calibration_data.copy())
            
            # Log de éxito
            print(f"✅ Calibración manual exitosa:")
            print(f"   📏 Longitud píxeles: {pixel_length:.1f}")
            print(f"   📐 Longitud real: {ruler_length_mm} mm")
            print(f"   🔄 Escala: {self.mm_per_pixel:.4f} mm/píxel")
            print(f"   📊 Resolución: {1/self.mm_per_pixel:.2f} píxeles/mm")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en calibración manual: {e}")
            return False
    
    def auto_calibrate_with_ruler(self, image, ruler_length_mm):
        """
        Calibrar automáticamente detectando una regla en la imagen
        
        Args:
            image (numpy.ndarray): Imagen que contiene la regla
            ruler_length_mm (float): Longitud del segmento de regla conocido
        
        Returns:
            bool: True si la calibración fue exitosa
        """
        try:
            # Detectar líneas de regla
            ruler_line = self._detect_ruler_line(image)
            
            if ruler_line is None:
                print("❌ No se detectó ninguna regla en la imagen")
                return False
            
            start_point = (ruler_line[0], ruler_line[1])
            end_point = (ruler_line[2], ruler_line[3])
            
            # Usar calibración manual con la línea detectada
            success = self.calibrate_with_line(start_point, end_point, ruler_length_mm)
            
            if success:
                self.calibration_data['method'] = 'auto_detection'
                print("🤖 Calibración automática completada")
            
            return success
            
        except Exception as e:
            print(f"❌ Error en calibración automática: {e}")
            return False
    
    def _detect_ruler_line(self, image):
        """
        Detectar la línea más prominente en la imagen (probablemente una regla)
        
        Args:
            image (numpy.ndarray): Imagen de entrada
        
        Returns:
            tuple: (x1, y1, x2, y2) de la línea detectada o None
        """
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Aplicar filtros para mejorar detección
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar bordes
        edges = cv2.Canny(
            blurred,
            self.ruler_detection_config['canny_low'],
            self.ruler_detection_config['canny_high'],
            apertureSize=3
        )
        
        # Detectar líneas usando Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.ruler_detection_config['threshold'],
            minLineLength=self.ruler_detection_config['min_line_length'],
            maxLineGap=self.ruler_detection_config['max_line_gap']
        )
        
        if lines is None or len(lines) == 0:
            return None
        
        # Encontrar la línea más larga
        longest_line = None
        max_length = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = self._calculate_pixel_distance((x1, y1), (x2, y2))
            
            # Verificar que la línea sea suficientemente larga
            if length > max_length and length > self.ruler_detection_config['min_ruler_length_pixels']:
                max_length = length
                longest_line = line[0]
        
        return longest_line
    
    def _calculate_pixel_distance(self, point1, point2):
        """
        Calcular distancia euclidiana entre dos puntos
        
        Args:
            point1 (tuple): Primer punto (x, y)
            point2 (tuple): Segundo punto (x, y)
        
        Returns:
            float: Distancia en píxeles
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def set_calibration(self, mm_per_pixel):
        """
        Establecer calibración manualmente
        
        Args:
            mm_per_pixel (float): Escala milímetros por píxel
        """
        if mm_per_pixel is not None and mm_per_pixel > 0:
            self.mm_per_pixel = mm_per_pixel
            self.calibrated = True
            
            self.calibration_data = {
                'mm_per_pixel': mm_per_pixel,
                'timestamp': datetime.now().isoformat(),
                'method': 'manual_input'
            }
            
            print(f"🔧 Calibración establecida manualmente: {mm_per_pixel:.4f} mm/píxel")
        else:
            self.mm_per_pixel = None
            self.calibrated = False
            print("❌ Calibración eliminada")
    
    def is_calibrated(self):
        """
        Verificar si el sistema está calibrado
        
        Returns:
            bool: True si está calibrado
        """
        return self.calibrated and self.mm_per_pixel is not None and self.mm_per_pixel > 0
    
    def get_mm_per_pixel(self):
        """
        Obtener la escala de conversión actual
        
        Returns:
            float: Milímetros por píxel o None si no está calibrado
        """
        return self.mm_per_pixel if self.is_calibrated() else None
    
    def pixels_to_mm(self, pixels):
        """
        Convertir píxeles a milímetros
        
        Args:
            pixels (float): Medida en píxeles
        
        Returns:
            float: Medida en milímetros o None si no está calibrado
        """
        if not self.is_calibrated():
            return None
        
        return pixels * self.mm_per_pixel
    
    def mm_to_pixels(self, mm):
        """
        Convertir milímetros a píxeles
        
        Args:
            mm (float): Medida en milímetros
        
        Returns:
            float: Medida en píxeles o None si no está calibrado
        """
        if not self.is_calibrated():
            return None
        
        return mm / self.mm_per_pixel
    
    def get_calibration_info(self):
        """
        Obtener información detallada de la calibración
        
        Returns:
            dict: Información de calibración
        """
        if not self.is_calibrated():
            return {'calibrated': False}
        
        info = {
            'calibrated': True,
            'mm_per_pixel': self.mm_per_pixel,
            'pixels_per_mm': 1/self.mm_per_pixel,
            'calibration_data': self.calibration_data,
            'accuracy_estimate': self._estimate_accuracy()
        }
        
        return info
    
    def _estimate_accuracy(self):
        """
        Estimar la precisión de la calibración
        
        Returns:
            dict: Estimación de precisión
        """
        if not self.is_calibrated():
            return None
        
        # Precisión teórica basada en la escala
        pixel_size_mm = self.mm_per_pixel
        
        # Clasificar precisión
        # Clasificar precisión
        if pixel_size_mm < 0.01:  # < 0.01mm/píxel
            precision_level = "Muy Alta"
            expected_error = "±0.1mm"
        elif pixel_size_mm < 0.05:  # < 0.05mm/píxel
            precision_level = "Alta"
            expected_error = "±0.5mm"
        elif pixel_size_mm < 0.1:  # < 0.1mm/píxel
            precision_level = "Media"
            expected_error = "±1.0mm"
        else:
            precision_level = "Baja"
            expected_error = "±2.0mm"
        
        return {
            'pixel_size_mm': pixel_size_mm,
            'precision_level': precision_level,
            'expected_error': expected_error,
            'recommended_for': self._get_precision_recommendations(pixel_size_mm)
        }
    
    def _get_precision_recommendations(self, pixel_size_mm):
        """Obtener recomendaciones basadas en la precisión"""
        if pixel_size_mm < 0.01:
            return "Aplicaciones de alta precisión, investigación científica"
        elif pixel_size_mm < 0.05:
            return "Control de calidad industrial, mediciones profesionales"
        elif pixel_size_mm < 0.1:
            return "Aplicaciones generales, clasificación de productos"
        else:
            return "Estimaciones aproximadas, aplicaciones de baja precisión"
    
    def validate_calibration(self, test_measurements=None):
        """
        Validar la precisión de la calibración actual
        
        Args:
            test_measurements (list): Lista de mediciones de prueba [(pixels, real_mm), ...]
        
        Returns:
            dict: Resultados de validación
        """
        if not self.is_calibrated():
            return {'valid': False, 'error': 'No calibrado'}
        
        validation_results = {
            'valid': True,
            'mm_per_pixel': self.mm_per_pixel,
            'consistency_check': True
        }
        
        # Verificar consistencia básica
        if self.mm_per_pixel <= 0 or self.mm_per_pixel > 10:  # > 10mm/píxel es irreal
            validation_results['valid'] = False
            validation_results['consistency_check'] = False
            validation_results['error'] = 'Escala fuera de rango realista'
        
        # Test con mediciones conocidas si se proporcionan
        if test_measurements:
            errors = []
            for pixel_measurement, real_mm in test_measurements:
                predicted_mm = self.pixels_to_mm(pixel_measurement)
                if predicted_mm is not None:
                    error = abs(predicted_mm - real_mm)
                    relative_error = error / real_mm * 100
                    errors.append(relative_error)
            
            if errors:
                avg_error = np.mean(errors)
                max_error = max(errors)
                
                validation_results.update({
                    'test_measurements': len(test_measurements),
                    'average_error_percent': avg_error,
                    'max_error_percent': max_error,
                    'accuracy_acceptable': avg_error < 5.0  # < 5% error
                })
        
        return validation_results
    
    def get_calibration_history(self):
        """
        Obtener historial de calibraciones
        
        Returns:
            list: Lista de calibraciones previas
        """
        return self.calibration_history.copy()
    
    def clear_calibration(self):
        """Limpiar calibración actual"""
        self.mm_per_pixel = None
        self.calibrated = False
        self.calibration_data = {}
        print("🧹 Calibración limpiada")
    
    def export_calibration(self, filepath):
        """
        Exportar calibración a archivo
        
        Args:
            filepath (str): Ruta del archivo
        """
        if not self.is_calibrated():
            print("❌ No hay calibración para exportar")
            return False
        
        try:
            import json
            
            export_data = {
                'calibration_data': self.calibration_data,
                'calibration_history': self.calibration_history,
                'export_timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=4)
            
            print(f"💾 Calibración exportada a: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error exportando calibración: {e}")
            return False
    
    def import_calibration(self, filepath):
        """
        Importar calibración desde archivo
        
        Args:
            filepath (str): Ruta del archivo
        
        Returns:
            bool: True si se importó exitosamente
        """
        try:
            import json
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Validar estructura
            if 'calibration_data' not in data:
                print("❌ Archivo de calibración inválido")
                return False
            
            calibration_data = data['calibration_data']
            
            if 'mm_per_pixel' not in calibration_data:
                print("❌ Datos de calibración incompletos")
                return False
            
            # Aplicar calibración
            self.mm_per_pixel = calibration_data['mm_per_pixel']
            self.calibrated = True
            self.calibration_data = calibration_data
            
            # Importar historial si está disponible
            if 'calibration_history' in data:
                self.calibration_history = data['calibration_history']
            
            print(f"📁 Calibración importada desde: {filepath}")
            print(f"   🔄 Escala: {self.mm_per_pixel:.4f} mm/píxel")
            
            return True
            
        except Exception as e:
            print(f"❌ Error importando calibración: {e}")
            return False
    
    def draw_calibration_overlay(self, image):
        """
        Dibujar overlay de información de calibración en imagen
        
        Args:
            image (numpy.ndarray): Imagen donde dibujar
        
        Returns:
            numpy.ndarray: Imagen con overlay
        """
        if not self.is_calibrated():
            return image
        
        overlay_image = image.copy()
        
        # Información de calibración
        calib_text = f"Calibrado: {self.mm_per_pixel:.4f} mm/px"
        resolution_text = f"Resolución: {1/self.mm_per_pixel:.1f} px/mm"
        
        # Dibujar texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)
        thickness = 2
        
        # Posiciones
        h, w = overlay_image.shape[:2]
        y_offset = 30
        
        cv2.putText(overlay_image, calib_text, (w-350, y_offset), 
                   font, font_scale, color, thickness)
        cv2.putText(overlay_image, resolution_text, (w-350, y_offset + 25), 
                   font, font_scale, color, thickness)
        
        # Dibujar línea de referencia si hay datos de calibración
        if 'start_point' in self.calibration_data and 'end_point' in self.calibration_data:
            start = tuple(map(int, self.calibration_data['start_point']))
            end = tuple(map(int, self.calibration_data['end_point']))
            
            cv2.line(overlay_image, start, end, (255, 255, 0), 2)
            
            # Etiqueta de longitud
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            length_text = f"{self.calibration_data['ruler_length_mm']}mm"
            cv2.putText(overlay_image, length_text, (mid_x, mid_y-10),
                       font, 0.5, (255, 255, 0), 2)
        
        return overlay_image
    
    def configure_detection_parameters(self, **kwargs):
        """
        Configurar parámetros para detección automática de regla
        
        Args:
            **kwargs: Parámetros de configuración
        """
        for key, value in kwargs.items():
            if key in self.ruler_detection_config:
                self.ruler_detection_config[key] = value
                print(f"🔧 {key} = {value}")
        
        print("⚙️ Parámetros de detección actualizados")


# Función de utilidad para testing
def test_ruler_calibrator():
    """Función de test para el calibrador"""
    print("🧪 Iniciando test del calibrador de regla...")
    
    # Crear calibrador
    calibrator = RulerCalibrator()
    
    # Test de calibración manual
    start_point = (100, 100)
    end_point = (300, 100)  # Línea horizontal de 200 píxeles
    ruler_length_mm = 50.0  # 5cm
    
    success = calibrator.calibrate_with_line(start_point, end_point, ruler_length_mm)
    
    if success:
        print("✅ Test de calibración manual exitoso")
        
        # Test de conversiones
        test_pixels = 100
        test_mm = calibrator.pixels_to_mm(test_pixels)
        test_pixels_back = calibrator.mm_to_pixels(test_mm)
        
        print(f"🔄 Test conversión: {test_pixels}px → {test_mm:.2f}mm → {test_pixels_back:.1f}px")
        
        # Test de información
        info = calibrator.get_calibration_info()
        print(f"📊 Precisión estimada: {info['accuracy_estimate']['precision_level']}")
    else:
        print("❌ Test de calibración fallido")
    
    print("✅ Test del calibrador completado")


if __name__ == "__main__":
    test_ruler_calibrator()