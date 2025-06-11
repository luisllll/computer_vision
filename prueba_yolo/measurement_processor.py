# src/measurement_processor.py - Módulo de procesamiento de mediciones
import numpy as np
import cv2
from datetime import datetime

class MeasurementProcessor:
    """
    Módulo para procesar mediciones de objetos detectados
    Filtra, valida y calcula mediciones precisas
    """
    def __init__(self):
        # Filtros específicos para almendras
        self.object_filters = {
            'min_confidence': 0.2,
            'min_area_pixels': 5,
            'max_area_pixels': 10000000,
            'aspect_ratio_range': (1., 6.0),  # Almendras son alargadas
            'min_width_pixels': 5,
            'min_height_pixels': 8
        }
        
        # Configuración de medición
        self.measurement_config = {
            'measurement_method': 'bounding_box',  # bounding_box, contour, ellipse
            'subpixel_precision': True,
            'outlier_detection': True,
            'statistical_filtering': True
        }
        
        # Estadísticas de sesión
        self.session_stats = {
            'total_detections': 0,
            'valid_measurements': 0,
            'filtered_out': 0,
            'in_range_count': 0,
            'out_of_range_count': 0
        }
        
        # Historial de mediciones
        self.measurement_history = []
    
    def process_detections(self, detections, mm_per_pixel, min_length_mm, max_length_mm):
        """
        Procesar lista de detecciones y generar mediciones
        
        Args:
            detections (list): Lista de detecciones de YOLO
            mm_per_pixel (float): Escala de conversión (None si no calibrado)
            min_length_mm (float): Longitud mínima válida
            max_length_mm (float): Longitud máxima válida
        
        Returns:
            list: Lista de mediciones procesadas
        """
        measurements = []
        
        self.session_stats['total_detections'] += len(detections)
        
        for i, detection in enumerate(detections):
            # Filtrar detección por calidad
            if not self._validate_detection(detection):
                self.session_stats['filtered_out'] += 1
                continue
            
            # Generar medición
            measurement = self._measure_object(
                detection, mm_per_pixel, min_length_mm, max_length_mm, i+1
            )
            
            if measurement is not None:
                measurements.append(measurement)
                self.session_stats['valid_measurements'] += 1
                
                # Actualizar estadísticas de rango
                if measurement['in_range']:
                    self.session_stats['in_range_count'] += 1
                else:
                    self.session_stats['out_of_range_count'] += 1
                
                # Añadir al historial
                self.measurement_history.append(measurement.copy())
        
        # Aplicar filtrado estadístico si está habilitado
        if self.measurement_config['statistical_filtering'] and len(measurements) > 3:
            measurements = self._apply_statistical_filtering(measurements)
        
        return measurements
    
    def _validate_detection(self, detection):
        """Validar objeto solo por forma y confianza (SIN filtro de área)"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 1. Filtro de confianza mínima
        if detection['confidence'] < self.object_filters['min_confidence']:
            return False
        
        # 2. Filtro de forma (aspect ratio) - OPCIONAL
        if self.object_filters.get('use_aspect_ratio_filter', False):
            aspect_ratio = max(width, height) / min(width, height)
            min_ratio, max_ratio = self.object_filters['aspect_ratio_range']
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                return False
        
        # 3. Filtro de dimensiones mínimas (evitar detecciones de 1-2 píxeles)
        min_dimension = self.object_filters.get('min_dimension_px', 5)
        if width < min_dimension or height < min_dimension:
            return False
        
        # NO filtro de área - acepta cualquier tamaño
        return True
    
    def _measure_object(self, detection, mm_per_pixel, min_length_mm, max_length_mm, object_id):
        """
        Medir un objeto individual
        
        Args:
            detection (dict): Detección de YOLO
            mm_per_pixel (float): Escala de conversión
            min_length_mm (float): Longitud mínima válida
            max_length_mm (float): Longitud máxima válida
            object_id (int): ID del objeto
        
        Returns:
            dict: Medición completa del objeto
        """
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Mediciones en píxeles
        width_px = x2 - x1
        height_px = y2 - y1
        area_px = detection.get('area', width_px * height_px)
        
        # Determinar longitud y ancho (longitud = dimensión mayor)
        length_px = max(width_px, height_px)
        width_px = min(width_px, height_px)
        
        # Aplicar precisión sub-píxel si está habilitada
        if self.measurement_config['subpixel_precision']:
            length_px, width_px = self._apply_subpixel_precision(
                bbox, length_px, width_px
            )
        
        # Crear medición base
        measurement = {
            'object_id': object_id,
            'bbox': bbox,
            'confidence': detection['confidence'],
            'class_name': detection.get('class_name', 'object'),
            
            # Mediciones en píxeles
            'length_px': length_px,
            'width_px': width_px,
            'area_px': area_px,
            'aspect_ratio': length_px / width_px if width_px > 0 else 0,
            
            # Mediciones en mm (si está calibrado)
            'length_mm': None,
            'width_mm': None,
            'area_mm2': None,
            
            # Estado de validación
            'in_range': False,
            'quality_score': self._calculate_quality_score(detection),
            
            # Metadatos
            'timestamp': datetime.now().isoformat(),
            'measurement_method': self.measurement_config['measurement_method']
        }
        
        # Convertir a milímetros si está calibrado
        if mm_per_pixel is not None:
            measurement.update({
                'length_mm': length_px * mm_per_pixel,
                'width_mm': width_px * mm_per_pixel,
                'area_mm2': area_px * (mm_per_pixel ** 2),
                'in_range': min_length_mm <= (length_px * mm_per_pixel) <= max_length_mm
            })
        
        return measurement
    
    def _apply_subpixel_precision(self, bbox, length_px, width_px):
        """
        Aplicar corrección de precisión sub-píxel
        
        Args:
            bbox (list): Bounding box [x1, y1, x2, y2]
            length_px (float): Longitud en píxeles
            width_px (float): Ancho en píxeles
        
        Returns:
            tuple: (length_corregida, width_corregido)
        """
        # Corrección simple basada en la confianza de la detección
        # En una implementación avanzada, se podría usar análisis de contornos
        
        # Factor de corrección (típicamente pequeño)
        correction_factor = 0.95  # Compensar por bounding box ligeramente más grande
        
        corrected_length = length_px * correction_factor
        corrected_width = width_px * correction_factor
        
        return corrected_length, corrected_width
    
    def _calculate_quality_score(self, detection):
        """
        Calcular puntuación de calidad para una detección
        
        Args:
            detection (dict): Detección de YOLO
        
        Returns:
            float: Puntuación entre 0.0 y 1.0
        """
        # Componentes de calidad
        confidence = detection['confidence']
        
        # Factor de área (objetos muy pequeños o grandes tienen menor calidad)
        area = detection.get('area', 0)
        optimal_area = 1000  # Área óptima aproximada
        area_factor = min(area / optimal_area, optimal_area / area) if area > 0 else 0
        area_factor = min(area_factor, 1.0)
        
        # Factor de aspect ratio (más cercano al ideal = mejor)
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 10
        
        # Aspect ratio ideal para almendras: ~2.0
        ideal_ratio = 2.0
        ratio_factor = min(aspect_ratio / ideal_ratio, ideal_ratio / aspect_ratio)
        ratio_factor = min(ratio_factor, 1.0)
        
        # Puntuación combinada
        quality_score = (confidence * 0.5) + (area_factor * 0.3) + (ratio_factor * 0.2)
        
        return min(quality_score, 1.0)
    
    def _apply_statistical_filtering(self, measurements):
        """
        Aplicar filtrado estadístico para remover outliers
        
        Args:
            measurements (list): Lista de mediciones
        
        Returns:
            list: Mediciones filtradas
        """
        if len(measurements) < 4:
            return measurements
        
        # Filtrar por longitud (remover outliers estadísticos)
        lengths = [m['length_px'] for m in measurements]
        q1 = np.percentile(lengths, 25)
        q3 = np.percentile(lengths, 75)
        iqr = q3 - q1
        
        # Límites para outliers (método IQR)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filtrar mediciones
        filtered_measurements = []
        outliers_removed = 0
        
        for measurement in measurements:
            length = measurement['length_px']
            if lower_bound <= length <= upper_bound:
                filtered_measurements.append(measurement)
            else:
                outliers_removed += 1
        
        if outliers_removed > 0:
            print(f"🔍 Filtrado estadístico: {outliers_removed} outliers removidos")
        
        return filtered_measurements
    
    def get_session_statistics(self):
        """
        Obtener estadísticas de la sesión actual
        
        Returns:
            dict: Estadísticas completas
        """
        stats = self.session_stats.copy()
        
        # Calcular porcentajes
        total = stats['total_detections']
        if total > 0:
            stats['validation_rate'] = (stats['valid_measurements'] / total) * 100
            stats['filter_rate'] = (stats['filtered_out'] / total) * 100
        else:
            stats['validation_rate'] = 0
            stats['filter_rate'] = 0
        
        # Estadísticas de mediciones válidas
        valid_total = stats['valid_measurements']
        if valid_total > 0:
            stats['in_range_percentage'] = (stats['in_range_count'] / valid_total) * 100
            stats['out_of_range_percentage'] = (stats['out_of_range_count'] / valid_total) * 100
        else:
            stats['in_range_percentage'] = 0
            stats['out_of_range_percentage'] = 0
        
        return stats
    
    def get_measurement_analysis(self, measurements):
        """
        Analizar lista de mediciones y generar estadísticas
        
        Args:
            measurements (list): Lista de mediciones
        
        Returns:
            dict: Análisis estadístico
        """
        if not measurements:
            return {'count': 0, 'analysis': 'No hay mediciones'}
        
        # Extraer valores para análisis
        lengths_mm = [m['length_mm'] for m in measurements if m['length_mm'] is not None]
        widths_mm = [m['width_mm'] for m in measurements if m['width_mm'] is not None]
        lengths_px = [m['length_px'] for m in measurements]
        quality_scores = [m['quality_score'] for m in measurements]
        
        analysis = {
            'count': len(measurements),
            'calibrated': len(lengths_mm) > 0
        }
        
        # Análisis de longitudes
        if lengths_mm:
            analysis.update({
                'length_stats_mm': {
                    'mean': np.mean(lengths_mm),
                    'std': np.std(lengths_mm),
                    'min': np.min(lengths_mm),
                    'max': np.max(lengths_mm),
                    'median': np.median(lengths_mm)
                },
                'width_stats_mm': {
                    'mean': np.mean(widths_mm),
                    'std': np.std(widths_mm),
                    'min': np.min(widths_mm),
                    'max': np.max(widths_mm),
                    'median': np.median(widths_mm)
                } if widths_mm else None
            })
        else:
            analysis.update({
                'length_stats_px': {
                    'mean': np.mean(lengths_px),
                    'std': np.std(lengths_px),
                    'min': np.min(lengths_px),
                    'max': np.max(lengths_px),
                    'median': np.median(lengths_px)
                }
            })
        
        # Análisis de calidad
        analysis['quality_stats'] = {
            'mean_quality': np.mean(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores),
            'high_quality_count': sum(1 for q in quality_scores if q > 0.8)
        }
        
        # Análisis de rangos
        in_range_count = sum(1 for m in measurements if m['in_range'])
        analysis['range_analysis'] = {
            'in_range': in_range_count,
            'out_of_range': len(measurements) - in_range_count,
            'in_range_percentage': (in_range_count / len(measurements)) * 100
        }
        
        return analysis
    
    def set_filters(self, **kwargs):
        """
        Configurar filtros de validación
        
        Args:
            **kwargs: Parámetros de filtros
        """
        for key, value in kwargs.items():
            if key in self.object_filters:
                self.object_filters[key] = value
                print(f"🔧 Filtro {key} = {value}")
        
        print("⚙️ Filtros de objeto actualizados")
    
    def set_measurement_config(self, **kwargs):
        """
        Configurar parámetros de medición
        
        Args:
            **kwargs: Parámetros de configuración
        """
        for key, value in kwargs.items():
            if key in self.measurement_config:
                self.measurement_config[key] = value
                print(f"🔧 Config {key} = {value}")
        
        print("⚙️ Configuración de medición actualizada")
    
    def reset_session_stats(self):
        """Reiniciar estadísticas de sesión"""
        self.session_stats = {
            'total_detections': 0,
            'valid_measurements': 0,
            'filtered_out': 0,
            'in_range_count': 0,
            'out_of_range_count': 0
        }
        print("📊 Estadísticas de sesión reiniciadas")
    
    def export_measurements(self, measurements, filepath, format='csv'):
        """
        Exportar mediciones a archivo
        
        Args:
            measurements (list): Lista de mediciones
            filepath (str): Ruta del archivo
            format (str): Formato ('csv', 'json', 'xlsx')
        """
        try:
            if format.lower() == 'csv':
                self._export_to_csv(measurements, filepath)
            elif format.lower() == 'json':
                self._export_to_json(measurements, filepath)
            elif format.lower() == 'xlsx':
                self._export_to_xlsx(measurements, filepath)
            else:
                print(f"❌ Formato no soportado: {format}")
                return False
            
            print(f"💾 Mediciones exportadas: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error exportando: {e}")
            return False
    
    def _export_to_csv(self, measurements, filepath):
        """Exportar a CSV"""
        import csv
        
        if not measurements:
            return
        
        fieldnames = [
            'object_id', 'confidence', 'class_name',
            'length_px', 'width_px', 'area_px', 'aspect_ratio',
            'length_mm', 'width_mm', 'area_mm2',
            'in_range', 'quality_score', 'timestamp'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for measurement in measurements:
                row = {field: measurement.get(field, '') for field in fieldnames}
                writer.writerow(row)
    
    def _export_to_json(self, measurements, filepath):
        """Exportar a JSON"""
        import json
        
        export_data = {
            'measurements': measurements,
            'session_stats': self.get_session_statistics(),
            'export_timestamp': datetime.now().isoformat(),
            'measurement_config': self.measurement_config,
            'object_filters': self.object_filters
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
    
    def _export_to_xlsx(self, measurements, filepath):
        """Exportar a Excel (requiere openpyxl)"""
        try:
            import pandas as pd
            
            df = pd.DataFrame(measurements)
            
            # Crear archivo Excel con múltiples hojas
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Mediciones', index=False)
                
                # Hoja de estadísticas
                stats_df = pd.DataFrame([self.get_session_statistics()])
                stats_df.to_excel(writer, sheet_name='Estadísticas', index=False)
                
                # Hoja de configuración
                config_df = pd.DataFrame([{
                    **self.measurement_config,
                    **self.object_filters
                }])
                config_df.to_excel(writer, sheet_name='Configuración', index=False)
            
        except ImportError:
            print("⚠️ pandas/openpyxl no disponible, usando CSV")
            self._export_to_csv(measurements, filepath.replace('.xlsx', '.csv'))


# Función de utilidad para testing
def test_measurement_processor():
    """Función de test para el procesador de mediciones"""
    print("🧪 Iniciando test del procesador de mediciones...")
    
    # Crear procesador
    processor = MeasurementProcessor()
    
    # Crear detecciones de prueba
    test_detections = [
        {
            'bbox': [100, 100, 150, 125],  # 50x25 píxeles
            'confidence': 0.85,
            'area': 1250,
            'class_name': 'almond'
        },
        {
            'bbox': [200, 200, 260, 230],  # 60x30 píxeles
            'confidence': 0.75,
            'area': 1800,
            'class_name': 'almond'
        },
        {
            'bbox': [300, 300, 310, 310],  # 10x10 píxeles (muy pequeño)
            'confidence': 0.90,
            'area': 100,
            'class_name': 'almond'
        }
    ]
    
    # Test sin calibración
    measurements = processor.process_detections(test_detections, None, 15, 30)
    print(f"✅ Test sin calibración: {len(measurements)} mediciones válidas")
    
    # Test con calibración
    mm_per_pixel = 0.5  # 0.5mm por píxel
    measurements_cal = processor.process_detections(test_detections, mm_per_pixel, 15, 30)
    print(f"✅ Test con calibración: {len(measurements_cal)} mediciones válidas")
    
    # Test de análisis
    analysis = processor.get_measurement_analysis(measurements_cal)
    print(f"📊 Análisis: {analysis['count']} mediciones, calidad promedio: {analysis['quality_stats']['mean_quality']:.2f}")
    
    # Test de estadísticas
    stats = processor.get_session_statistics()
    print(f"📈 Estadísticas: {stats['valid_measurements']}/{stats['total_detections']} válidas ({stats['validation_rate']:.1f}%)")
    
    print("✅ Test del procesador completado")


if __name__ == "__main__":
    test_measurement_processor()