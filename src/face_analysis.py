import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')


class FaceShapeAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )

    def analyze(self, image_path):
        try:
            image_pil = Image.open(image_path).convert('RGB')
            image_np = np.array(image_pil)
        except Exception as e:
            return None, "Failed to load image"
        
        height, width = image_np.shape[:2]
        results = self.face_mesh.process(image_np)
        
        if not results.multi_face_landmarks:
            return None, "No face detected"
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Измерения
        forehead_width = abs(landmarks[151].x - landmarks[381].x) * width
        cheekbone_width = abs(landmarks[234].x - landmarks[454].x) * width
        jaw_width = abs(landmarks[152].x - landmarks[377].x) * width
        face_length = abs(landmarks[10].y - landmarks[152].y) * height
        
        ratios = {
            'forehead': forehead_width,
            'cheekbone': cheekbone_width,
            'jaw': jaw_width,
            'length': face_length
        }
        
        shape, confidence = self._classify_face_shape(ratios)
        recommendations = self._get_recommendations(shape)
        
        return {
            'shape': shape,
            'confidence': confidence,
            'measurements': ratios,
            'recommendations': recommendations
        }, "Success"
    
    def _classify_face_shape(self, ratios):
        length_to_width = ratios['length'] / max(ratios['cheekbone'], 1)
        forehead_to_jaw = ratios['forehead'] / max(ratios['jaw'], 1)
        cheekbone_to_jaw = ratios['cheekbone'] / max(ratios['jaw'], 1)
        
        if length_to_width > 1.3 and abs(forehead_to_jaw - 1.0) < 0.2:
            return "Овал", 0.8
        if abs(length_to_width - 1.0) < 0.15 and cheekbone_to_jaw < 1.1:
            return "Круг", 0.7
        if abs(length_to_width - 1.0) < 0.15 and forehead_to_jaw < 1.1:
            return "Квадрат", 0.7
        if forehead_to_jaw > 1.2:
            return "Сердце", 0.75
        if ratios['cheekbone'] > ratios['forehead'] and ratios['cheekbone'] > ratios['jaw']:
            return "Ромб", 0.75
        return "Овал (предположительно)", 0.5
    
    def _get_recommendations(self, shape):
        recommendations = {
            "Овал": ["✅ Подходит большинство причесок", "✅ Длинные волосы с объемом", "✅ Короткие стрижки (каре, боб)"],
            "Круг": ["✅ Объем на макушке", "✅ Асимметричные стрижки", "⚠️ Избегайте объема по бокам"],
            "Квадрат": ["✅ Мягкие волны и кудри", "✅ Асимметричные челки", "⚠️ Избегайте прямых линий у челюсти"],
            "Сердце": ["✅ Объем у подбородка", "✅ Боковой пробор", "⚠️ Избегайте объема на макушке"],
            "Ромб": ["✅ Объем у лба и подбородка", "✅ Челка", "⚠️ Избегайте объема у скул"]
        }
        return recommendations.get(shape, ["Нет рекомендаций"])
    
    def close(self):
        self.face_mesh.close()


if __name__ == "__main__":
    analyzer = FaceShapeAnalyzer()
    result, status = analyzer.analyze('data/input.jpeg')
    
    if result:
        print(f"\n{'='*50}")
        print(f"📊 Форма лица: {result['shape']} (уверенность: {result['confidence']:.0%})")
        print(f"{'='*50}")
        print(f"💡 Рекомендации:")
        for rec in result['recommendations']:
            print(f"   {rec}")
        print(f"{'='*50}\n")
    else:
        print(f"❌ Error: {status}")
    
    analyzer.close()