import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub')
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class FaceShapeAnalyzer:
    """
    Анализ формы лица на основе модели metadome/face_shape_classification
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        self.processor = ViTImageProcessor.from_pretrained(
            "metadome/face_shape_classification"
        )
        self.model = ViTForImageClassification.from_pretrained(
            "metadome/face_shape_classification"
        ).to(self.device)
        self.model.eval()
    
    def analyze(self, image_path):
        """
        Анализирует форму лица на изображении.
        Возвращает: (форма, уверенность, все вероятности)
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return None, f"Ошибка загрузки изображения: {e}", None
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Прогноз
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Интерпретация
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
        
        # Маппинг классов (проверено на модели metadome)
        class_labels = {
            0: "Сердце",
            1: "Продолговатый",
            2: "Овал",
            3: "Круглый",
            4: "Квадрат"
        }
        
        shape = class_labels.get(predicted_idx, f"Неизвестная форма (класс {predicted_idx})")
        all_probs = {class_labels.get(i, f"Класс {i}"): prob.item() 
                    for i, prob in enumerate(probabilities)}
        
        return shape, confidence, all_probs
    
    def get_recommendations(self, shape):
        """Возвращает рекомендации по прическе"""
        recommendations = {
            "Овал": [
                " Подходит большинство причесок",
                " Длинные волосы с объемом",
                " Короткие стрижки (каре, боб)"
            ],
            "Круглый": [
                " Объем на макушке (удлиняет лицо)",
                " Асимметричные стрижки",
                " Длинные волосы с косым пробором",
                " Избегайте объема по бокам"
            ],
            "Квадрат": [
                " Мягкие волны и кудри",
                " Асимметричные челки",
                " Длинные волосы с объемом",
                " Избегайте прямых линий у челюсти"
            ],
            "Сердце": [
                " Объем у подбородка (каре)",
                " Боковой пробор",
                " Мягкие волны",
                " Избегайте объема на макушке"
            ],
            "Продолговатый": [
                " Короткие стрижки (до подбородка)",
                " Челка (скрывает длину)",
                " Волнистые волосы",
                " Избегайте длинных прямых волос"
            ]
        }
        return recommendations.get(shape, ["Нет рекомендаций"])

if __name__ == "__main__":
    analyzer = FaceShapeAnalyzer()
    
    shape, confidence, probs = analyzer.analyze('data/input.jpeg')
    
    if shape:
        for rec in analyzer.get_recommendations(shape):
            print(f"   {rec}")
    else:
        print(f"Ошибка: {confidence}")