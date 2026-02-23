import cv2
import numpy as np
from PIL import Image
import os


def refine_hair_mask(hair_mask_path, output_path='results/hair_only_refined.png'):
    """
    Улучшает маску волос от YBIGTA модели (морфология + сглаживание).
    """
    # Загрузка маски
    mask_pil = Image.open(hair_mask_path).convert('L')
    mask = np.array(mask_pil)
    
    # Загрузка оригинала
    original_pil = Image.open('data/input.jpeg').convert('RGB')
    original = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
    
    # === Морфология для сглаживания ===
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)
    
    # === Gaussian Blur для мягких краёв ===
    blurred_mask = cv2.GaussianBlur(eroded_mask, (5, 5), 0)
    
    # === Порог для чёткости ===
    _, final_mask = cv2.threshold(blurred_mask, 30, 255, cv2.THRESH_BINARY)
    
    # === Сохранение ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    b, g, r = cv2.split(original)
    dst = cv2.merge([b, g, r, final_mask])
    cv2.imwrite(output_path, dst)
    
    print(f"✅ Refined hair mask saved to {output_path}")


if __name__ == "__main__":
    refine_hair_mask('results/hair_only.png', 'results/hair_only_refined.png')