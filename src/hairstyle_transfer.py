import torch
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
from PIL import Image
import numpy as np
import cv2
import os
import warnings

warnings.filterwarnings('ignore')


class HairstyleTransfer:
    """
    Перенос прически с одного человека на другого с помощью Stable Diffusion.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        print("⏳ Загрузка Stable Diffusion Inpainting...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            safety_checker=None
        )
        self.pipe = self.pipe.to(self.device)
        print(f"✅ Stable Diffusion загружен на {self.device}")
    
    def extract_hair_from_reference(self, reference_image_path, expand_pixels=20):
        """
        Выделяет прическу из "донорского" фото (с использованием сегментации).
        
        Возвращает: маска прически
        """
        # В реальном проекте здесь должна быть сегментация волос
        # Пока используем простой подход: пользователь сам загружает маску
        # или используем эвристику
        
        reference = cv2.imread(reference_image_path)
        if reference is None:
            raise ValueError(f"Не удалось загрузить изображение: {reference_image_path}")
        
        # Простая эвристика: предполагаем что волосы в верхней части изображения
        h, w = reference.shape[:2]
        hair_mask = np.zeros((h, w), dtype=np.uint8)
        hair_mask[:int(h * 0.5), :] = 255  # Верхняя половина = волосы
        
        # Расширение маски
        kernel = np.ones((expand_pixels, expand_pixels), np.uint8)
        hair_mask = cv2.dilate(hair_mask, kernel, iterations=1)
        
        return hair_mask
    
    def prepare_target_mask(self, target_image_path, expand_pixels=30):
        """
        Подготавливает маску для целевого изображения (где будет прическа).
        """
        # В реальном проекте используем PSPNet для сегментации волос
        # Пока простая маска
        
        target = cv2.imread(target_image_path)
        h, w = target.shape[:2]
        
        # Маска для зоны прически (верхняя часть головы)
        target_mask = np.zeros((h, w), dtype=np.uint8)
        target_mask[:int(h * 0.4), :] = 255
        
        # Расширение
        kernel = np.ones((expand_pixels, expand_pixels), np.uint8)
        target_mask = cv2.dilate(target_mask, kernel, iterations=1)
        
        return target_mask
    
    def transfer_hairstyle(self, 
                          target_image_path,
                          reference_image_path,
                          prompt="professional portrait, realistic hair, high quality",
                          negative_prompt="ugly, blurry, distorted, bad anatomy, unnatural hair",
                          num_inference_steps=50,
                          guidance_scale=7.5,
                          strength=0.8,
                          output_path='results/hairstyle_transferred.png'):
        """
        Переносит прическу с reference на target изображение.
        
        Параметры:
            target_image_path: путь к фото пользователя (куда переносим)
            reference_image_path: путь к фото с желаемой прической
            prompt: текстовое описание
            negative_prompt: негативные подсказки
            num_inference_steps: шаги генерации
            guidance_scale: сила следования промпту
            strength: сила изменения (0.0-1.0, чем выше - тем больше изменений)
            output_path: путь для сохранения
        
        Возвращает:
            PIL изображение с перенесённой прической
        """
        print(f"\n🎨 Перенос прически...")
        print(f"   Целевое изображение: {target_image_path}")
        print(f"   Донорское изображение: {reference_image_path}")
        
        # Загрузка изображений
        target = Image.open(target_image_path).convert("RGB")
        reference = Image.open(reference_image_path).convert("RGB")
        
        # Подготовка масок
        print("⏳ Подготовка масок...")
        target_mask = self.prepare_target_mask(target_image_path)
        reference_hair_mask = self.extract_hair_from_reference(reference_image_path)
        
        # Преобразование маски в PIL
        mask_pil = Image.fromarray(target_mask).convert("L")
        
        # Изменение размера до 512х512 (для Stable Diffusion)
        target_resized = target.resize((512, 512))
        mask_resized = mask_pil.resize((512, 512))
        
        print("⏳ Запуск генерации с переносом стиля...")
        
        # Генерация с использованием reference image как дополнительного промпта
        # В простом режиме используем текстовый промпт, описывающий прическу
        result = self.pipe(
            prompt=prompt,
            image=target_resized,
            mask_image=mask_resized,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength
        ).images[0]
        
        print("✅ Генерация завершена!")
        
        # Восстановление оригинального размера
        result_full = result.resize(target.size, Image.BILINEAR)
        
        # Сохранение результата
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_full.save(output_path)
        print(f"✅ Результат сохранён: {output_path}")
        
        return result_full
    
    def transfer_with_ip_adapter(self,
                                 target_image_path,
                                 reference_image_path,
                                 prompt="professional portrait, realistic hair",
                                 negative_prompt="ugly, blurry, distorted",
                                 num_inference_steps=50,
                                 guidance_scale=7.5,
                                 output_path='results/hairstyle_transferred_ip.png'):
        """
        Перенос прически с использованием IP-Adapter (более точный перенос стиля).
        
        Требует установки: pip install ip-adapter
        """
        try:
            from ip_adapter import IPAdapter
            
            print("⏳ Загрузка IP-Adapter...")
            
            # Загрузка изображений
            target = Image.open(target_image_path).convert("RGB")
            reference = Image.open(reference_image_path).convert("RGB")
            
            # Подготовка маски
            target_mask = self.prepare_target_mask(target_image_path)
            mask_pil = Image.fromarray(target_mask).convert("L")
            
            # Изменение размера
            target_resized = target.resize((512, 512))
            reference_resized = reference.resize((512, 512))
            mask_resized = mask_pil.resize((512, 512))
            
            print("⏳ Запуск генерации с IP-Adapter...")
            
            # Генерация с использованием reference image
            result = self.pipe(
                prompt=prompt,
                image=target_resized,
                mask_image=mask_resized,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ip_adapter_image=reference_resized  # IP-Adapter image prompt
            ).images[0]
            
            result_full = result.resize(target.size, Image.BILINEAR)
            result_full.save(output_path)
            print(f"✅ Результат сохранён: {output_path}")
            
            return result_full
            
        except ImportError:
            print("⚠️ IP-Adapter не установлен. Установите: pip install ip-adapter")
            print("Используем обычный режим...")
            return self.transfer_hairstyle(
                target_image_path, reference_image_path,
                prompt, negative_prompt,
                num_inference_steps, guidance_scale,
                output_path=output_path
            )


# ============================================================================
# Примеры использования
# ============================================================================

if __name__ == "__main__":
    # Создание трансферера
    transfer = HairstyleTransfer()
    
    # Перенос прически
    result = transfer.transfer_hairstyle(
        target_image_path='data/input.jpeg',  # Фото пользователя
        reference_image_path='data/reference_hairstyle.jpg',  # Фото с прической
        prompt="professional portrait, realistic long wavy hair, soft lighting",
        negative_prompt="ugly, blurry, distorted, bad anatomy, unnatural hair",
        num_inference_steps=50,
        guidance_scale=7.5,
        strength=0.8,
        output_path='results/hairstyle_transferred.png'
    )