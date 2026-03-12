import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import os
import sys
import torchvision.transforms as transforms

sys.path.append("hair_seg_model/networks")
from hair_seg_model.networks import PSPNet


class HairSegmenter:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = PSPNet(num_class=1, base_network='resnet101')
        
        ckpt_path = "hair_seg_model/checkpoints/pspnet_hair.pth"
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Weights not found at {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'weight' in checkpoint:
            self.model.load_state_dict(checkpoint['weight'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        self.orig_size = image.size
        image_resized = image.resize((512, 512), Image.BILINEAR)
        im_tensor = self.transform(image_resized).unsqueeze(0)
        return im_tensor.to(self.device)

    def predict(self, image_path, output_path='results/hair_only.png', mask_path='results/hair_mask.png'):
        input_tensor = self.preprocess_image(image_path)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Sigmoid для 1 класса
        mask = torch.sigmoid(output)[0, 0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        
        # Ресайз к оригинальному размеру
        mask_pil = Image.fromarray(mask)
        mask_full = mask_pil.resize(self.orig_size, Image.BILINEAR)
        
        #  Сохраняем ЧИСТУЮ маску
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        mask_full.save(mask_path)
        
        # Создаем RGBA изображение с альфа-каналом
        original = Image.open(image_path).convert('RGBA')
        original.putalpha(mask_full)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        original.save(output_path)
        
        return mask_full


if __name__ == "__main__":
    segmenter = HairSegmenter()
    if os.path.exists('data/input.jpeg'):
        segmenter.predict('data/input.jpeg', 'results/hair_only.png', 'results/hair_mask.png')
    else:
        print("Please put an image at data/input.jpeg")