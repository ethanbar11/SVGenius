'''
@File       :   compute_clip_score.py
@Time       :   2023/02/12 13:14:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   CLIPScore Metric for SVGMetrics framework.
* Based on CLIP code base
* https://github.com/openai/CLIP
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base_metric import BaseMetric
import os

class CLIPScoreCalculator(BaseMetric):
    def __init__(self, download_root=None):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set download directory
        if download_root is None:
            download_root = os.path.expanduser("~/.cache/clip")
        
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device, jit=False, 
                                                    download_root=download_root)
        
        if self.device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(self.clip_model)
        
        # Freeze logit_scale parameter
        self.clip_model.logit_scale.requires_grad_(False)
    
    def collate_fn(self, batch):
        images, captions = zip(*batch)
        return images, captions
    
    def metric(self, gen_im, caption=None, **kwargs):
        """
        Calculate CLIP similarity score between single image and text
        Args:
            gen_im: Generated image (PIL Image)
            caption: Text description
        Returns:
            score: CLIP score
        """
        if caption is None:
            raise ValueError("CLIP scoring requires caption parameter")
        
        # Ensure image is in RGB format
        if gen_im.mode != 'RGB':
            gen_im = gen_im.convert('RGB')
        
        # Text encoding
        text = clip.tokenize([caption], truncate=True).to(self.device)
        with torch.no_grad():
            txt_features = F.normalize(self.clip_model.encode_text(text))
        
        # Image encoding
        image = self.preprocess(gen_im).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = F.normalize(self.clip_model.encode_image(image))
        
        # Calculate score (cosine similarity)
        rewards = torch.sum(torch.mul(txt_features, image_features), dim=1, keepdim=True)
        
        return rewards.item() * 100.0  # Multiply by 100 to make score in 0-100 range, consistent with common CLIP scores
    
    def calculate_score(self, batch, batch_size=32, update=True):
        """
        Calculate CLIP scores for batch of images
        Args:
            batch: Dictionary containing 'gen_im' and 'caption' keys
            batch_size: Size of each mini-batch
            update: Whether to update internal meter
        Returns:
            avg_score: Average CLIP score
            all_scores: List of CLIP scores for all images
        """
        gen_images = batch['gen_im']
        
        if 'caption' not in batch:
            raise ValueError("CLIP scoring requires 'caption' field")
        captions = batch['caption']
        
        if len(gen_images) != len(captions):
            raise ValueError(f"Number of generated images ({len(gen_images)}) does not match number of captions ({len(captions)})")
        
        # Create DataLoader for batch processing
        data_loader = DataLoader(
            list(zip(gen_images, captions)),
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True,
            shuffle=False
        )
        
        all_scores = []
        for batch_images, batch_captions in tqdm(data_loader, desc="Computing CLIP scores"):
            for img, cap in zip(batch_images, batch_captions):
                try:
                    score = self.metric(img, cap)
                    all_scores.append(score)
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue
        
        if not all_scores:
            print("No valid CLIP scores found.")
            return float("nan"), []
        
        avg_score = sum(all_scores) / len(all_scores)
        if update:
            self.meter.update(avg_score, len(all_scores))
            return self.meter.avg, all_scores
        else:
            return avg_score, all_scores
    
    def rank_images(self, prompt, image_paths):
        """
        Rank images based on similarity to text
        Args:
            prompt: Text description
            image_paths: List of image paths
        Returns:
            indices: Ranking indices (starting from 1)
            scores: List of similarity scores
        """
        # Text encoding
        text = clip.tokenize([prompt], truncate=True).to(self.device)
        with torch.no_grad():
            txt_feature = F.normalize(self.clip_model.encode_text(text))
        
        txt_set = []
        img_set = []
        for img_path in image_paths:
            # Image encoding
            pil_image = Image.open(img_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = F.normalize(self.clip_model.encode_image(image))
            img_set.append(image_features)
            txt_set.append(txt_feature)
        
        txt_features = torch.cat(txt_set, 0).float()  # [image_num, feature_dim]
        img_features = torch.cat(img_set, 0).float()  # [image_num, feature_dim]
        rewards = torch.sum(torch.mul(txt_features, img_features), dim=1, keepdim=True)
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()

if __name__ == '__main__':
    calculator = CLIPScoreCalculator()
    # Example 1: Calculate score for single image
    batch = {
        'gen_im': [Image.open('./example.png')], 
        'caption': ['A portrait of a smiling person']
    }
    avg_score, all_scores = calculator.calculate_score(batch)
    print(f"Average CLIP score: {avg_score}")
    
    # Example 2: Rank multiple images
    # image_paths = ['/path/to/image1.png', '/path/to/image2.png', '/path/to/image3.png']
    # prompt = 'A beautiful landscape'
    # ranks, scores = calculator.rank_images(prompt, image_paths)
    # print(f"Rankings: {ranks}")
    # print(f"Scores: {scores}")