'''
@File       :   compute_hpsv2.py
@Description:   Implementation of HPSv2(Human Preference Score v2) based image evaluation metric using official library
@Reference  :   https://github.com/tgxs002/HPSv2
'''
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base_metric import BaseMetric  # Assume BaseMetric is in the same directory
import hpsv2  # Import official HPSv2 library

class HPSv2Calculator(BaseMetric):
    def __init__(self, version="v2.1"):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.version = version  # HPSv2 version, default use v2.1
    
    # def metric(self, gen_im, caption=None, **kwargs):
    #     """
    #     Calculate HPSv2 score for a single image
    #     Args:
    #         gen_im: Generated image (PIL Image)
    #         caption: Text prompt (required)
    #     Returns:
    #         score: HPSv2 score
    #     """
    #     if caption is None:
    #         raise ValueError("HPSv2 requires caption parameter")
            
    #     # Use official API to calculate score
    #     score = hpsv2.score(gen_im, caption, hps_version=self.version)
        
    #     return score
    
    def metric(self, gen_im, caption=None, **kwargs):
        """
        Calculate HPSv2 score for a single image
        Args:
            gen_im: Generated image (PIL Image)
            caption: Text prompt (required)
        Returns:
            score: HPSv2 score
        """
        if caption is None:
            raise ValueError("HPSv2 requires caption parameter")
            
        # Ensure image is in RGB format
        if gen_im.mode != 'RGB':
            gen_im = gen_im.convert('RGB')
            
        # Use official API to calculate score
        score_result = hpsv2.score(gen_im, caption, hps_version=self.version)
        
        # Handle return value
        if isinstance(score_result, list):
            # If return value is a list, take the first element as score
            if score_result:  # Ensure list is not empty
                return float(score_result[0])
            else:
                return float("nan")  # If list is empty, return NaN
        else:
            # If not a list, try to return directly
            return score_result
    
    def calculate_score(self, batch, update=True):
        """
        Calculate HPSv2 scores for batch images
        Args:
            batch: Dictionary containing 'gen_im' and 'caption' keys
            update: Whether to update internal meter
        Returns:
            avg_score: Average HPSv2 score
            all_scores: List of HPSv2 scores for all images
        """
        gen_images = batch['gen_im']
        
        if 'caption' not in batch:
            raise ValueError("HPSv2 calculation requires 'caption' field")
        captions = batch['caption']
        
        if len(gen_images) != len(captions):
            raise ValueError(f"Number of generated images ({len(gen_images)}) does not match number of captions ({len(captions)})")
        
        all_scores = []
        for i, (img, caption) in enumerate(tqdm(zip(gen_images, captions), total=len(gen_images), desc="Calculating HPSv2 scores")):
            try:
                # Ensure image is in RGB format
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Official API supports batch processing, but we process one by one for better error tracking
                score_result = self.metric(img, caption)
                
                # Check score_result type and handle accordingly
                if isinstance(score_result, list):
                    # If return value is a list, take the first element as score
                    if score_result:  # Ensure list is not empty
                        score_value = float(score_result[0])
                        all_scores.append(score_value)
                    else:
                        print(f"Warning: Score result for image {i+1} is an empty list")
                else:
                    # If not a list, try to convert directly to float
                    score_value = float(score_result)
                    all_scores.append(score_value)
                    
            except Exception as e:
                print(f"Error processing image {i+1}: {e}")
                continue
        
        if not all_scores:
            print("No valid HPSv2 calculation results found.")
            return float("nan"), []
        
        # Calculate average score
        avg_score = sum(all_scores) / len(all_scores)
        if update:
            self.meter.update(avg_score, len(all_scores))
            return self.meter.avg, all_scores
        else:
            return avg_score, all_scores
            
if __name__ == '__main__':
    # Test code
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    # Example usage
    calculator = HPSv2Calculator()
    batch = {'gen_im': [Image.open('example1.png')], 'caption': ['A red flower in bloom']}
    avg_score, all_scores = calculator.calculate_score(batch)
    print(f"Average HPSv2 score: {avg_score}")