'''
@File       :   AestheticScoreMetric.py
@Time       :   2023/02/12 14:54:00
@Modified   :   2025/04/24
@Description:   Modified AestheticScore using aesthetic-predictor-v2-5
'''
import torch
from PIL import Image
from tqdm import tqdm
from .base_metric import BaseMetric
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
import gc

class AestheticScoreMetric(BaseMetric):
    def __init__(self, batch_size=1):
        super().__init__()
        self.class_name = self.__class__.__name__
        
        # Use single device instead of device_map="auto"
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model and preprocessor
        print("Loading model...")
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # Removed device_map="auto" parameter, explicitly use single device
        )
        
        # Explicitly move entire model to same device
        self.model = self.model.float().to(self.device)
        print("Model loading complete!")
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        self.batch_size = batch_size
    
    def metric(self, gen_im, **kwargs):
        """
        Calculate aesthetic score for a single image
        Args:
            gen_im: Generated PIL Image
        Returns:
            aesthetic score (float)
        """
        # Ensure image is in RGB mode
        if gen_im.mode != 'RGB':
            gen_im = gen_im.convert('RGB')
        
        try:
            # Preprocess image
            pixel_values = (
                self.preprocessor(images=gen_im, return_tensors="pt")
                .pixel_values.float()
                .to(self.device)  # Use same device
            )
            
            # Predict aesthetic score
            with torch.inference_mode():
                # Ensure all operations are on same device
                output = self.model(pixel_values)
                
                # If output is on different device, move to same device first
                if hasattr(output, 'logits') and output.logits.device != pixel_values.device:
                    output.logits = output.logits.to(pixel_values.device)
                    
                score = output.logits.squeeze().float().cpu().numpy()
            
            # Clean GPU memory
            del pixel_values
            torch.cuda.empty_cache()
            gc.collect()
            
            return float(score)
        except Exception as e:
            print(f"Error processing single image: {e}")
            return 5.0  # Return default medium score
    
    def batch_score(self, images):
        """
        Score a batch of images
        Args:
            images: List of PIL Images
        Returns:
            list of aesthetic scores
        """
        # Process one image at a time to reduce memory pressure
        scores = []
        for img in images:
            score = self.metric(img)
            scores.append(score)
        
        return scores
    
    def calculate_score(self, batch, update=True):
        """
        Calculate aesthetic scores for a batch of images
        Args:
            batch: Dictionary with key 'gen_im' containing list of PIL Images
            update: Whether to update the meter
        Returns:
            average score and list of individual scores
        """
        gen_images = batch['gen_im']
        
        # Process images directly without DataLoader and multiprocessing
        all_scores = []
        try:
            for i in tqdm(range(0, len(gen_images), self.batch_size)):
                batch_images = gen_images[i:i+self.batch_size]
                batch_scores = self.batch_score(batch_images)
                all_scores.extend(batch_scores)
                
                # Show progress
                if i % 10 == 0:
                    print(f"Processed {i}/{len(gen_images)} images")
        
        except Exception as e:
            print(f"Error in batch processing images: {e}")
            
        if not all_scores:
            print("No valid scores found for aesthetic metric calculation.")
            return float("nan"), []
        
        avg_score = sum(all_scores) / len(all_scores)
        if update:
            self.meter.update(avg_score, len(all_scores))
            return self.meter.avg, all_scores
        else:
            return avg_score, all_scores

if __name__ == '__main__':
    # Set environment variables to help with debugging
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Set CUDA visible devices, use only one GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    try:
        # Example usage
        print("Initializing aesthetic score model...")
        metric = AestheticScoreMetric()
        
        # Open image and ensure RGB mode
        print("Loading test image...")
        image = Image.open('./aminate.png').convert('RGB')
        
        print("Calculating aesthetic score...")
        scores = metric.calculate_score({'gen_im': [image]})
        print(f"Aesthetic score: {scores[0]:.2f}, detailed scores: {scores[1]}")
    except Exception as e:
        print(f"Program execution error: {e}")