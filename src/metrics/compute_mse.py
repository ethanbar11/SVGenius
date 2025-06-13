'''
@File       :   compute_mse.py
@Description:   Implementation of MSE (Mean Squared Error) metric calculation.
'''
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base_metric import BaseMetric
import cairosvg
from io import BytesIO
from .data_util import clean_svg
class MSEDistanceCalculator(BaseMetric):
    def __init__(self):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.transform = transforms.ToTensor()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def collate_fn(self, batch):
        gt_imgs, gen_imgs = zip(*batch)
        return gt_imgs, gen_imgs
    
    def metric(self, gt_im, gen_im, **kwargs):
        """
        Calculate MSE for a single image pair
        Args:
            gt_im: Reference image (PIL Image)
            gen_im: Generated image (PIL Image)
        Returns:
            mse: Mean squared error value (lower is better)
        """
        # Ensure image sizes are consistent
        if gt_im.size != gen_im.size:
            gen_im = gen_im.resize(gt_im.size, Image.BICUBIC)
            
        # Convert to tensor
        gt_tensor = self.transform(gt_im).to(self.device)
        gen_tensor = self.transform(gen_im).to(self.device)
        
        # Calculate MSE
        mse = torch.mean((gt_tensor - gen_tensor) ** 2).item()
        return mse
    
    def calculate_score(self, batch, batch_size=64, update=True):
        """
        Calculate MSE for batch images
        Args:
            batch: Dictionary containing 'gt_im' and 'gen_im' keys with image lists as values
            batch_size: Size of each mini-batch
            update: Whether to update internal meter
        Returns:
            avg_mse: Average MSE value
            all_mse: List of MSE values for all image pairs
        """
        gt_images = batch['gt_im']
        gen_images = batch['gen_im']
        
        if len(gt_images) != len(gen_images):
            raise ValueError(f"Number of reference images ({len(gt_images)}) does not match number of generated images ({len(gen_images)})")
        
        # Create DataLoader for batch processing
        data_loader = DataLoader(
            list(zip(gt_images, gen_images)),
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True,
            shuffle=False
        )
        
        all_mse = []
        for batch_gt, batch_gen in tqdm(data_loader, desc="Calculating MSE"):
            for gt, gen in zip(batch_gt, batch_gen):
                mse = self.metric(gt, gen)
                all_mse.append(mse)
        
        if not all_mse:
            print("No valid MSE calculation results found.")
            return float("nan"), []
        
        avg_mse = sum(all_mse) / len(all_mse)
        if update:
            self.meter.update(avg_mse, len(all_mse))
            return self.meter.avg, all_mse
        else:
            return avg_mse, all_mse

def rasterize_svg(svg_string, resolution=224, dpi = 128, scale=2):
    try:
        svg_raster_bytes = cairosvg.svg2png(
            bytestring=svg_string,
            background_color='white',
            output_width=resolution, 
            output_height=resolution,
            dpi=dpi,
            scale=scale) 
        svg_raster = Image.open(BytesIO(svg_raster_bytes))
    except: 
        try:
            svg = clean_svg(svg_string)
            # svg = svg_string
            svg_raster_bytes = cairosvg.svg2png(
                bytestring=svg,
                background_color='white',
                output_width=resolution, 
                output_height=resolution,
                dpi=dpi,
                scale=scale) 
            svg_raster = Image.open(BytesIO(svg_raster_bytes))
        except:
            svg_raster = Image.new('RGB', (resolution, resolution), color = 'white')
    return svg_raster

if __name__ == '__main__':
    # Test code
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    # Example usage
    calculator = MSEDistanceCalculator()
    gt_svg='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="200" height="200"><path d="M516.3008 766.0544l-418.56-199.936-48.5888-189.44A62.1568 62.1568 0 0 1 128 301.7728l388.5056 121.0368 386.7648-127.4368a54.528 54.528 0 0 1 70.5536 62.208l-38.8096 199.68z" fill="
#FFDE8D" /><path d="M174.08 849.4592l-76.3392-283.3408L462.4896 138.24a68.8128 68.8128 0 0 1 104.0896-0.768l368.2304 419.84-62.9248 288.5632a81.92 81.92 0 0 1-80.1792 64.5632H253.3888A81.92 81.92 0 0 1 174.08 849.4592z" fill="
#FDBA54" /><path d="M508.0576 786.2784a40.96 40.96 0 0 1-30.72-13.7728l-141.4656-159.744a40.96 40.96 0 1 1 61.44-54.3232l110.592 124.8256 108.544-124.5696a40.96 40.96 0 1 1 61.7472 53.8112l-139.1616 159.744a40.96 40.96 0 0 1-30.72 14.0288z" fill="
#FFFFFF" /></svg>'''
    gen_svg='''<svg xmlns:svg="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 82.35294117647058 200">
        <g transform="translate(-55.0 5.0)"><circle cx="160" cy="40" r="30" fill="#800080" />
        <rect x="150" y="80" width="40" height="40" fill="#ffa500" />
        <path d="M140 140 L180 180 L120 180 Z" fill="#00ffff" />
    <g></svg>'''
    gt_svg = rasterize_svg(gt_svg)
    gen_svg = rasterize_svg(gen_svg)
    batch = {'gt_im': [gt_svg], 'gen_im': [gen_svg]}
    avg_mse, all_mse = calculator.calculate_score(batch)
    print(f"Average MSE: {avg_mse}")