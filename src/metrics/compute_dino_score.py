import torch
from torch.utils.data import DataLoader
from .base_metric import BaseMetric
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import torch.nn as nn
import cairosvg
from io import BytesIO
class DINOScoreCalculator(BaseMetric): 
    def __init__(self, config=None, device='cuda'):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config
        self.model, self.processor = self.get_DINOv2_model("base")
        self.model = self.model.to(device)
        self.device = device

        self.metric = self.calculate_DINOv2_similarity_score

    def get_DINOv2_model(self, model_size):
        if model_size == "small":
            model_size = "./dinov2-base"
        elif model_size == "base":
            model_size = "./dinov2-base"
        elif model_size == "large":
            model_size = "./dinov2-base"
        else:
            raise ValueError(f"model_size should be either 'small', 'base' or 'large', got {model_size}")
        return AutoModel.from_pretrained(model_size), AutoImageProcessor.from_pretrained(model_size)
    def process_input(self, image, processor):
        if isinstance(image, str):
            image = Image.open(image)
        
        if isinstance(image, Image.Image):
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(image, torch.Tensor):
            features = image.unsqueeze(0) if image.dim() == 1 else image
        else:
            raise ValueError("Input must be a file path, PIL Image, or tensor of features")
        return features

    def calculate_DINOv2_similarity_score(self, **kwargs):
        image1 = kwargs.get('gt_im')
        image2 = kwargs.get('gen_im')
        
        # Check if images exist
        if image1 is None or image2 is None:
            print("Warning: Missing image input")
            return 0.0
            
        try:
            features1 = self.process_input(image1, self.processor)
            features2 = self.process_input(image2, self.processor)

            cos = nn.CosineSimilarity(dim=1)
            sim = cos(features1, features2).item()
            sim = (sim + 1) / 2

            return sim
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0          
    def calculate_score(self, batch, update=True):
        """
        Calculate DINO similarity scores for a batch of images
        Args:
            batch: Dictionary with keys 'gen_im' and 'gt_im' containing lists of PIL Images
            update: Whether to update the meter
        Returns:
            average score and list of individual scores
        """
        gen_images = batch.get('gen_im', [])
        gt_images = batch.get('gt_im', [])
        
        # Ensure both lists have same length
        if len(gen_images) != len(gt_images):
            print("Warning: Generated images and reference images count mismatch")
            # Use shorter list length
            length = min(len(gen_images), len(gt_images))
            gen_images = gen_images[:length]
            gt_images = gt_images[:length]
            
        all_scores = []
        for i in tqdm(range(len(gen_images))):
            # Calculate similarity score for each image pair
            score = self.calculate_DINOv2_similarity_score(
                gen_im=gen_images[i], 
                gt_im=gt_images[i]
            )
            all_scores.append(score)
        
        if not all_scores:
            print("No valid scores found for DINO similarity calculation.")
            return float("nan"), []
        
        avg_score = sum(all_scores) / len(all_scores)
        if update:
            self.meter.update(avg_score, len(all_scores))
            return self.meter.avg, all_scores
        else:
            return avg_score, all_scores
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
            # svg = clean_svg(svg_string)
            svg=svg_string
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
    # try:
    calculator = DINOScoreCalculator()
    gt_svg='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="200" height="200"><path d="M516.3008 766.0544l-418.56-199.936-48.5888-189.44A62.1568 62.1568 0 0 1 128 301.7728l388.5056 121.0368 386.7648-127.4368a54.528 54.528 0 0 1 70.5536 62.208l-38.8096 199.68z" fill="
#FFDE8D" /><path d="M174.08 849.4592l-76.3392-283.3408L462.4896 138.24a68.8128 68.8128 0 0 1 104.0896-0.768l368.2304 419.84-62.9248 288.5632a81.92 81.92 0 0 1-80.1792 64.5632H253.3888A81.92 81.92 0 0 1 174.08 849.4592z" fill="
#FDBA54" /><path d="M508.0576 786.2784a40.96 40.96 0 0 1-30.72-13.7728l-141.4656-159.744a40.96 40.96 0 1 1 61.44-54.3232l110.592 124.8256 108.544-124.5696a40.96 40.96 0 1 1 61.7472 53.8112l-139.1616 159.744a40.96 40.96 0 0 1-30.72 14.0288z" fill="
#FFFFFF" /></svg>'''
    gen_svg='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="200" height="200">
<path d="M516.3008 766.0544c-139.52-66.6453-279.04-133.2907-418.56-199.936l-48.5888-189.44A62.1568 62.1568 0 0 1 128 301.7728l388.5056 121.0368 386.7648-127.4368a54.528 54.528 0 0 1 70.5536 62.208l-38.8096 199.68c-83.8656 27.8857-167.7312 55.7714-251.5968 83.657C628.5744 661.28 573.7472 681.84 516.3008 686.4c-57.4464 4.56-117.4848-6.88-155.648-41.984-38.1632-35.104-54.4512-93.8667-25.6-131.6267 28.8512-37.76 103.808-54.5067 150.5696-33.4933 46.7616 21.0133 65.3184 79.8667 40.3584 126.72-24.96 46.8533-93.4144 81.7067-149.2288 67.84-55.8144-13.8667-98.9696-76.4533-85.4016-128.4267 13.568-51.9733 83.7888-93.3333 140.6016-85.3333 56.8128 8 100.2496 65.28 106.3616 123.4667 6.112 58.1867-25.1392 117.2533-70.5024 142.4533-45.3632 25.2-104.8576 16.5333-145.6128-17.0667-40.7552-33.6-62.7712-92.1333-46.592-139.4667C292.1984 572.16 346.5728 536 398.4832 536c51.9104 0 101.2992 36.32 120.832 84.48 19.5328 48.16 9.2032 108.1601-29.9008 141.12-39.104 32.96-107.0848 38.88-155.5712 9.6-48.4864-29.28-77.4656-94.72-63.7184-147.84 13.7472-53.12 70.1952-93.92 125.1264-93.12 54.9312 0.8 108.3776 43.2 123.136 96 14.7584 52.8-9.1776 116-60.9088 149.12 M700 700 C750 730 800 720 830 680 C860 640 850 570 800 530 C750 490 670 490 620 530 C570 570 560 650 600 700 C640 750 720 760 770 730 C820 700 840 630 820 580 C800 530 750 490 700 480 C650 470 590 490 550 530 C510 570 490 630 500 680 C510 730 550 770 600 790 C650 810 710 810 760 790 C810 770 850 730 870 680" fill="#FFDE8D" />
<path d="M174.08 849.4592l-76.3392-283.3408L462.4896 138.24a68.8128 68.8128 0 0 1 104.0896-0.768l368.2304 419.84-62.9248 288.5632a81.92 81.92 0 0 1-80.1792 64.5632H253.3888A81.92 81.92 0 0 1 174.08 849.4592z" fill="
#FDBA54" /><path d="M508.0576 786.2784a40.96 40.96 0 0 1-30.72-13.7728l-141.4656-159.744a40.96 40.96 0 1 1 61.44-54.3232l110.592 124.8256 108.544-124.5696a40.96 40.96 0 1 1 61.7472 53.8112l-139.1616 159.744a40.96 40.96 0 0 1-30.72 14.0288z" fill="
#FFFFFF" />
</svg>'''
    gt_svg = rasterize_svg(gt_svg)
    gen_svg = rasterize_svg(gen_svg)
    batch = {'gt_im': [gt_svg], 'gen_im': [gen_svg]}
    avg_score, all_scores = calculator.calculate_score(batch)
    print(f"Average DINO score: {avg_score}")
    print(f"Detailed scores: {all_scores}")
    # except Exception as e:
    #     print(f"Program execution error: {e}")