from torchvision.transforms import ToTensor, Normalize, Resize
import torch
from torch.utils.data import DataLoader
from .base_metric import BaseMetric
import lpips
from tqdm import tqdm
from PIL import Image
import numpy as np
import cairosvg
from io import BytesIO
from .data_util import clean_svg

class LPIPSDistanceCalculator(BaseMetric): 
    def __init__(self, config=None, device='cuda', size=(224, 224)):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config
        self.model = lpips.LPIPS(net='vgg').to(device)
        self.metric = self.LPIPS
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = device
        self.size = size  # Add default size
        self.resize = Resize(size)  # Add resize transform

    def LPIPS(self, tensor_image1, tensor_image2):
        tensor_image1, tensor_image2 = tensor_image1.to(self.device), tensor_image2.to(self.device)
        return self.model(tensor_image1, tensor_image2)
    
    def to_tensor_transform(self, pil_img):
        """Convert image to normalized tensor, handle multiple input types and resize"""
        # Handle PIL images
        if isinstance(pil_img, Image.Image):
            # Ensure image is in RGB format
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            # Resize first
            pil_img = self.resize(pil_img)
            return self.normalize(self.to_tensor(pil_img))
        
        # Handle numpy arrays
        elif isinstance(pil_img, np.ndarray):
            # If uint8 array, convert to PIL first
            if pil_img.dtype == np.uint8:
                return self.to_tensor_transform(Image.fromarray(pil_img))
            # If already float array, assume it's in [0,1] range
            else:
                img = Image.fromarray((pil_img * 255).astype(np.uint8))
                return self.to_tensor_transform(img)
        
        # Handle tensor input
        elif isinstance(pil_img, torch.Tensor):
            # Ensure correct format (C,H,W) and in [0,1] range
            if pil_img.dim() == 3 and pil_img.shape[0] == 3:
                if pil_img.max() > 1.0 and pil_img.min() >= 0:
                    pil_img = pil_img.float() / 255.0
                # Resize
                pil_img = torch.nn.functional.interpolate(
                    pil_img.unsqueeze(0), 
                    size=self.size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                return self.normalize(pil_img)
            else:
                raise ValueError(f"Incorrect tensor input format, expected (3,H,W), but got {pil_img.shape}")
        
        else:
            raise TypeError(f"Unsupported input type: {type(pil_img)}")

    def collate_fn(self, batch):
        gt_imgs, gen_imgs = zip(*batch)
        
        # Handle potential type inconsistency issues
        tensor_gt_imgs = []
        tensor_gen_imgs = []
        
        for gt_img, gen_img in zip(gt_imgs, gen_imgs):
            try:
                tensor_gt = self.to_tensor_transform(gt_img)
                tensor_gen = self.to_tensor_transform(gen_img)
                tensor_gt_imgs.append(tensor_gt)
                tensor_gen_imgs.append(tensor_gen)
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        if not tensor_gt_imgs:
            return torch.empty(0, 3, self.size[0], self.size[1]), torch.empty(0, 3, self.size[0], self.size[1])
        
        return torch.stack(tensor_gt_imgs), torch.stack(tensor_gen_imgs)

    def calculate_score(self, batch, batch_size=8, update=True):
        gt_images = batch['gt_im']
        gen_images = batch['gen_im']
        
        if len(gt_images) != len(gen_images):
            raise ValueError(f"Number of reference images ({len(gt_images)}) does not match number of generated images ({len(gen_images)})")
        
        # Check if image lists are empty
        if not gt_images or not gen_images:
            print("Image lists are empty")
            return float("nan"), []

        # Create DataLoader with custom collate function
        data_loader = DataLoader(
            list(zip(gt_images, gen_images)), 
            batch_size=batch_size, 
            collate_fn=self.collate_fn, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        values = []
        for tensor_gt_batch, tensor_gen_batch in tqdm(data_loader, desc="Calculating LPIPS distance"):
            # Check if there are valid image pairs
            if tensor_gt_batch.size(0) == 0:
                continue
                
            # Ensure both batches have the same shape
            if tensor_gt_batch.shape != tensor_gen_batch.shape:
                print(f"Warning: Batch shapes don't match: {tensor_gt_batch.shape} vs {tensor_gen_batch.shape}")
                # Resize to match shapes
                h, w = max(tensor_gt_batch.shape[2], tensor_gen_batch.shape[2]), max(tensor_gt_batch.shape[3], tensor_gen_batch.shape[3])
                tensor_gt_batch = torch.nn.functional.interpolate(tensor_gt_batch, size=(h, w), mode='bilinear', align_corners=False)
                tensor_gen_batch = torch.nn.functional.interpolate(tensor_gen_batch, size=(h, w), mode='bilinear', align_corners=False)
                
            # Compute LPIPS
            lpips_values = self.LPIPS(tensor_gt_batch, tensor_gen_batch)
            
            # Handle different return types
            if lpips_values.numel() == 1:
                values.append(lpips_values.squeeze().cpu().detach().item())
            else:
                values.extend(lpips_values.squeeze().cpu().detach().tolist())

        if not values:
            print("No valid LPIPS calculation results found.")
            return float("nan"), []

        avg_score = sum(values) / len(values)
        if update:
            self.meter.update(avg_score, len(values))
            return self.meter.avg, values
        else:
            return avg_score, values

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
            # svg=svg_string
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
    try:
        calculator = LPIPSDistanceCalculator()
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
        print(f"Average LPIPS score: {avg_score}")
        print(f"Detailed scores: {all_scores}")
    except Exception as e:
        print(f"Program execution error: {e}")