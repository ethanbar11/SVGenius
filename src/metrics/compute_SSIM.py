from .base_metric import BaseMetric
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image
import cairosvg
from io import BytesIO
from .data_util import clean_svg

class SSIMDistanceCalculator(BaseMetric): 
    def __init__(self, config=None):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config
        self.metric = self.compute_SSIM
    
    def compute_SSIM(self, **kwargs):
        image1 = kwargs.get('gt_im')
        image2 = kwargs.get('gen_im')
        win_size = kwargs.get('win_size', 11)  # Increase win_size for more accuracy
        channel_axis = kwargs.get('channel_axis', -1)  # Default channel_axis to -1
        sigma = kwargs.get('sigma', 1.5)  # Add sigma parameter for Gaussian filter

        # Check if images exist
        if image1 is None or image2 is None:
            raise ValueError("Reference image or generated image is missing")
        
        # Check image types and convert to numpy arrays
        img1_np = self._ensure_numpy_array(image1)
        img2_np = self._ensure_numpy_array(image2)
        
        # Ensure image dimensions are consistent
        if img1_np.shape != img2_np.shape:
            # Resize second image to match first image's dimensions
            img2_np = self._resize_to_match(img2_np, img1_np.shape)
        
        # Check if image is grayscale or RGB
        if len(img1_np.shape) == 3 and img1_np.shape[2] == 3:
            # RGB image
            score, _ = ssim(img1_np, img2_np, win_size=win_size, channel_axis=channel_axis, 
                           sigma=sigma, full=True, data_range=255)
        else:
            # If not grayscale image, convert to grayscale
            if len(img1_np.shape) == 3:
                img1_np = np.mean(img1_np, axis=2).astype(np.uint8)
                img2_np = np.mean(img2_np, axis=2).astype(np.uint8)
            
            score, _ = ssim(img1_np, img2_np, win_size=win_size, 
                           sigma=sigma, full=True, data_range=255)

        return score
    
    def _ensure_numpy_array(self, image):
        """Ensure input is a numpy array and handle various possible input types"""
        if isinstance(image, np.ndarray):
            # Already a numpy array
            return image
        elif isinstance(image, Image.Image):
            # PIL image
            return np.array(image)
        elif isinstance(image, str):
            # Image path
            try:
                return np.array(Image.open(image))
            except Exception as e:
                raise ValueError(f"Cannot open image file {image}: {e}")
        elif hasattr(image, 'cpu') and hasattr(image, 'numpy'):
            # PyTorch tensor
            img_np = image.cpu().numpy()
            # Handle different tensor shapes
            if img_np.shape[0] == 3:  # If in (C,H,W) format
                img_np = np.transpose(img_np, (1, 2, 0))
            return img_np
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _resize_to_match(self, img, target_shape):
        """Resize image to match target shape"""
        # If 3D array (with color channels)
        if len(img.shape) == 3 and len(target_shape) == 3:
            from PIL import Image
            pil_img = Image.fromarray(img)
            target_height, target_width = target_shape[0], target_shape[1]
            resized_img = pil_img.resize((target_width, target_height), Image.BICUBIC)
            return np.array(resized_img)
        # If 2D array (grayscale image)
        elif len(img.shape) == 2 and len(target_shape) == 2:
            from PIL import Image
            pil_img = Image.fromarray(img)
            target_height, target_width = target_shape
            resized_img = pil_img.resize((target_width, target_height), Image.BICUBIC)
            return np.array(resized_img)
        else:
            raise ValueError(f"Cannot resize image with shape {img.shape} to target shape {target_shape}")

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
    calculator = SSIMDistanceCalculator()
    # Example 1: Calculate score for individual images
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
    print(f"Average SSIM score: {avg_score}")