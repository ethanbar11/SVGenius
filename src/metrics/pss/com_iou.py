import numpy as np
from PIL import Image
import cairosvg
import io
import xml.etree.ElementTree as ET
import re
from skimage.measure import label, regionprops

def svg_to_alpha_image(svg_string, width=512, height=512):
    """
    Converts an SVG to a PNG image with an alpha channel to focus on the actual drawn pixels.
    
    Args:
    svg_string: The SVG string.
    width, height: Dimensions of the output image.
    
    Returns:
    PIL.Image: The image with an alpha channel.
    """
    # Use cairosvg to convert the SVG to a PNG with an alpha channel
    png_data = cairosvg.svg2png(
        bytestring=svg_string.encode('utf-8'),
        output_width=width,
        output_height=height,
        background_color="transparent"  # Set transparent background
    )
    
    # Create a PIL image from the byte stream (with alpha channel)
    image = Image.open(io.BytesIO(png_data)).convert("RGBA")
    return image

def calculate_alpha_mask_iou(svg1, svg2, width=512, height=512, output_path=None):
    """
    Calculates the Intersection over Union (IoU) for two SVGs, considering only non-transparent pixels.
    
    Args:
    svg1, svg2: The SVG strings.
    width, height: Dimensions of the output image.
    
    Returns:
    float: The IoU value.
    """
    # Convert SVGs to images with alpha channel
    img1 = svg_to_alpha_image(svg1, width, height)
    img2 = svg_to_alpha_image(svg2, width, height)
    
    # Extract the alpha channel as a mask (any pixel with a value greater than 0 is considered drawn)
    mask1 = np.array(img1)[:,:,3] > 0
    mask2 = np.array(img2)[:,:,3] > 0
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Calculate IoU
    if union == 0:
        return 0.0
    iou = intersection / union
    vis_img, comp_iou = visualize_path_comparison(svg1, svg2)
    vis_img.save(output_path)
    print(f"Comparison image saved, overall visual IoU: {iou:.4f}")

    return iou

def visualize_path_comparison(svg1, svg2, width=512, height=512):
    """
    Visualizes the comparison of paths between two SVGs.
    """
    # Convert the SVGs to images with alpha channel
    img1 = svg_to_alpha_image(svg1, width, height)
    img2 = svg_to_alpha_image(svg2, width, height)
    
    # Extract the alpha masks
    mask1 = np.array(img1)[:,:,3] > 0
    mask2 = np.array(img2)[:,:,3] > 0
    
    # Create comparison image
    comparison = np.zeros((height, width, 3), dtype=np.uint8)
    comparison[mask1 & ~mask2] = [255, 0, 0]    # Red - only in SVG1
    comparison[~mask1 & mask2] = [0, 255, 0]    # Green - only in SVG2
    comparison[mask1 & mask2] = [255, 255, 0]   # Yellow - intersection
    
    # Calculate IoU
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0
    
    # Create PIL image and add text description
    result_img = Image.fromarray(comparison)
    return result_img, iou


# Example usage
if __name__ == "__main__":
    with open('example1.svg', 'r') as f:
        svg1 = f.read()
    with open('example2.svg', 'r') as f:
        svg2 = f.read()
    
    # Calculate alpha mask IoU
    alpha_iou = calculate_alpha_mask_iou(svg1, svg2)
    print(f"Alpha mask IoU: {alpha_iou:.4f}")
    
    # Visualize comparison
    vis_img, comp_iou = visualize_path_comparison(svg1, svg2)
    vis_img.save('./svg_alpha_comparison.png')
    print(f"Comparison image saved, visual IoU: {comp_iou:.4f}")
