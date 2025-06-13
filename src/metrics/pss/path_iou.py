import numpy as np
from PIL import Image
import cairosvg
import io
import xml.etree.ElementTree as ET
import re
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import time
from normal import SVGScaler

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

def extract_svg_paths(svg_string):
    """
    Extract all path elements and their attributes from SVG string
    
    Args:
        svg_string (str): SVG string to parse
    
    Returns:
        tuple: (paths_list, viewBox) where paths_list contains path elements 
               with d attribute and style attributes, viewBox is the SVG viewBox
    """
    # Add namespace if missing
    if 'xmlns' not in svg_string:
        svg_string = svg_string.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
    
    # Parse XML
    try:
        root = ET.fromstring(svg_string)
    except ET.ParseError:
        # Try to fix XML issues by escaping unescaped ampersands
        svg_string = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', svg_string)
        root = ET.fromstring(svg_string)
    
    # Extract viewBox attribute
    viewBox = root.get('viewBox', '0 0 1024 1024')
    
    # Find all path elements
    paths = []
    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        d = path.get('d', '')
        fill = path.get('fill', 'black')
        stroke = path.get('stroke', 'none')
        stroke_width = path.get('stroke-width', '1')
        opacity = path.get('opacity', '1')
        paths.append({
            'd': d,
            'fill': fill,
            'stroke': stroke,
            'stroke-width': stroke_width,
            'opacity': opacity
        })
    
    return paths, viewBox

def path_to_alpha_image(path_data, width=512, height=512, viewBox="0 0 1024 1024"):
    """
    Convert a single SVG path to an image with alpha channel
    
    Args:
        path_data (dict): Dictionary containing path attributes (d, fill, stroke, etc.)
        width (int): Output image width
        height (int): Output image height
        viewBox (str): SVG viewBox attribute
    
    Returns:
        PIL.Image: RGBA image with path rendered
    """
    # Create SVG containing only this path
    svg_template = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewBox}" width="{width}" height="{height}">
        <path d="{path_data['d']}" 
              fill="{path_data['fill']}" 
              stroke="{path_data['stroke']}" 
              stroke-width="{path_data['stroke-width']}"
              opacity="{path_data['opacity']}"/>
    </svg>'''
    
    # Convert to PNG and read as image with alpha channel
    png_data = cairosvg.svg2png(
        bytestring=svg_template.encode('utf-8'),
        output_width=width,
        output_height=height,
        background_color="transparent"
    )
    
    image = Image.open(io.BytesIO(png_data)).convert("RGBA")
    return image

def calculate_path_iou(path1, path2, width=512, height=512, viewBox="0 0 1024 1024"):
    """
    Calculate Intersection over Union (IoU) between two SVG paths using alpha channel
    to determine drawn regions
    
    Args:
        path1 (dict): First path data
        path2 (dict): Second path data
        width (int): Image width for comparison
        height (int): Image height for comparison
        viewBox (str): SVG viewBox attribute
    
    Returns:
        float: IoU score between 0 and 1
    """
    # Convert paths to images
    img1 = path_to_alpha_image(path1, width, height, viewBox)
    img2 = path_to_alpha_image(path2, width, height, viewBox)
    
    # Extract alpha channel (focus on whether anything is drawn, no binarization)
    alpha1 = np.array(img1)[:,:,3] > 0  # Any non-zero alpha value indicates drawing
    alpha2 = np.array(img2)[:,:,3] > 0
    
    # Calculate intersection and union
    intersection = np.logical_and(alpha1, alpha2).sum()
    union = np.logical_or(alpha1, alpha2).sum()
    
    # Calculate IoU
    if union == 0:  # Avoid division by zero
        return 0.0
    return intersection / union

def calculate_path_area(path, width=512, height=512, viewBox="0 0 1024 1024"):
    """
    Calculate area of SVG path (number of non-zero alpha pixels)
    
    Args:
        path (dict): Path data
        width (int): Image width
        height (int): Image height
        viewBox (str): SVG viewBox attribute
    
    Returns:
        int: Number of pixels covered by the path
    """
    img = path_to_alpha_image(path, width, height, viewBox)
    alpha = np.array(img)[:,:,3] > 0
    return alpha.sum()

def hungarian_path_matching(svg1, svg2, width=512, height=512):
    """
    Use Hungarian algorithm to find optimal path matching between two SVGs,
    calculate weighted IoU
    
    Algorithm steps:
    1. Extract all paths from both SVGs
    2. Calculate IoU between all path pairs to build cost matrix
    3. Use Hungarian algorithm to find optimal matching
    4. Calculate weighted average IoU as overall score
    
    Args:
        svg1 (str): First SVG string
        svg2 (str): Second SVG string
        width (int): Image width for comparison
        height (int): Image height for comparison
    
    Returns:
        tuple: (weighted_iou, matches, match_ious)
               - weighted_iou: Overall weighted IoU score
               - matches: List of (path1_idx, path2_idx) tuples
               - match_ious: List of IoU scores for each match
    """
    # Extract paths
    paths1, viewBox1 = extract_svg_paths(svg1)
    paths2, viewBox2 = extract_svg_paths(svg2)
    
    # If either SVG has no paths, return 0
    if not paths1 or not paths2:
        return 0.0, [], []
    
    # Calculate area for each path
    areas1 = [calculate_path_area(path, width, height, viewBox1) for path in paths1]
    areas2 = [calculate_path_area(path, width, height, viewBox2) for path in paths2]
    
    # Calculate IoU between all path pairs to build cost matrix
    # Note: Hungarian algorithm finds minimum cost matching, so we use 1-IoU as cost
    cost_matrix = np.zeros((len(paths1), len(paths2)))
    iou_matrix = np.zeros((len(paths1), len(paths2)))
    
    for i, path1 in enumerate(paths1):
        for j, path2 in enumerate(paths2):
            iou = calculate_path_iou(path1, path2, width, height, viewBox1)
            iou_matrix[i, j] = iou
            cost_matrix[i, j] = 1.0 - iou
    
    # To handle different numbers of paths, create a square cost matrix
    # Add dummy paths that have IoU of 0 with any real path
    n = max(len(paths1), len(paths2))
    square_cost_matrix = np.ones((n, n))
    square_cost_matrix[:len(paths1), :len(paths2)] = cost_matrix
    
    # Use Hungarian algorithm to find optimal matching
    row_ind, col_ind = linear_sum_assignment(square_cost_matrix)
    
    # Collect valid matches and corresponding IoUs
    matches = []
    match_ious = []
    
    for i, j in zip(row_ind, col_ind):
        if i < len(paths1) and j < len(paths2):
            matches.append((i, j))
            match_ious.append(iou_matrix[i, j])
    
    # Calculate weighted IoU
    total_area1 = sum(areas1)
    weighted_iou = 0.0
    
    if total_area1 > 0:
        for (i, j), iou in zip(matches, match_ious):
            weight = areas1[i] / total_area1
            weighted_iou += weight * iou
    
    return weighted_iou, matches, match_ious

def visualize_path_matching(svg1, svg2, matches, match_ious, width=512, height=512, output_file='path_matching_visualization.png'):
    """
    Visualize path matching results
    
    Creates an image showing each pair of matched paths and their IoU
    
    Args:
        svg1 (str): First SVG string
        svg2 (str): Second SVG string
        matches (list): List of (path1_idx, path2_idx) tuples
        match_ious (list): List of IoU scores for each match
        width (int): Image width
        height (int): Image height
        output_file (str): Output file path for visualization
    """
    paths1, viewBox1 = extract_svg_paths(svg1)
    paths2, viewBox2 = extract_svg_paths(svg2)
    
    if not matches:
        print("No matches to visualize")
        return
    
    # Create a large image to display all matches
    fig, axes = plt.subplots(len(matches), 3, figsize=(15, 5*len(matches)))
    
    # If only one match, ensure axes is 2D
    if len(matches) == 1:
        axes = np.array([axes])
    
    for idx, ((i, j), iou) in enumerate(zip(matches, match_ious)):
        # Extract matched paths
        path1 = paths1[i]
        path2 = paths2[j]
        
        # Convert paths to images
        img1 = path_to_alpha_image(path1, width, height, viewBox1)
        img2 = path_to_alpha_image(path2, width, height, viewBox2)
        
        # Create comparison image
        alpha1 = np.array(img1)[:,:,3] > 0
        alpha2 = np.array(img2)[:,:,3] > 0
        
        comparison = np.zeros((height, width, 3), dtype=np.uint8)
        comparison[alpha1 & ~alpha2] = [255, 0, 0]    # Red - only in path1
        comparison[~alpha1 & alpha2] = [0, 255, 0]    # Green - only in path2
        comparison[alpha1 & alpha2] = [255, 255, 0]   # Yellow - intersection
        
        # Display images
        axes[idx, 0].imshow(img1)
        axes[idx, 0].set_title(f'SVG1 Path {i}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(img2)
        axes[idx, 1].set_title(f'SVG2 Path {j}')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(comparison)
        axes[idx, 2].set_title(f'Overlay (IoU: {iou:.4f})')
        axes[idx, 2].axis('off')
        
        # Add legend for the comparison image
        if idx == 0:  # Only add legend to the first row
            legend_elements = [
                Patch(facecolor='red', label='SVG1 only'),
                Patch(facecolor='green', label='SVG2 only'),
                Patch(facecolor='yellow', label='Intersection')
            ]
            axes[idx, 2].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Path matching visualization saved to: {output_file}")

def full_svg_comparison(svg1, svg2, width=512, height=512):
    """
    Perform comprehensive SVG comparison and return detailed results
    
    Args:
        svg1 (str): First SVG string
        svg2 (str): Second SVG string
        width (int): Image width for comparison
        height (int): Image height for comparison
    
    Returns:
        dict: Comprehensive comparison results including:
              - weighted_iou: Overall IoU score
              - paths_count1/2: Number of paths in each SVG
              - matches: List of matched path indices
              - match_ious: IoU scores for each match
              - path_areas1/2: Areas of paths in each SVG
              - area_percentages1/2: Relative area percentages
              - processing_time: Time taken for comparison
    """
    start_time = time.time()
    
    # Calculate Hungarian algorithm-based path matching IoU
    weighted_iou, matches, match_ious = hungarian_path_matching(svg1, svg2, width, height)
    
    # Extract paths and calculate areas
    paths1, viewBox1 = extract_svg_paths(svg1)
    paths2, viewBox2 = extract_svg_paths(svg2)
    areas1 = [calculate_path_area(path, width, height, viewBox1) for path in paths1]
    areas2 = [calculate_path_area(path, width, height, viewBox2) for path in paths2]
    
    # Calculate relative area percentages for each path
    total_area1 = sum(areas1)
    total_area2 = sum(areas2)
    
    area_percentages1 = [area/total_area1*100 if total_area1 > 0 else 0 for area in areas1]
    area_percentages2 = [area/total_area2*100 if total_area2 > 0 else 0 for area in areas2]
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    
    # Build comprehensive results
    result = {
        "weighted_iou": weighted_iou,
        "paths_count1": len(paths1),
        "paths_count2": len(paths2),
        "matches": matches,
        "match_ious": match_ious,
        "path_areas1": areas1,
        "path_areas2": areas2,
        "area_percentages1": area_percentages1,
        "area_percentages2": area_percentages2,
        "processing_time": elapsed_time
    }
    
    return result

def compare_svg_files(file_path1, file_path2, width=512, height=512, 
                     visualize=True, output_dir='./'):
    """
    Compare two SVG files and generate comprehensive analysis
    
    Args:
        file_path1 (str): Path to first SVG file
        file_path2 (str): Path to second SVG file
        width (int): Image width for comparison
        height (int): Image height for comparison
        visualize (bool): Whether to generate visualization
        output_dir (str): Directory to save output files
    
    Returns:
        dict: Comprehensive comparison results
    """
    # Read SVG files
    try:
        with open(file_path1, 'r', encoding='utf-8') as f:
            svg1 = f.read()
        with open(file_path2, 'r', encoding='utf-8') as f:
            svg2 = f.read()
    except FileNotFoundError as e:
        print(f"Error reading SVG files: {e}")
        return None
    
    # Perform comparison
    weighted_iou, matches, match_ious = hungarian_path_matching(svg1, svg2, width, height)
    
    # Generate visualization if requested
    if visualize and matches:
        vis_output = f"{output_dir}/svg_path_comparison.png"
        visualize_path_matching(svg1, svg2, matches, match_ious, width, height, vis_output)
    
    # Get comprehensive results
    result = full_svg_comparison(svg1, svg2, width, height)
    
    return result

# Example usage and testing
if __name__ == "__main__":
    # Initialize SVG scaler for preprocessing
    scaler = SVGScaler()
    
    # Example usage with file paths (update these paths as needed)
    svg_file1 = '/path/to/your/first.svg'
    svg_file2 = '/path/to/your/second.svg'
    
    # For demonstration, we'll use inline SVG strings
    svg1 = '''<svg viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">
        <path d="M200 300 L350 280 L400 150 L450 280 L600 300 L480 380 L520 520 L400 440 L280 520 L320 380 Z" 
              fill="gold" stroke="orange" stroke-width="5"/>
    </svg>'''
    
    svg2 = '''<svg viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">
        <path d="M100 100 L300 100 L300 300 L100 300 Z" fill="red"/>
        <path d="M350 100 L550 100 L550 300 L350 300 Z" fill="green"/>
        <path d="M600 100 L800 100 L800 300 L600 300 Z" fill="blue"/>
    </svg>'''
    
    print("=== SVG Comparison and Scaling Example ===\n")
    
    # First, normalize and scale SVGs using SVGScaler
    print("Step 1: Normalizing and scaling SVGs...")
    scaling_result = scaler.compare_and_scale_svg(
        svg1, 
        svg2, 
        preserve_aspect_ratio=True,
        output_file='svg_scaling_comparison.png',
        show_plot=False  # Set to True if you want to display the plot
    )
    
    print("Scaling Results:")
    print(f"  Width ratio: {scaling_result['ratios']['widthRatio']:.4f}")
    print(f"  Height ratio: {scaling_result['ratios']['heightRatio']:.4f}")
    print(f"  Area ratio: {scaling_result['ratios']['areaRatio']:.4f}")
    
    # Step 2: Compare normalized SVGs using Hungarian algorithm
    print("\nStep 2: Performing path matching with Hungarian algorithm...")
    normalized_svg1 = scaling_result['normalizedAnswerSvg']
    normalized_svg2 = scaling_result['finalSvg']
    
    # Execute Hungarian algorithm path matching
    weighted_iou, matches, match_ious = hungarian_path_matching(normalized_svg1, normalized_svg2)
    
    print("Path Matching Results:")
    print(f"  Weighted IoU Score: {weighted_iou:.4f}")
    print(f"  Number of matches: {len(matches)}")
    print(f"  Matched path pairs: {matches}")
    print(f"  Individual IoU scores: {[f'{iou:.4f}' for iou in match_ious]}")
    
    # Visualize matching results
    if matches:
        print("\nStep 3: Generating visualization...")
        visualize_path_matching(normalized_svg1, normalized_svg2, matches, match_ious, 
                              output_file='path_matching_analysis.png')
    
    # Perform comprehensive comparison
    print("\nStep 4: Comprehensive analysis...")
    comprehensive_result = full_svg_comparison(normalized_svg1, normalized_svg2)
    
    print("Comprehensive Comparison Results:")
    print(f"  SVG1 path count: {comprehensive_result['paths_count1']}")
    print(f"  SVG2 path count: {comprehensive_result['paths_count2']}")
    print(f"  Processing time: {comprehensive_result['processing_time']:.4f} seconds")
    print(f"  Total area coverage (SVG1): {sum(comprehensive_result['path_areas1'])} pixels")
    print(f"  Total area coverage (SVG2): {sum(comprehensive_result['path_areas2'])} pixels")
    
    # Display area percentages for each path
    if comprehensive_result['area_percentages1']:
        print("  Area distribution in SVG1:")
        for i, pct in enumerate(comprehensive_result['area_percentages1']):
            print(f"    Path {i}: {pct:.2f}%")
    
    if comprehensive_result['area_percentages2']:
        print("  Area distribution in SVG2:")
        for i, pct in enumerate(comprehensive_result['area_percentages2']):
            print(f"    Path {i}: {pct:.2f}%")
    
    print(f"\nAnalysis complete! Check the generated visualization files.")