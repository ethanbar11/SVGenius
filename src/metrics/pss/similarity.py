import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import math
from skimage.metrics import structural_similarity as ssim
import colorsys
import svgpathtools as spt
import xml.etree.ElementTree as ET
import re
from io import BytesIO
from PIL import Image, ImageDraw
import cairosvg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib import gridspec
import traceback
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image, ImageDraw
import os

class SVGComparisonSystem:
    """
    Comprehensive SVG comparison system using Hungarian algorithm for optimal path matching
    
    This system compares two SVGs by:
    1. Extracting individual paths from each SVG
    2. Computing multi-dimensional similarity matrices (shape, color, position)
    3. Using Hungarian algorithm to find optimal path matching
    4. Generating detailed feedback and visualizations
    """
    
    def __init__(self, weights=None):
        """
        Initialize comparison system with scoring dimension weights
        
        Parameters:
        ----------
        weights : dict, optional
            Weights for different scoring dimensions. Default weights:
            - shape: 0.4 (shape accuracy via IoU)
            - color: 0.2 (color accuracy)
            - position: 0.4 (scale and position similarity)
        """
        self.weights = weights or {
            'shape': 0.4,     # Shape accuracy
            'color': 0.2,     # Color accuracy
            'position': 0.4,  # Scale and position
        }
        self.similarity_threshold = 0.3  # Minimum threshold for valid matches
    
    def calculate_iou(self, path1_image, path2_image):
        """
        Calculate Intersection over Union (IoU) between two path masks
        
        Parameters:
        ----------
        path1_image : PIL.Image
            First path image with alpha channel
        path2_image : PIL.Image
            Second path image with alpha channel
            
        Returns:
        -------
        float
            IoU score between 0 and 1
        """
        # Extract alpha channels to create binary masks
        path1_mask = np.array(path1_image)[:,:,3] > 0  # Any non-zero alpha indicates drawing
        path2_mask = np.array(path2_image)[:,:,3] > 0 
        
        # Calculate intersection and union
        intersection = np.logical_and(path1_mask, path2_mask).sum()
        union = np.logical_or(path1_mask, path2_mask).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_color_similarity(self, color1, color2):
        """
        Calculate color similarity between two RGB colors using perceptual color space
        
        Uses HSV color space to better reflect human visual perception, with dynamic
        weighting based on saturation levels.
        
        Parameters:
        ----------
        color1 : tuple
            First color RGB values (0-1 range)
        color2 : tuple
            Second color RGB values (0-1 range)
            
        Returns:
        -------
        float
            Color similarity score (0-1)
        """
        try:
            # Ensure valid color values
            if color1 is None:
                color1 = (0, 0, 0)  # Default black
            if color2 is None:
                color2 = (0, 0, 0)  # Default black
                
            # Ensure colors are valid RGB tuples
            if not isinstance(color1, tuple) or len(color1) < 3:
                color1 = (0, 0, 0)
            if not isinstance(color2, tuple) or len(color2) < 3:
                color2 = (0, 0, 0)
                
            # Handle transparency
            is_transparent1 = len(color1) > 3 and color1[3] == 0
            is_transparent2 = len(color2) > 3 and color2[3] == 0
            
            if is_transparent1 and is_transparent2:
                return 1.0  # Both transparent, perfect match
            
            if is_transparent1 or is_transparent2:
                return 0.0  # One transparent, one opaque, no match
            
            # Convert to HSV space for better perceptual comparison
            def rgb_to_hsv(rgb):
                r, g, b = rgb[:3]  # Take only RGB components
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                return (h, s, v)
            
            hsv1 = rgb_to_hsv(color1)
            hsv2 = rgb_to_hsv(color2)
            
            # Calculate hue difference (circular)
            h_weight = min(hsv1[1], hsv2[1])  # Use minimum saturation as hue weight
            h_diff = min(abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0]))
            
            # Calculate saturation and value differences
            s_diff = abs(hsv1[1] - hsv2[1])
            v_diff = abs(hsv1[2] - hsv2[2])
            
            # Dynamic weight allocation
            weighted_diff = (0.6 * h_weight * h_diff) + (0.2 * s_diff) + (0.2 * v_diff)
            
            # For grayscale colors (low saturation), increase brightness weight
            if hsv1[1] < 0.2 and hsv2[1] < 0.2:
                weighted_diff = 0.1 * h_diff + 0.1 * s_diff + 0.8 * v_diff
            
            return 1 - weighted_diff
            
        except Exception as e:
            print(f"Error calculating color similarity: {e}")
            return 0.5  # Return neutral score on error
    
    def calculate_position_similarity(self, bbox1, bbox2, img_size):
        """
        Calculate position and scale similarity between two path bounding boxes
        
        Parameters:
        ----------
        bbox1 : tuple
            First path bounding box (x, y, width, height)
        bbox2 : tuple
            Second path bounding box (x, y, width, height)
        img_size : tuple
            Image dimensions (width, height)
            
        Returns:
        -------
        float
            Position similarity score (0-1)
        """
        # Normalize bounding box coordinates
        img_width, img_height = img_size
        
        def normalize_bbox(bbox):
            x, y, w, h = bbox
            return (x/img_width, y/img_height, w/img_width, h/img_height)
        
        nbbox1 = normalize_bbox(bbox1)
        nbbox2 = normalize_bbox(bbox2)
        
        # Calculate center points
        center1 = (nbbox1[0] + nbbox1[2]/2, nbbox1[1] + nbbox1[3]/2)
        center2 = (nbbox2[0] + nbbox2[2]/2, nbbox2[1] + nbbox2[3]/2)
        
        # Calculate center distance
        center_dist = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Calculate size difference
        size_diff = abs(nbbox1[2]*nbbox1[3] - nbbox2[2]*nbbox2[3])
        
        # Position similarity
        position_sim = 1 - min(1, center_dist * 2)  # Distance > 0.5 considered completely different
        
        # Size similarity
        size_sim = 1 - min(1, size_diff * 3)  # Size difference > 0.33 considered completely different
        
        # Combine position and size
        similarity = 0.6 * position_sim + 0.4 * size_sim
        
        return similarity
    
    def build_similarity_matrices(self, reference_paths, generated_paths, img_size):
        """
        Build similarity matrices for all dimensions
        
        Parameters:
        ----------
        reference_paths : list
            Reference SVG path information list
        generated_paths : list
            Generated SVG path information list
        img_size : tuple
            Image dimensions (width, height)
            
        Returns:
        -------
        dict
            Dictionary containing similarity matrices for each dimension and combined matrix
        """
        n_ref = len(reference_paths)
        n_gen = len(generated_paths)
        
        # Initialize similarity matrices for each dimension
        shape_matrix = np.zeros((n_ref, n_gen))
        color_matrix = np.zeros((n_ref, n_gen))
        position_matrix = np.zeros((n_ref, n_gen))
        
        # Calculate similarities for each dimension
        for i, ref_path in enumerate(reference_paths):
            for j, gen_path in enumerate(generated_paths):
                # Calculate shape similarity using IoU
                shape_matrix[i, j] = self.calculate_iou(ref_path['mask'], gen_path['mask'])
                
                # Calculate color similarity
                color_matrix[i, j] = self.calculate_color_similarity(ref_path['color'], gen_path['color'])
                
                # Calculate position and scale similarity
                position_matrix[i, j] = self.calculate_position_similarity(
                    ref_path['bbox'], gen_path['bbox'], img_size)
        
        # Calculate combined similarity matrix using weighted sum
        combined_matrix = (
            self.weights['shape'] * shape_matrix +
            self.weights['color'] * color_matrix +
            self.weights['position'] * position_matrix
        )
        
        return {
            'shape': shape_matrix,
            'color': color_matrix,
            'position': position_matrix,
            'combined': combined_matrix
        }
    
    def hungarian_matching(self, similarity_matrix):
        """
        Use Hungarian algorithm to match paths optimally
        
        Parameters:
        ----------
        similarity_matrix : numpy.ndarray
            Combined similarity matrix
            
        Returns:
        -------
        tuple
            Matching results (row indices, column indices) and corresponding scores
        """
        # Hungarian algorithm solves maximization assignment problem
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)  # Convert to minimization problem
        
        # Get matching scores
        scores = similarity_matrix[row_ind, col_ind]
        
        return (row_ind, col_ind), scores
    
    def evaluate_matching(self, matches, scores):
        """
        Evaluate matching quality and calculate final scores
        
        Parameters:
        ----------
        matches : tuple
            Matching results (row indices, column indices)
        scores : numpy.ndarray
            Score for each matched pair
            
        Returns:
        -------
        dict
            Evaluation results including metrics and final scores
        """
        row_ind, col_ind = matches
        
        # Calculate valid matches (above threshold)
        valid_matches = scores >= self.similarity_threshold
        valid_row_ind = row_ind[valid_matches]
        valid_col_ind = col_ind[valid_matches]
        valid_scores = scores[valid_matches]
        
        # Calculate average score
        avg_score = valid_scores.mean() if len(valid_scores) > 0 else 0
        
        # Calculate match rate (valid matches / total paths)
        match_rate = len(valid_scores) / len(row_ind) if len(row_ind) > 0 else 0
        
        # Calculate final weighted score (average score * match rate)
        final_score = avg_score * match_rate
        
        return {
            'matches': list(zip(valid_row_ind.tolist(), valid_col_ind.tolist())),
            'scores': valid_scores.tolist(),
            'avg_score': float(avg_score),
            'match_rate': float(match_rate),
            'final_score': float(final_score),
            'overall_quality_score': int(final_score * 100)  # 0-100 integer score
        }
    
    def generate_visual_feedback(self, reference_svg, generated_svg, matches, dimension_matrices):
        """
        Generate visual feedback highlighting differences
        
        Parameters:
        ----------
        reference_svg : dict
            Reference SVG information
        generated_svg : dict
            Generated SVG information
        matches : list
            Valid match list [(ref_idx, gen_idx), ...]
        dimension_matrices : dict
            Similarity matrices for each dimension
            
        Returns:
        -------
        list
            Detailed feedback information for each matched path
        """
        feedback = []
        
        for ref_idx, gen_idx in matches:
            match_feedback = {
                'ref_path_id': reference_svg['paths'][ref_idx].get('id', f'path_{ref_idx}'),
                'gen_path_id': generated_svg['paths'][gen_idx].get('id', f'path_{gen_idx}'),
                'shape_score': dimension_matrices['shape'][ref_idx, gen_idx],
                'color_score': dimension_matrices['color'][ref_idx, gen_idx],
                'position_score': dimension_matrices['position'][ref_idx, gen_idx],
                'suggestions': []
            }
            
            # Generate specific suggestions based on dimension scores
            if match_feedback['shape_score'] < 0.7:
                match_feedback['suggestions'].append("Shape needs adjustment - contour not precise enough")
                
            if match_feedback['color_score'] < 0.7:
                match_feedback['suggestions'].append("Color not accurate - adjust hue or saturation")
                
            if match_feedback['position_score'] < 0.7:
                match_feedback['suggestions'].append("Position or scale incorrect - check element size and placement")
            
            feedback.append(match_feedback)
        
        return feedback
    
    def compare_svgs(self, reference_paths, generated_paths, img_size):
        """
        Compare two SVGs and generate comprehensive evaluation
        
        Parameters:
        ----------
        reference_paths : list
            Reference SVG path information list
        generated_paths : list
            Generated SVG path information list
        img_size : tuple
            Image dimensions (width, height)
            
        Returns:
        -------
        dict
            Comprehensive evaluation results
        """
        # Build similarity matrices for each dimension
        similarity_matrices = self.build_similarity_matrices(
            reference_paths, generated_paths, img_size)
        
        # Use Hungarian algorithm for matching
        matches, scores = self.hungarian_matching(similarity_matrices['combined'])
        
        # Evaluate matching quality
        evaluation = self.evaluate_matching(matches, scores)
        
        # Generate visual feedback
        feedback = self.generate_visual_feedback(
            {'paths': reference_paths}, 
            {'paths': generated_paths}, 
            evaluation['matches'],
            similarity_matrices
        )
        
        # Calculate average scores for each dimension
        dimension_scores = {}
        if evaluation['matches']:
            dimension_scores = {
                'shape': float(np.mean([similarity_matrices['shape'][i, j] for i, j in evaluation['matches']])),
                'color': float(np.mean([similarity_matrices['color'][i, j] for i, j in evaluation['matches']])),
                'position': float(np.mean([similarity_matrices['position'][i, j] for i, j in evaluation['matches']])),
            }
        else:
            dimension_scores = {
                'shape': 0.0,
                'color': 0.0,
                'position': 0.0,
            }
        
        # Combine final results
        result = {
            **evaluation,
            'feedback': feedback,
            'dimension_scores': dimension_scores
        }
        
        return result

def get_path_bounds(path):
    """
    Get SVG path bounding box using svgpathtools
    
    Parameters:
    ----------
    path : svgpathtools.Path
        SVG path object
            
    Returns:
    -------
    dict
        Dictionary containing bounding box information {minX, minY, maxX, maxY, width, height}
    """
    try:
        if not path:
            return {
                'minX': 0, 'minY': 0, 'maxX': 0, 'maxY': 0, 'width': 0, 'height': 0
            }
        
        # Get bounding box (using real and imaginary parts for x and y coordinates)
        x_values = []
        y_values = []
        
        # Traverse all segments in the path
        for segment in path:
            # Get segment endpoints and control points
            if isinstance(segment, spt.Line):
                points = [segment.start, segment.end]
            elif isinstance(segment, spt.CubicBezier):
                points = [segment.start, segment.control1, segment.control2, segment.end]
            elif isinstance(segment, spt.QuadraticBezier):
                points = [segment.start, segment.control, segment.end]
            elif isinstance(segment, spt.Arc):
                # For arcs, consider at least start and end points
                points = [segment.start, segment.end]
                # Add more points to approximate the arc
                for t in np.linspace(0, 1, 10):
                    try:
                        points.append(segment.point(t))
                    except:
                        pass  # Skip if point calculation fails
            else:
                # Handle other segment types
                points = [segment.start, segment.end]
            
            # Extract coordinates from all points
            for point in points:
                x_values.append(point.real)
                y_values.append(point.imag)
        
        # Calculate bounding box
        if not x_values or not y_values:
            return {
                'minX': 0, 'minY': 0, 'maxX': 0, 'maxY': 0, 'width': 0, 'height': 0
            }
            
        min_x = min(x_values)
        min_y = min(y_values)
        max_x = max(x_values)
        max_y = max(y_values)
        width = max_x - min_x
        height = max_y - min_y
        
        return {
            'minX': min_x,
            'minY': min_y,
            'maxX': max_x,
            'maxY': max_y,
            'width': width,
            'height': height
        }
    except Exception as e:
        print(f"Error calculating path bounds: {e}")
        return {
            'minX': 0, 'minY': 0, 'maxX': 0, 'maxY': 0, 'width': 0, 'height': 0
        }

def parse_color(color_str):
    """
    Parse SVG color string to RGB tuple
    
    Supports various color formats:
    - Hexadecimal: #RGB, #RRGGBB
    - RGB functions: rgb(r,g,b), rgba(r,g,b,a)
    - Named colors: red, blue, etc.
    
    Parameters:
    ----------
    color_str : str
        Color string
        
    Returns:
    -------
    tuple
        RGB color tuple (0-1 range)
    """
    if color_str == '':
        return (0, 0, 0)  # Black
    
    # Handle none or transparent
    if color_str is None or color_str.lower() in ['none', 'transparent']:
        return (0, 0, 0, 0)  # Transparent
    
    # Handle hexadecimal colors
    if color_str.startswith('#'):
        if len(color_str) == 4:  # #RGB format
            r = int(color_str[1] + color_str[1], 16) / 255
            g = int(color_str[2] + color_str[2], 16) / 255
            b = int(color_str[3] + color_str[3], 16) / 255
        else:  # #RRGGBB format
            r = int(color_str[1:3], 16) / 255
            g = int(color_str[3:5], 16) / 255
            b = int(color_str[5:7], 16) / 255
        return (r, g, b)
    
    # Handle rgb() format
    rgb_match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_str)
    if rgb_match:
        r = int(rgb_match.group(1)) / 255
        g = int(rgb_match.group(2)) / 255
        b = int(rgb_match.group(3)) / 255
        return (r, g, b)
    
    # Handle rgba() format
    rgba_match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)', color_str)
    if rgba_match:
        r = int(rgba_match.group(1)) / 255
        g = int(rgba_match.group(2)) / 255
        b = int(rgba_match.group(3)) / 255
        return (r, g, b)
    
    # Common named colors
    color_map = {
        'black': (0, 0, 0), 'white': (1, 1, 1), 'red': (1, 0, 0), 'green': (0, 1, 0),
        'blue': (0, 0, 1), 'yellow': (1, 1, 0), 'cyan': (0, 1, 1), 'magenta': (1, 0, 1),
        'gray': (0.5, 0.5, 0.5), 'silver': (0.75, 0.75, 0.75), 'maroon': (0.5, 0, 0),
        'olive': (0.5, 0.5, 0), 'lime': (0, 1, 0), 'aqua': (0, 1, 1), 'teal': (0, 0.5, 0.5),
        'navy': (0, 0, 0.5), 'fuchsia': (1, 0, 1), 'purple': (0.5, 0, 0.5)
    }
    
    return color_map.get(color_str.lower(), (0, 0, 0))

def render_path_to_image(path, attributes, svg_width, svg_height, target_size=(512, 512), viewBox="0 0 1024 1024"):
    """
    Render SVG path to image with alpha channel
    
    Parameters:
    ----------
    path : svgpathtools.Path
        SVG path object
    attributes : dict
        Path attributes including fill, stroke, etc.
    svg_width : float
        SVG width
    svg_height : float
        SVG height
    target_size : tuple
        Target image dimensions
    viewBox : str
        SVG viewBox attribute
    
    Returns:
    -------
    tuple
        (image, mask) - both are RGBA PIL Images
    """
    try:
        # Get path data and attributes
        path_d = path.d()
        fill = attributes.get('fill', '#000000')  # Default to black

        # Construct SVG string
        svg_str = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewBox}" width="{svg_width}" height="{svg_height}">
            <path d="{path_d}" fill="{fill}" />
        </svg>'''

        # Render SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(
            bytestring=svg_str.encode('utf-8'),
            output_width=target_size[0],
            output_height=target_size[1],
            background_color="transparent"  # Preserve transparency
        )

        # Convert to PIL RGBA images
        image = Image.open(BytesIO(png_data)).convert("RGBA")
        mask = Image.open(BytesIO(png_data)).convert("RGBA")

        return image, mask
        
    except Exception as e:
        print(f"Error rendering path to image: {e}")
        # Return empty images as fallback
        try:
            empty_image = Image.new("RGBA", target_size, (0, 0, 0, 0))
            return empty_image, empty_image
        except:
            return None, None

def extract_svg_data_from_string(svg_string, target_size=(256, 256)):
    """
    Efficiently extract data from SVG string, optimized for memory usage and performance
    
    Uses lxml for efficient parsing and svgelements for fast bounding box calculation
    when available, with fallback to svgpathtools.
    
    Parameters:
    ----------
    svg_string : str
        SVG string content
    target_size : tuple
        Target image dimensions
        
    Returns:
    -------
    list
        List of path information dictionaries
    """
    try:
        # Use lxml's efficient parser
        from lxml import etree
        import svgpathtools as spt
        import re
        
        # Use lxml's efficient parsing method
        parser = etree.XMLParser(remove_blank_text=True, recover=True)
        try:
            root = etree.fromstring(svg_string.encode('utf-8'), parser)
        except Exception as e:
            print(f"XML parsing failed: {e}")
            return []
        
        # Extract SVG attributes using XPath for efficiency
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        svg_element = root if root.tag.endswith('svg') else root.find('.//svg:svg', namespaces=namespace)
        
        if svg_element is None:
            print("Cannot find SVG element")
            return []
        
        # Extract dimension information
        svg_svg_attributes = dict(svg_element.attrib)
        width = height = None
        
        # Quick extraction of width and height
        if 'width' in svg_svg_attributes and 'height' in svg_svg_attributes:
            width_str = svg_svg_attributes['width']
            height_str = svg_svg_attributes['height']
            
            # Efficient numerical extraction
            width_match = re.search(r'[\d.]+', width_str)
            height_match = re.search(r'[\d.]+', height_str)
            
            if width_match and height_match:
                width = float(width_match.group())
                height = float(height_match.group())
        
        # Extract dimensions from viewBox
        if (width is None or height is None) and 'viewBox' in svg_svg_attributes:
            viewbox = svg_svg_attributes['viewBox'].split()
            if len(viewbox) >= 4:
                try:
                    width = float(viewbox[2])
                    height = float(viewbox[3])
                except (ValueError, IndexError):
                    pass
        
        # Use default dimensions
        if width is None or height is None:
            width, height = target_size
        
        # Get viewBox
        viewBox = svg_svg_attributes.get('viewBox', "0 0 1024 1024")
        
        # Use XPath to directly get all paths
        path_elements = root.xpath('.//svg:path', namespaces=namespace)
        
        # Pre-allocate results list
        paths_data = []
        
        # Pre-cache scaling ratios
        x_scale = target_size[0] / width
        y_scale = target_size[1] / height
        
        # Batch process paths to reduce loop overhead
        for i, path_elem in enumerate(path_elements):
            attributes = dict(path_elem.attrib)
            path_d = attributes.get('d')
            
            if not path_d:  # Skip paths without d attribute
                continue
            
            try:
                # First try using svgelements for bounding box calculation
                try:
                    from svgelements import Path as SVGElementsPath
                    # Use svgelements library to parse path and calculate bounds
                    svg_path = SVGElementsPath(path_d)
                    bounds = svg_path.bbox()
                    
                    if bounds is None:
                        print(f"svgelements cannot calculate bounding box, skipping path {i}")
                        continue
                        
                    path_bounds = {
                        'minX': bounds[0],
                        'minY': bounds[1],
                        'maxX': bounds[2],
                        'maxY': bounds[3],
                        'width': bounds[2] - bounds[0],
                        'height': bounds[3] - bounds[1]
                    }
                except ImportError:
                    # Fallback to original method if svgelements not available
                    print("svgelements library not available, using original method")
                    path = spt.parse_path(path_d)
                    path_bounds = get_path_bounds(path)
                except Exception as e:
                    print(f"svgelements calculation failed: {e}, trying original method")
                    path = spt.parse_path(path_d)
                    path_bounds = get_path_bounds(path)
                
                # Only calculate valid bounding boxes
                if path_bounds['width'] <= 0 or path_bounds['height'] <= 0:
                    continue
                
                # Extract fill color
                fill = attributes.get('fill', '#000000')
                
                # Efficiently calculate bounding box coordinates
                bbox = (
                    int(path_bounds['minX'] * x_scale),
                    int(path_bounds['minY'] * y_scale),
                    int(path_bounds['width'] * x_scale),
                    int(path_bounds['height'] * y_scale)
                )
                
                # Parse path for rendering - still use svgpathtools
                path = spt.parse_path(path_d)
                
                # Only render image when needed
                path_image, mask = render_path_to_image(
                    path, attributes, width, height, target_size, viewBox)
                
                if path_image is None or mask is None:
                    continue
                
                # Parse color
                fill_rgb = parse_color(fill)
                
                # Add to results list
                paths_data.append({
                    'id': attributes.get('id', f'path_{i}'),
                    'color': fill_rgb,
                    'bbox': bbox,
                    'mask': mask,
                    'image': path_image,
                    'bounds': path_bounds
                })
            
            except Exception as e:
                # Refined error handling, allow partial failures
                print(f"Error processing path {i}: {e}")
                continue
        
        return paths_data
        
    except Exception as e:
        print(f"Error extracting SVG data from string: {e}")
        import traceback
        traceback.print_exc()
        return []

def create_final_visualization(reference_paths, generated_paths, result, similarity_matrices, output_path):
    """
    Create comprehensive SVG comparison visualization showing path matching and differences
    
    Generates a multi-panel visualization with:
    - Reference and generated path images side by side
    - Difference visualization with color-coded overlays
    - Matching scores and overall quality metrics
    
    Parameters:
    ----------
    reference_paths : list
        List of reference SVG path information
    generated_paths : list
        List of generated SVG path information
    result : dict
        Comparison results from SVG analysis
    similarity_matrices : dict
        Similarity matrices for each dimension
    output_path : str
        Output image file path
    """
    
    
    # Get matching results
    matches = result['matches']
    
    # If no matches, create error message image
    if not matches:
        print("No valid matches found")
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(0.5, 0.5, "No valid path matches found", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=16, color='red')
        ax.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return
    
    # Calculate number of matched pairs
    n_matches = len(matches)
    
    # Set image dimensions and DPI
    fig_width = 12  # inches
    row_height = 3  # height per row (inches)
    fig_height = n_matches * row_height + 2  # extra space for overall score
    dpi = 100
    
    # Create figure with white background
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='white')
    
    # Add title area at top
    title_ax = fig.add_subplot(gridspec.GridSpec(n_matches + 1, 1)[0])
    title_ax.text(0.5, 0.6, f"SVG Comparison Result - Overall Score: {result['overall_quality_score']}/100", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=18, fontweight='bold')
    
    # Add dimension score information
    dimension_scores = result['dimension_scores']
    score_text = (f"Shape: {dimension_scores['shape']:.2f}  "
                 f"Color: {dimension_scores['color']:.2f}  "
                 f"Position: {dimension_scores['position']:.2f}")
    title_ax.text(0.5, 0.3, score_text, 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12)
    title_ax.axis('off')
    
    # Create 3-column grid, starting from second row
    gs = gridspec.GridSpec(n_matches + 1, 3)
    gs.update(wspace=0.3, hspace=0.4)
    
    # Add column headers
    col_titles = ["Reference Path", "Generated Path", "Difference Analysis"]
    for i, title in enumerate(col_titles):
        col_title_ax = fig.add_subplot(gs[0, i])
        col_title_ax.text(0.5, 0.5, title, 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14, fontweight='bold')
        col_title_ax.axis('off')
    
    # Create visualization for each matched pair
    for i, (ref_idx, gen_idx) in enumerate(matches):
        row_idx = i + 1  # Start from second row
        ref_path = reference_paths[ref_idx]
        gen_path = generated_paths[gen_idx]
        
        # Ensure we have valid mask data
        if 'mask' not in ref_path or 'mask' not in gen_path:
            print(f"Warning: Match {i+1} missing mask data, skipping")
            continue
            
        # Left column: reference path
        ax1 = fig.add_subplot(gs[row_idx, 0])
        ref_mask_array = np.array(ref_path['mask'])
        ax1.imshow(ref_mask_array)
        ax1.set_title(f"Reference Path {ref_idx}", fontsize=10)
        ax1.axis('off')
        
        # Middle column: generated path
        ax2 = fig.add_subplot(gs[row_idx, 1])
        gen_mask_array = np.array(gen_path['mask'])
        ax2.imshow(gen_mask_array)
        ax2.set_title(f"Generated Path {gen_idx}", fontsize=10)
        ax2.axis('off')
        
        # Right column: difference visualization
        ax3 = fig.add_subplot(gs[row_idx, 2])
        
        # Get mask dimensions
        width, height = ref_path['mask'].size
        
        # Create RGB difference image
        diff_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Convert PIL images to binary masks
        ref_mask_bool = np.array(ref_path['mask'])[:,:,3] > 0
        gen_mask_bool = np.array(gen_path['mask'])[:,:,3] > 0
        
        # Find overlapping and non-overlapping regions
        overlap = np.logical_and(ref_mask_bool, gen_mask_bool)
        ref_only = np.logical_and(ref_mask_bool, ~gen_mask_bool)
        gen_only = np.logical_and(gen_mask_bool, ~ref_mask_bool)
        
        # Color coding
        diff_img[overlap] = [0, 255, 0]     # Green for overlap
        diff_img[ref_only] = [255, 0, 0]    # Red for reference only
        diff_img[gen_only] = [255, 255, 0]  # Yellow for generated only
        
        ax3.imshow(diff_img)
        
        # Calculate IoU for title
        iou = similarity_matrices['shape'][ref_idx, gen_idx]
        ax3.set_title(f"IoU: {iou:.2f}", fontsize=10)
        ax3.axis('off')
        
        # Add detailed score information
        shape_score = similarity_matrices['shape'][ref_idx, gen_idx]
        color_score = similarity_matrices['color'][ref_idx, gen_idx]
        position_score = similarity_matrices['position'][ref_idx, gen_idx]
        
        score_text = f"Shape: {shape_score:.2f}, Color: {color_score:.2f}, Position: {position_score:.2f}"
        ax3.text(0.5, -0.15, score_text, 
                transform=ax3.transAxes,
                horizontalalignment='center', 
                fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Overlap'),
        Patch(facecolor='red', label='Reference Only'),
        Patch(facecolor='yellow', label='Generated Only')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    print(f"Saving visualization to: {output_path}")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def visualize_dimension_scores(result, output_path):
    """
    Create visualization of dimension scores with bar chart
    
    Parameters:
    ----------
    result : dict
        Comparison results containing dimension scores
    output_path : str
        Base output path for generating dimension score chart
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Extract dimension scores
    dimension_scores = result['dimension_scores']
    
    # Create chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Prepare data
    dimensions = list(dimension_scores.keys())
    scores = [dimension_scores[dim] for dim in dimensions]
    
    # Define colors for each dimension
    colors = ['#ff6b6b', '#48dbfb', '#1dd1a1', '#feca57']
    
    # Create bar chart
    bars = ax.bar(dimensions, scores, color=colors[:len(dimensions)], width=0.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontsize=12)
    
    # Add overall score label
    ax.text(0.5, 0.92, f"Overall Quality Score: {result['overall_quality_score']}/100",
            ha='center', va='center', transform=ax.transAxes,
            fontsize=16, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Set chart properties
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('SVG Comparison Dimension Scores', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Beautify chart
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save chart
    plt.tight_layout()
    
    # Build dimension score output path
    output_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    dimension_output_path = os.path.join(output_dir, f"{base_name}_dimensions.png")
    
    plt.savefig(dimension_output_path)
    plt.close()
    
    print(f"Dimension score visualization saved to: {dimension_output_path}")

def example_usage(reference_svg, generated_svg, target_size=(256, 256), output_path=None):
    """
    Compare two SVG strings directly, perform evaluation and generate visualizations
    
    This is the main entry point for SVG comparison. It handles the complete workflow:
    1. Extract path information from both SVGs
    2. Perform multi-dimensional similarity analysis
    3. Generate comprehensive evaluation results
    4. Create visualization outputs
    
    Parameters:
    ----------
    reference_svg : str
        Reference SVG string (ground truth)
    generated_svg : str
        Generated SVG string (to be evaluated)
    target_size : tuple
        Target image dimensions for comparison (width, height)
    output_path : str, optional
        Output file path for visualization, if None no visualization is saved
        
    Returns:
    -------
    dict
        Comprehensive comparison results including:
        - overall_quality_score: 0-100 integer score
        - avg_score: Average similarity score for matched paths
        - match_rate: Ratio of successfully matched paths
        - dimension_scores: Scores for each dimension (shape, color, position)
        - feedback: Detailed feedback for each matched path pair
        - reference_paths: Extracted reference path data
        - generated_paths: Extracted generated path data
    """
    print("Starting SVG comparison analysis...")
    
    # Extract path information from SVG strings
    print("Extracting reference SVG paths...")
    reference_paths = extract_svg_data_from_string(reference_svg, target_size)
    print(f"Extracted {len(reference_paths)} reference paths")
    
    print("Extracting generated SVG paths...")
    generated_paths = extract_svg_data_from_string(generated_svg, target_size)
    print(f"Extracted {len(generated_paths)} generated paths")
    
    # Handle edge case where no paths are found
    if not reference_paths or not generated_paths:
        print("Warning: At least one SVG has no extractable paths")
        return {
            'overall_quality_score': 0,
            'avg_score': 0.0,
            'match_rate': 0.0,
            'dimension_scores': {
                'shape': 0.0,
                'color': 0.0,
                'position': 0.0,
            },
            'feedback': [],
            'reference_paths': reference_paths,
            'generated_paths': generated_paths
        }
    
    # Initialize comparison system
    print("Initializing SVG comparison system...")
    comparator = SVGComparisonSystem()
    
    # Perform SVG comparison
    print("Performing multi-dimensional SVG analysis...")
    result = comparator.compare_svgs(reference_paths, generated_paths, target_size)
    
    # Output results to console
    print(f"\n=== SVG Comparison Results ===")
    print(f"Overall Quality Score: {result['overall_quality_score']}/100")
    print(f"Average Match Score: {result['avg_score']:.3f}")
    print(f"Match Rate: {result['match_rate']:.3f}")
    
    # Output dimension scores
    print(f"\nDimension-wise Average Scores:")
    for dim, score in result['dimension_scores'].items():
        print(f"  {dim.capitalize()}: {score:.3f}")
    
    # Output detailed feedback
    print(f"\nDetailed Path Matching Feedback:")
    for i, fb in enumerate(result['feedback']):
        print(f"Match {i+1}: Path {fb['ref_path_id']} ↔ Path {fb['gen_path_id']}")
        print(f"  Shape: {fb['shape_score']:.3f}")
        print(f"  Color: {fb['color_score']:.3f}")
        print(f"  Position: {fb['position_score']:.3f}")
        
        if fb['suggestions']:
            print(f"  Suggestions:")
            for suggestion in fb['suggestions']:
                print(f"    • {suggestion}")
        print()
    
    # Generate visualizations if output path provided
    if output_path:
        print("Generating visualization outputs...")
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get similarity matrices for visualization
        similarity_matrices = comparator.build_similarity_matrices(
            reference_paths, generated_paths, target_size)
        
        # Add path information to results for visualization
        result['reference_paths'] = reference_paths
        result['generated_paths'] = generated_paths
        
        # Create comprehensive visualization
        create_final_visualization(
            reference_paths, generated_paths, result, 
            similarity_matrices, output_path
        )
        
        # Create dimension score visualization
        visualize_dimension_scores(result, output_path)
        
        print(f"Visualizations saved to: {output_path}")
    
    print("SVG comparison analysis complete!")
    return result

def compare_svg_files(reference_path, generated_path, target_size=(256, 256), output_path=None):
    """
    Compare two SVG files and generate comprehensive analysis
    
    Convenience function for file-based SVG comparison.
    
    Parameters:
    ----------
    reference_path : str
        Path to reference SVG file
    generated_path : str
        Path to generated SVG file
    target_size : tuple
        Target image dimensions
    output_path : str, optional
        Output path for visualizations
        
    Returns:
    -------
    dict
        Comparison results
    """
    try:
        # Read SVG files
        with open(reference_path, 'r', encoding='utf-8') as f:
            reference_svg = f.read()
        with open(generated_path, 'r', encoding='utf-8') as f:
            generated_svg = f.read()
            
        # Use main comparison function
        return example_usage(reference_svg, generated_svg, target_size, output_path)
        
    except FileNotFoundError as e:
        print(f"Error reading SVG files: {e}")
        return None
    except Exception as e:
        print(f"Error during SVG comparison: {e}")
        return None

# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the SVG comparison system
    
    This demonstrates how to use the system for comparing two SVGs,
    either from file paths or SVG strings directly.
    """
    
    # Example 1: Compare sample SVG strings
    print("=== Example 1: Comparing Sample SVG Strings ===")
    
    # Sample reference SVG (crown shape)
    reference_svg = '''<svg viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">
        <path d="M200 300 L350 280 L400 150 L450 280 L600 300 L480 380 L520 520 L400 440 L280 520 L320 380 Z" 
              fill="gold" stroke="orange" stroke-width="5"/>
    </svg>'''
    
    # Sample generated SVG (rectangles)
    generated_svg = '''<svg viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">
        <path d="M100 100 L300 100 L300 300 L100 300 Z" fill="red"/>
        <path d="M350 100 L550 100 L550 300 L350 300 Z" fill="green"/>
        <path d="M600 100 L800 100 L800 300 L600 300 Z" fill="blue"/>
    </svg>'''
    
    # Perform comparison
    result = example_usage(
        reference_svg, 
        generated_svg, 
        target_size=(256, 256),
        output_path='svg_comparison_example.png'
    )
    
    print(f"Example 1 Results: Overall Score = {result['overall_quality_score']}/100")
    print("-" * 60)
    
    # Example 2: Compare SVG files (if available)
    print("\n=== Example 2: File-based Comparison ===")
    
    # Example file paths (update these as needed)
    reference_file = '/path/to/reference.svg'
    generated_file = '/path/to/generated.svg'
    
    # Check if files exist before attempting comparison
    if os.path.exists(reference_file) and os.path.exists(generated_file):
        result = compare_svg_files(
            reference_file,
            generated_file,
            target_size=(256, 256),
            output_path='file_comparison_result.png'
        )
        
        if result:
            print(f"File comparison completed: Overall Score = {result['overall_quality_score']}/100")
        else:
            print("File comparison failed")
    else:
        print("Example SVG files not found, skipping file comparison demo")
        print("To test file comparison, update the file paths in the example")
    
    print("\n=== SVG Comparison System Demo Complete ===")
    print("Features demonstrated:")
    print("• Multi-dimensional similarity analysis (shape, color, position)")
    print("• Hungarian algorithm for optimal path matching")
    print("• Comprehensive evaluation metrics")
    print("• Visual feedback and suggestions")
    print("• Detailed comparison visualizations")