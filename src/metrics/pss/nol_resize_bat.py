#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from PIL import Image
from lxml import etree
from svgpathtools import parse_path, Path, Line, CubicBezier, QuadraticBezier, Arc
import svgpath2mpl
import cairosvg
import tempfile
import platform

def configure_chinese_font():
    """Configure Chinese font support for matplotlib based on the operating system"""
    system = platform.system()
    
    if system == 'Windows':
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    elif system == 'Darwin':  
        font_list = ['PingFang SC', 'STHeiti', 'Heiti SC', 'Hiragino Sans GB', 'Apple LiGothic']
    else: 
        font_list = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']
    
    font_found = False
    for font in font_list:
        try:
            if font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                font_found = True
                break
            for f in fm.fontManager.ttflist:
                if font.lower() in f.name.lower():
                    plt.rcParams['font.sans-serif'] = [f.name] + plt.rcParams['font.sans-serif']
                    font_found = True
                    break
            if font_found:
                break
        except Exception as e:
            print(f"Failed to load font {font}: {e}")
    
    # If no suitable Chinese font is found, show warning
    if not font_found:
        print("Warning: No suitable Chinese font found. Chinese characters may not display correctly.")
        print("You can install one of these fonts to fix this:")
        print(" - Noto Sans CJK SC (Recommended, open-source Chinese font by Google)")
        print(" - WenQuanYi Micro Hei")
        print(" - Adobe Source Han Sans")
    
    # Enable minus sign display
    plt.rcParams['axes.unicode_minus'] = False

class SVGScaler:
    """SVG scaling utility class for comparing and adjusting SVG path proportions"""
    
    def __init__(self):
        """Initialize SVG scaler"""
        pass
        
    def parse_svg(self, svg_str):
        """
        Parse complete SVG string
        
        Args:
            svg_str (str): SVG string
            
        Returns:
            lxml.etree._Element: Parsed SVG element tree
        """
        try:
            # Parse SVG using lxml
            return etree.fromstring(svg_str.encode('utf-8'))
        except Exception as e:
            print(f"Error parsing SVG: {e}")
            # Create empty SVG element
            return etree.fromstring('<svg xmlns="http://www.w3.org/2000/svg"></svg>'.encode('utf-8'))
   
    def get_path_bounds(self, path_str):
        """
        Get bounding box of SVG path using svgpathtools
        
        Args:
            path_str (str): SVG path string
            
        Returns:
            dict: Dictionary containing bounding box info {minX, minY, maxX, maxY, width, height}
        """
        try:
            # Parse path using svgpathtools
            path = parse_path(path_str)
            
            if not path:
                # Return default bounding box
                return {
                    'minX': 0, 'minY': 0, 'maxX': 0, 'maxY': 0, 'width': 0, 'height': 0
                }
            
            # Get bounding box (using real and imaginary parts for x and y coordinates)
            x_values = []
            y_values = []
            
            # Iterate through all points in path
            for segment in path:
                if isinstance(segment, Line):
                    points = [segment.start, segment.end]
                elif isinstance(segment, CubicBezier):
                    points = [segment.start, segment.control1, segment.control2, segment.end]
                elif isinstance(segment, QuadraticBezier):
                    points = [segment.start, segment.control, segment.end]
                elif isinstance(segment, Arc):
                    points = [segment.start, segment.end]
                    for t in np.linspace(0, 1, 10):
                        points.append(segment.point(t))
                else:
                    points = [segment.start, segment.end]
                
                for point in points:
                    x_values.append(point.real)
                    y_values.append(point.imag)
            
            # Calculate bounding box
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
            print(f"Error getting path bounds: {e}")
            return {
                'minX': 0, 'minY': 0, 'maxX': 0, 'maxY': 0, 'width': 0, 'height': 0
            }
            
    def get_svg_bounds(self, svg_str):
        """
        Get overall bounding box of all paths in SVG
        
        Args:
            svg_str (str): SVG string
                
        Returns:
            dict: Dictionary containing bounding box info of all paths
        """
        try:
            # Parse SVG
            svg_elem = self.parse_svg(svg_str)
            
            # Find all paths
            paths = []
            paths.extend(svg_elem.findall('.//{http://www.w3.org/2000/svg}path'))
            print(f"paths: {paths}")
            if not paths:
                return {
                    'minX': 0, 'minY': 0, 'maxX': 0, 'maxY': 0, 'width': 0, 'height': 0
                }
                
            # Initialize boundary values
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
                
            # Calculate overall bounds of all paths
            for path in paths:
                d = path.get('d')
                if d:
                    path_bounds = self.get_path_bounds(d)
                    min_x = min(min_x, path_bounds['minX'])
                    min_y = min(min_y, path_bounds['minY'])
                    max_x = max(max_x, path_bounds['maxX'])
                    max_y = max(max_y, path_bounds['maxY'])
                
            # Calculate width and height
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
            print(f"Error getting bounds of all paths: {e}")
            return {
                'minX': 0, 'minY': 0, 'maxX': 0, 'maxY': 0, 'width': 0, 'height': 0
            }
            
    def scale_path_with_svgpathtools(self, path_str, scale_x, scale_y, center_x=None, center_y=None):
        """
        Scale SVG path using svgpathtools with specified ratios
        
        Args:
            path_str (str): Original SVG path string
            scale_x (float): X-axis scaling ratio
            scale_y (float): Y-axis scaling ratio
            center_x (float, optional): X coordinate of scaling center point
            center_y (float, optional): Y coordinate of scaling center point
            
        Returns:
            str: Scaled SVG path string
        """
        try:
            # Parse path
            path = parse_path(path_str)
            
            # If center point not specified, use path center
            if center_x is None or center_y is None:
                bounds = self.get_path_bounds(path_str)
                center_x = bounds['minX'] + bounds['width'] / 2
                center_y = bounds['minY'] + bounds['height'] / 2
            
            # Create center point
            center = complex(center_x, center_y)
            
            # Define scaling function
            def scale_transform(point):
                # First move to origin
                translated = point - center
                # Scale
                scaled = complex(translated.real * scale_x, translated.imag * scale_y)
                # Move back
                return scaled + center
            
            # Apply transformation
            scaled_path = Path()
            
            for segment in path:
                if isinstance(segment, Line):
                    new_start = scale_transform(segment.start)
                    new_end = scale_transform(segment.end)
                    scaled_path.append(Line(new_start, new_end))
                    
                elif isinstance(segment, CubicBezier):
                    new_start = scale_transform(segment.start)
                    new_control1 = scale_transform(segment.control1)
                    new_control2 = scale_transform(segment.control2)
                    new_end = scale_transform(segment.end)
                    scaled_path.append(CubicBezier(new_start, new_control1, new_control2, new_end))
                    
                elif isinstance(segment, QuadraticBezier):
                    new_start = scale_transform(segment.start)
                    new_control = scale_transform(segment.control)
                    new_end = scale_transform(segment.end)
                    scaled_path.append(QuadraticBezier(new_start, new_control, new_end))
                    
                elif isinstance(segment, Arc):
                    # For arcs, we need special handling
                    new_start = scale_transform(segment.start)
                    new_end = scale_transform(segment.end)
                    # Scale radius
                    new_radius = complex(segment.radius.real * scale_x, segment.radius.imag * scale_y)
                    # Create new arc
                    scaled_path.append(Arc(new_start, new_radius, segment.rotation, 
                                        segment.large_arc, segment.sweep, new_end))
                else:
                    # Handle other segment types
                    scaled_path.append(segment)
            
            # Convert back to SVG path string
            return scaled_path.d()
            
        except Exception as e:
            print(f"Error scaling path with svgpathtools: {e}")
            return path_str  # Return original path if error occurs
            
    def scale_svg(self, svg_str, scale_x, scale_y):
        """
        Scale only paths in SVG while keeping viewBox unchanged
        
        Args:
            svg_str (str): Original SVG string or pure path string
            scale_x (float): X-axis scaling ratio
            scale_y (float): Y-axis scaling ratio
            
        Returns:
            str: Scaled SVG string
        """
        try:
            # Parse SVG
            svg_elem = self.parse_svg(svg_str)
            
            # Get overall bounds of all paths to determine scaling center
            bounds = self.get_svg_bounds(svg_str)
            global_center_x = bounds['minX'] + bounds['width'] / 2
            global_center_y = bounds['minY'] + bounds['height'] / 2
                
            # Find all paths
            paths = []
            paths.extend(svg_elem.findall('.//{http://www.w3.org/2000/svg}path'))
            
            for path in paths:
                d = path.get('d')
                if d:
                    # Scale each path using global center point
                    scaled_d = self.scale_path_with_svgpathtools(d, scale_x, scale_y, global_center_x, global_center_y)
                    # Update path data
                    path.set('d', scaled_d)
            
            # Convert back to string
            return etree.tostring(svg_elem, encoding='unicode')
                
        except Exception as e:
            print(f"Error scaling SVG paths: {e}")
            return svg_str
        
    def _render_svg_to_axes(self, svg_str, ax):
        """
        Render SVG string to matplotlib axes
        
        Args:
            svg_str (str): SVG string
            ax (matplotlib.axes.Axes): matplotlib axes object
        """
        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
        img = Image.open(io.BytesIO(png_data))
        ax.imshow(np.array(img))
        ax.axis('off')

    def compare_and_scale_svg(self, generated_svg, answer_svg, preserve_aspect_ratio=True, 
                       output_file='comparison.png', dpi=150, show_plot=False):
        """
        Main function: Compare and scale path content in SVG
        
        Args:
            generated_svg (str): Model-generated SVG string or pure path string
            answer_svg (str): Ground truth SVG string or pure path string
            preserve_aspect_ratio (bool): Whether to preserve aspect ratio, default True
            output_file (str): Output PNG file path
            dpi (int): Image DPI
            show_plot (bool): Whether to show image
            
        Returns:
            dict: Dictionary containing result information
        """
        # Ensure Chinese characters display correctly
        configure_chinese_font()
        
        # Determine if it's pure path or SVG
        is_path_only = not (generated_svg.strip().startswith('<') and answer_svg.strip().startswith('<'))
        
        # Get comparison objects
        gen_for_visualization = generated_svg
        ans_for_visualization = answer_svg
        gen_for_comparison = generated_svg
        ans_for_comparison = answer_svg
        
        # Calculate path proportion differences
        gen_bounds = self.get_svg_bounds(gen_for_comparison)
        ans_bounds = self.get_svg_bounds(ans_for_comparison)
        
        # Calculate ratios
        width_ratio = ans_bounds['width'] / gen_bounds['width'] if gen_bounds['width'] > 0 else 1
        height_ratio = ans_bounds['height'] / gen_bounds['height'] if gen_bounds['height'] > 0 else 1
        area_ratio = (ans_bounds['width'] * ans_bounds['height']) / (gen_bounds['width'] * gen_bounds['height']) if (gen_bounds['width'] * gen_bounds['height']) > 0 else 1
        
        ratios = {
            'widthRatio': width_ratio,
            'heightRatio': height_ratio,
            'areaRatio': area_ratio,
            'generated': gen_bounds,
            'answer': ans_bounds
        }
        
        # Calculate scaling factors
        scale_x = width_ratio
        scale_y = height_ratio
        
        # If preserving aspect ratio, use the smaller ratio
        if preserve_aspect_ratio:
            min_ratio = min(scale_x, scale_y)
            scale_x = scale_y = min_ratio
        
        scaled_svg = self.scale_svg(generated_svg, scale_x, scale_y)
        
        # Create visualization comparison
        vis_path = self._visualize_svg_comparison(
            gen_for_visualization, 
            ans_for_visualization, 
            scaled_svg, 
            output_file=output_file, 
            dpi=dpi, 
            show_plot=show_plot,
            ratios=ratios
        )
        
        # Return results
        result = {
            'originalSvg': generated_svg,
            'answerSvg': answer_svg,
            'ratios': ratios,
            'visualizationPath': vis_path
        }
        
        # Return different results based on input type
        result['scaledSvg'] = scaled_svg
        
        return result
        
    def _visualize_svg_comparison(self, generated_svg, answer_svg, scaled_svg, 
                             output_file='comparison.png', dpi=150, show_plot=False,
                             ratios=None):
        """
        Create visualization comparison of complete SVG
        
        Args:
            generated_svg (str): Model-generated SVG string
            answer_svg (str): Ground truth SVG string
            scaled_svg (str): Scaled SVG string
            output_file (str): Output PNG file path
            dpi (int): Image DPI
            show_plot (bool): Whether to show image
            ratios (dict): Ratio information dictionary
            
        Returns:
            str: Output file path
        """
        # Ensure Chinese characters display correctly
        configure_chinese_font()
        
        # Create figure
        fig = plt.figure(figsize=(15, 10), dpi=dpi)
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 2, 2])
        
        # Display ratio information
        ax_info = plt.subplot(gs[0, :])
        ax_info.axis('off')
        
        if ratios:
            table_data = [
                ['', 'Generated SVG', 'Answer SVG', 'Ratio'],
                ['Width', f"{ratios['generated']['width']:.2f}", f"{ratios['answer']['width']:.2f}", f"{ratios['widthRatio']:.4f}"],
                ['Height', f"{ratios['generated']['height']:.2f}", f"{ratios['answer']['height']:.2f}", f"{ratios['heightRatio']:.4f}"],
                ['Area', f"{ratios['generated']['width'] * ratios['generated']['height']:.2f}", 
                f"{ratios['answer']['width'] * ratios['answer']['height']:.2f}", f"{ratios['areaRatio']:.4f}"]
            ]
            
            table = ax_info.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Set table style to ensure Chinese characters display correctly
            for (i, j), cell in table.get_celld().items():
                cell.set_text_props(fontproperties=fm.FontProperties(family=plt.rcParams['font.sans-serif'][0]))
        
        ax_info.text(0.5, 0.9, 'SVG Scaling Comparison', ha='center', va='center', fontsize=14, fontweight='bold',
                    fontproperties=fm.FontProperties(family=plt.rcParams['font.sans-serif'][0]))
        
        # Render original SVG
        ax_generated = plt.subplot(gs[1, 0])
        ax_generated.set_title('Generated SVG', fontproperties=fm.FontProperties(family=plt.rcParams['font.sans-serif'][0]))
        self._render_svg_to_axes(generated_svg, ax_generated)
        
        # Render answer SVG
        ax_answer = plt.subplot(gs[1, 1])
        ax_answer.set_title('Answer SVG', fontproperties=fm.FontProperties(family=plt.rcParams['font.sans-serif'][0]))
        self._render_svg_to_axes(answer_svg, ax_answer)
        
        # Render scaled SVG
        ax_scaled = plt.subplot(gs[1, 2])
        ax_scaled.set_title('Scaled SVG', fontproperties=fm.FontProperties(family=plt.rcParams['font.sans-serif'][0]))
        self._render_svg_to_axes(scaled_svg, ax_scaled)
        
        # Create overlay comparison
        ax_overlap = plt.subplot(gs[2, :])
        ax_overlap.set_title('Overlay Comparison (Scaled vs Answer)', fontproperties=fm.FontProperties(family=plt.rcParams['font.sans-serif'][0]))
        self._render_svg_overlay(scaled_svg, answer_svg, ax_overlap)
        
        plt.tight_layout()
        
        # Save image
        try:
            # Try using sans-serif font
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'] + plt.rcParams['font.sans-serif']
            # Try using Agg backend
            plt.switch_backend('Agg')
            # Save directly as PNG
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        except Exception as e:
            print(f"Failed to save image: {e}")
            # Try alternative method
            try:
                # First save as SVG
                svg_temp = f"{os.path.splitext(output_file)[0]}_temp.svg"
                plt.savefig(svg_temp, format='svg', dpi=dpi, bbox_inches='tight')
                # Convert SVG to PNG using cairosvg
                cairosvg.svg2png(url=svg_temp, write_to=output_file, dpi=dpi)
                os.remove(svg_temp)  # Remove temporary SVG file
            except Exception as e2:
                print(f"SVG conversion failed: {e2}")
                # Final attempt without bbox_inches parameter
                plt.savefig(output_file, dpi=dpi)
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return output_file

    def _render_svg_overlay(self, svg1, svg2, ax):
        """
        Render overlay of two SVGs on same axes
        
        Args:
            svg1 (str): First SVG string
            svg2 (str): Second SVG string
            ax (matplotlib.axes.Axes): matplotlib axes object
        """
        # Create temporary overlay SVG
        svg1_elem = self.parse_svg(svg1)
        svg2_elem = self.parse_svg(svg2)
        
        # Get viewBox
        viewbox1 = svg1_elem.get('viewBox')
        viewbox2 = svg2_elem.get('viewBox')
        
        # Use union of both viewBoxes
        if viewbox1 and viewbox2:
            parts1 = list(map(float, viewbox1.split()))
            parts2 = list(map(float, viewbox2.split()))
            
            if len(parts1) == 4 and len(parts2) == 4:
                min_x = min(parts1[0], parts2[0])
                min_y = min(parts1[1], parts2[1])
                max_x1 = parts1[0] + parts1[2]
                max_y1 = parts1[1] + parts1[3]
                max_x2 = parts2[0] + parts2[2]
                max_y2 = parts2[1] + parts2[3]
                max_x = max(max_x1, max_x2)
                max_y = max(max_y1, max_y2)
                
                width = max_x - min_x
                height = max_y - min_y
                
                combined_viewbox = f"{min_x} {min_y} {width} {height}"
                svg1_elem.set('viewBox', combined_viewbox)
                svg2_elem.set('viewBox', combined_viewbox)
        
        # Add different colors and transparency to elements
        for elem in svg1_elem.xpath('//*[@fill or @stroke]'):
            # Set blue fill
            if elem.get('fill') and elem.get('fill') != 'none':
                elem.set('fill', '#0066cc')
            # Set blue stroke
            if elem.get('stroke') and elem.get('stroke') != 'none':
                elem.set('stroke', '#0066cc')
            # Add transparency
            elem.set('opacity', '0.6')
        
        for elem in svg2_elem.xpath('//*[@fill or @stroke]'):
            # Set green fill
            if elem.get('fill') and elem.get('fill') != 'none':
                elem.set('fill', '#00cc66')
            # Set green stroke
            if elem.get('stroke') and elem.get('stroke') != 'none':
                elem.set('stroke', '#00cc66')
            # Add transparency
            elem.set('opacity', '0.6')
        
        # Add svg2 content to svg1
        for child in svg2_elem:
            if child.tag != '{http://www.w3.org/2000/svg}defs' and child.tag != '{http://www.w3.org/2000/svg}metadata':
                svg1_elem.append(child)
        
        # Convert to string
        combined_svg = etree.tostring(svg1_elem, encoding='unicode')
        
        # Render to Axes
        png_data = cairosvg.svg2png(bytestring=combined_svg.encode('utf-8'))
        img = Image.open(io.BytesIO(png_data))
        ax.imshow(np.array(img))
        ax.axis('off')

def main():
    # Configure Chinese font
    configure_chinese_font()
    
    print("\n--- Complete SVG Comparison and Scaling Example ---")
    # Example SVG files
    generated_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
    <path d="M30,40 C35,35 40,35 45,40 C50,45 55,45 60,40 C65,35 70,35 75,40 
             L75,60 C70,65 65,65 60,60 C55,55 50,55 45,60 C40,65 35,65 30,60 Z" fill="purple" />
</svg>'''
        
    answer_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
    <path d="M20,40 C30,30 40,30 50,40 C60,50 70,50 80,40 C90,30 100,30 110,40 
             L110,60 C100,70 90,70 80,60 C70,50 60,50 50,60 C40,70 30,70 20,60 Z" fill="purple" />
</svg>'''
    
    # Create SVG scaler
    scaler = SVGScaler()
    
    # Use new method to perform SVG comparison and scaling
    result = scaler.compare_and_scale_svg(
        generated_svg, 
        answer_svg, 
        preserve_aspect_ratio=True,
        output_file='svg_comparison_full.png',
        show_plot=True
    )
    
    print(f"Width ratio: {result['ratios']['widthRatio']:.4f}")
    print(f"Height ratio: {result['ratios']['heightRatio']:.4f}")
    print(f"Area ratio: {result['ratios']['areaRatio']:.4f}")
    print(f"Visualization saved to: {result['visualizationPath']}")
    
if __name__ == "__main__":
    main()