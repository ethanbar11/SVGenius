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

def configure_font():
    """Configure matplotlib font settings for better display"""
    try:
        # Check for available fonts on the system
        import subprocess
        font_check = subprocess.run(['fc-list', ':', 'family'], 
                                  stdout=subprocess.PIPE, 
                                  text=True)
        available_fonts = font_check.stdout.lower()
        
        if 'noto sans' in available_fonts:
            plt.rcParams['font.sans-serif'] = ['Noto Sans'] + plt.rcParams['font.sans-serif']
            print("Using Noto Sans font")
        elif 'dejavu sans' in available_fonts:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif'] 
            print("Using DejaVu Sans font")
        else:
            # Use matplotlib's default font
            plt.rcParams['font.sans-serif'] = ['sans-serif']
            print("Using default sans-serif font")
    except Exception as e:
        print(f"Error checking fonts: {e}")
        plt.rcParams['font.sans-serif'] = ['sans-serif']
    
    # Allow proper display of minus signs
    plt.rcParams['axes.unicode_minus'] = False

class SVGScaler:
    """SVG scaling utility class for comparing and adjusting SVG path proportions"""

    def __init__(self):
        """Initialize SVG scaler"""
        pass
        
    def parse_svg(self, svg_str):
        """
        Parse entire SVG string
        
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
            # Create an empty SVG element
            return etree.fromstring('<svg xmlns="http://www.w3.org/2000/svg"></svg>'.encode('utf-8'))
   
    def get_path_bounds(self, path_str):
        """
        Get SVG path bounding box using svgelements for robust handling of all SVG path commands
        
        Args:
            path_str (str): SVG path string
            
        Returns:
            dict: Dictionary containing bounding box info {minX, minY, maxX, maxY, width, height}
        """
        try:
            # Try using svgelements library
            try:
                from svgelements import Path as SVGElementsPath
                
                # Use svgelements to parse path and get bounds
                svg_path = SVGElementsPath(path_str)
                bounds = svg_path.bbox()
                
                if bounds is None:
                    # Return default bounding box
                    return {
                        'minX': 0, 'minY': 0, 'maxX': 0, 'maxY': 0, 'width': 0, 'height': 0
                    }
                
                return {
                    'minX': bounds[0],
                    'minY': bounds[1],
                    'maxX': bounds[2],
                    'maxY': bounds[3],
                    'width': bounds[2] - bounds[0],
                    'height': bounds[3] - bounds[1]
                }
            except ImportError:
                print("svgelements library not installed, using fallback method")
                
                # Fallback method: use original svgpathtools with enhanced error handling
                path = parse_path(path_str)
                
                if not path:
                    # Return default bounding box
                    return {
                        'minX': 0, 'minY': 0, 'maxX': 0, 'maxY': 0, 'width': 0, 'height': 0
                    }
                
                # Get bounding box (using real and imaginary parts for x and y coordinates)
                x_values = []
                y_values = []
                
                # Traverse all segments in the path with robust exception handling
                for segment in path:
                    try:
                        # Get points based on segment type
                        if isinstance(segment, Line):
                            points = [segment.start, segment.end]
                        elif isinstance(segment, CubicBezier):
                            points = [segment.start, segment.control1, segment.control2, segment.end]
                        elif isinstance(segment, QuadraticBezier):
                            points = [segment.start, segment.control, segment.end]
                        elif isinstance(segment, Arc):
                            # For arcs, only use start and end points to avoid potential issues with point() method
                            points = [segment.start, segment.end]
                        else:
                            # Handle other segment types
                            points = [segment.start, segment.end]
                        
                        # Extract coordinates from all points
                        for point in points:
                            if hasattr(point, 'real') and hasattr(point, 'imag'):
                                x_values.append(point.real)
                                y_values.append(point.imag)
                    except Exception as e:
                        print(f"Error processing path segment: {e}, skipping this segment")
                        continue
                
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
            print(f"Error getting path bounds: {e}")
            import traceback
            traceback.print_exc()
            # Return default bounding box
            return {
                'minX': 0, 'minY': 0, 'maxX': 0, 'maxY': 0, 'width': 0, 'height': 0
            }
            
    def get_svg_bounds(self, svg_str):
        """
        Get overall bounding box of all paths in SVG
        
        Args:
            svg_str (str): SVG string
                
        Returns:
            dict: Dictionary containing bounding box info for all paths
        """
        try:
            # Parse SVG
            svg_elem = self.parse_svg(svg_str)
            
            # Find all paths
            paths = []
            paths.extend(svg_elem.findall('.//{http://www.w3.org/2000/svg}path'))
            
            if not paths:
                return {
                    'minX': 0, 'minY': 0, 'maxX': 0, 'maxY': 0, 'width': 0, 'height': 0
                }
                
            # Initialize boundary values
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
                
            # Calculate overall bounds for all paths
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
            print(f"Error getting all path bounds: {e}")
            return {
                'minX': 0, 'minY': 0, 'maxX': 0, 'maxY': 0, 'width': 0, 'height': 0
            }
            
    def scale_path_with_svgpathtools(self, path_str, scale_x, scale_y, center_x=None, center_y=None):
        """
        Scale SVG path by specified ratios using svgpathtools
        
        Args:
            path_str (str): Original SVG path string
            scale_x (float): X-axis scaling ratio
            scale_y (float): Y-axis scaling ratio
            center_x (float, optional): X coordinate of scaling center
            center_y (float, optional): Y coordinate of scaling center
            
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
                # Translate to origin
                translated = point - center
                # Scale
                scaled = complex(translated.real * scale_x, translated.imag * scale_y)
                # Translate back
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
                    # Special handling for arcs
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
            
    def _normalize_color(self, color_str):
        """
        Normalize various color formats to hexadecimal format
        
        Args:
            color_str (str): Input color string
            
        Returns:
            str: Normalized hexadecimal color string
        """
        if color_str == '':
            return '#000000'  # Default empty fill to black
            
        # Convert to lowercase for processing
        color = color_str.lower().strip()
        
        # If 'none', return directly
        if color == 'none':
            return 'none'
        
        # If already hexadecimal format, ensure uppercase letters
        if color.startswith('#'):
            # Handle shorthand hexadecimal format
            if len(color) == 4:  # #RGB format
                r, g, b = color[1], color[2], color[3]
                return f'#{r}{r}{g}{g}{b}{b}'.upper()
            # Normal hexadecimal format
            return color.upper()
        
        # Handle rgb(), rgba() formats
        if color.startswith('rgb'):
            # Extract values
            try:
                content = color[color.find('(')+1:color.find(')')].strip()
                parts = [p.strip() for p in content.split(',')]
                
                # Handle percentage values
                numbers = []
                for part in parts[:3]:  # Only take RGB part
                    if part.endswith('%'):
                        # Convert percentage to 0-255
                        value = int(float(part[:-1]) * 2.55)
                    else:
                        value = int(float(part))
                    numbers.append(max(0, min(255, value)))  # Ensure in 0-255 range
                
                # Convert to hexadecimal
                return f'#{numbers[0]:02X}{numbers[1]:02X}{numbers[2]:02X}'
            except (ValueError, IndexError):
                pass
        
        # Handle HSL format (simple implementation)
        if color.startswith('hsl'):
            try:
                import colorsys
                content = color[color.find('(')+1:color.find(')')].strip()
                parts = [p.strip() for p in content.split(',')]
                
                # Extract HSL values
                h = float(parts[0].rstrip('°')) / 360.0
                
                s_part = parts[1]
                s = float(s_part[:-1]) / 100.0 if s_part.endswith('%') else float(s_part)
                
                l_part = parts[2]
                l = float(l_part[:-1]) / 100.0 if l_part.endswith('%') else float(l_part)
                
                # Convert to RGB
                r, g, b = colorsys.hls_to_rgb(h, l, s)
                r = int(r * 255)
                g = int(g * 255)
                b = int(b * 255)
                
                return f'#{r:02X}{g:02X}{b:02X}'
            except (ValueError, IndexError, ImportError):
                pass
        
        # Handle common named colors
        color_map = {
            'red': '#FF0000',
            'green': '#00FF00',
            'blue': '#0000FF',
            'yellow': '#FFFF00',
            'cyan': '#00FFFF',
            'magenta': '#FF00FF',
            'black': '#000000',
            'white': '#FFFFFF',
            'gray': '#808080',
            'grey': '#808080',
            'purple': '#800080',
            'orange': '#FFA500',
            'brown': '#A52A2A',
            'pink': '#FFC0CB',
            'lime': '#00FF00',
            'navy': '#000080',
            'teal': '#008080',
            'aqua': '#00FFFF',
            'silver': '#C0C0C0',
            'gold': '#FFD700'
        }
        
        # If it's a known named color
        if color in color_map:
            return color_map[color]
        
        # Unrecognized color, return original value
        return color_str
        
    def center_and_normalize_svg(self, svg_str, target_viewbox="0 0 1024 1024"):
        """
        Center align SVG and normalize colors, using svgpathtools for precise processing
        
        Args:
            svg_str (str): Original SVG string
            target_viewbox (str, optional): Target viewBox, defaults to "0 0 1024 1024"
            
        Returns:
            str: Processed SVG string
        """
        try:
            # Parse SVG
            svg_elem = self.parse_svg(svg_str)
            
            # Parse target viewBox
            target_vb_parts = list(map(float, target_viewbox.split()))
            target_width = target_vb_parts[2]
            target_height = target_vb_parts[3]
            target_center_x = target_vb_parts[0] + target_width / 2
            target_center_y = target_vb_parts[1] + target_height / 2
            
            # Get overall bounds of all paths
            bounds = self.get_svg_bounds(svg_str)
            
            # Calculate current SVG center point
            current_center_x = bounds['minX'] + bounds['width'] / 2
            current_center_y = bounds['minY'] + bounds['height'] / 2
            
            # Calculate required translation
            translate_x = target_center_x - current_center_x
            translate_y = target_center_y - current_center_y
            
            # Find all paths and process them
            paths = []
            paths.extend(svg_elem.findall('.//{http://www.w3.org/2000/svg}path'))
            
            for path in paths:
                # 1. Move path to canvas center
                d = path.get('d')
                if d:
                    # Parse path using svgpathtools
                    parsed_path = parse_path(d)
                    
                    # Apply translation transformation
                    translated_path = Path()
                    
                    for segment in parsed_path:
                        if isinstance(segment, Line):
                            new_start = segment.start + complex(translate_x, translate_y)
                            new_end = segment.end + complex(translate_x, translate_y)
                            translated_path.append(Line(new_start, new_end))
                            
                        elif isinstance(segment, CubicBezier):
                            new_start = segment.start + complex(translate_x, translate_y)
                            new_control1 = segment.control1 + complex(translate_x, translate_y)
                            new_control2 = segment.control2 + complex(translate_x, translate_y)
                            new_end = segment.end + complex(translate_x, translate_y)
                            translated_path.append(CubicBezier(new_start, new_control1, new_control2, new_end))
                            
                        elif isinstance(segment, QuadraticBezier):
                            new_start = segment.start + complex(translate_x, translate_y)
                            new_control = segment.control + complex(translate_x, translate_y)
                            new_end = segment.end + complex(translate_x, translate_y)
                            translated_path.append(QuadraticBezier(new_start, new_control, new_end))
                            
                        elif isinstance(segment, Arc):
                            new_start = segment.start + complex(translate_x, translate_y)
                            new_end = segment.end + complex(translate_x, translate_y)
                            # Translation doesn't change radius and rotation
                            translated_path.append(Arc(new_start, segment.radius, segment.rotation, 
                                                segment.large_arc, segment.sweep, new_end))
                        else:
                            # Handle other segment types - also apply translation
                            translated_segment = segment.translated(complex(translate_x, translate_y))
                            translated_path.append(translated_segment)
                    
                    # Update path data - ensure precision is preserved
                    path.set('d', translated_path.d())
                
                # 2. Normalize colors
                fill = path.get('fill')
                if fill:
                    normalized_fill = self._normalize_color(fill)
                    path.set('fill', normalized_fill)
                
                stroke = path.get('stroke')
                if stroke:
                    normalized_stroke = self._normalize_color(stroke)
                    path.set('stroke', normalized_stroke)
            
            # Set SVG viewBox attribute
            svg_elem.set('viewBox', target_viewbox)
            
            # Convert back to string
            return etree.tostring(svg_elem, encoding='unicode')
                
        except Exception as e:
            print(f"Error centering and normalizing SVG: {e}")
            return svg_str  # Return original SVG if error occurs
            
    def scale_svg(self, svg_str, scale_x, scale_y):
        """Scale paths in SVG by ratio, keeping viewBox unchanged"""
        
        # Add safety thresholds for scaling factors
        MIN_SCALE = 0.05  # Minimum scaling factor to prevent excessive shrinking
        MAX_SCALE = 10.0  # Maximum scaling factor to prevent excessive enlargement
        
        # Safely adjust scaling factors
        scale_x = max(MIN_SCALE, min(MAX_SCALE, scale_x))
        scale_y = max(MIN_SCALE, min(MAX_SCALE, scale_y))
        
        print(f"Safe scaling factors: X={scale_x:.4f}, Y={scale_y:.4f}")
        
        # Original scaling logic
        try:
            # Parse SVG
            svg_elem = self.parse_svg(svg_str)
            
            # Get overall bounds of all paths
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
                    scaled_d = self.scale_path_with_svgpathtools(
                        d, scale_x, scale_y, global_center_x, global_center_y
                    )
                    # Update path data
                    path.set('d', scaled_d)
            
            # Convert back to string
            return etree.tostring(svg_elem, encoding='unicode')
            
        except Exception as e:
            print(f"Error scaling SVG paths: {e}")
            return svg_str
        
    def _render_svg_to_axes(self, svg_str, ax):
        """
        Render SVG string to matplotlib axes with improved clarity
        
        Args:
            svg_str (str): SVG string
            ax (matplotlib.axes.Axes): matplotlib axes object
        """
        # Ensure SVG has correct namespace
        if "xmlns" not in svg_str and "<svg" in svg_str:
            svg_str = svg_str.replace("<svg", "<svg xmlns='http://www.w3.org/2000/svg'")
        
        # Use cairosvg for high-quality rendering
        png_data = cairosvg.svg2png(
            bytestring=svg_str.encode('utf-8'),
            scale=3.0,  # Increase scale factor for higher resolution
            background_color="white",  # Ensure white background
            unsafe=True  # Allow more complex SVG processing
        )
        
        # Use high-quality image processing
        img = Image.open(io.BytesIO(png_data))
        ax.imshow(np.array(img), interpolation='nearest')  # Use nearest neighbor interpolation for sharp edges
        ax.axis('off')

    def _render_svg_overlay(self, svg1, svg2, ax):
        """
        Render overlay of two SVGs on same axes with improved clarity and contrast
        
        Args:
            svg1 (str): First SVG string
            svg2 (str): Second SVG string
            ax (matplotlib.axes.Axes): matplotlib axes object
        """
        # Create a temporary overlay SVG
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
        
        # Add white background for better contrast
        background = etree.Element('{http://www.w3.org/2000/svg}rect')
        background.set('width', '100%')
        background.set('height', '100%')
        background.set('fill', 'white')
        svg1_elem.insert(0, background)
        
        # Add different colors and transparency to elements - use higher contrast colors
        for elem in svg1_elem.xpath('//*[@fill or @stroke]'):
            # Set blue fill (more vivid)
            if elem.get('fill') and elem.get('fill') != 'none':
                elem.set('fill', '#0055AA')  # Use darker blue for better contrast
            # Set blue stroke and increase width
            if elem.get('stroke') and elem.get('stroke') != 'none':
                elem.set('stroke', '#0055AA')
                # Increase stroke width for better visibility
                stroke_width = elem.get('stroke-width')
                if stroke_width:
                    try:
                        elem.set('stroke-width', str(float(stroke_width) * 1.5))
                    except (ValueError, TypeError):
                        pass
                else:
                    elem.set('stroke-width', '2')
            # Reduce transparency for better clarity
            elem.set('opacity', '0.8')
        
        for elem in svg2_elem.xpath('//*[@fill or @stroke]'):
            # Set green fill (more vivid)
            if elem.get('fill') and elem.get('fill') != 'none':
                elem.set('fill', '#00AA55')  # Use darker green for better contrast
            # Set green stroke and increase width
            if elem.get('stroke') and elem.get('stroke') != 'none':
                elem.set('stroke', '#00AA55')
                # Increase stroke width
                stroke_width = elem.get('stroke-width')
                if stroke_width:
                    try:
                        elem.set('stroke-width', str(float(stroke_width) * 1.5))
                    except (ValueError, TypeError):
                        pass
                else:
                    elem.set('stroke-width', '2')
            # Reduce transparency for better clarity
            elem.set('opacity', '0.8')
        
        # Add svg2 content to svg1
        for child in svg2_elem:
            if child.tag != '{http://www.w3.org/2000/svg}defs' and child.tag != '{http://www.w3.org/2000/svg}metadata':
                svg1_elem.append(child)
        
        # Convert to string
        combined_svg = etree.tostring(svg1_elem, encoding='unicode')
        
        # Use high-quality rendering settings
        png_data = cairosvg.svg2png(
            bytestring=combined_svg.encode('utf-8'),
            scale=3.0,  # Increase scale factor
            background_color="white",  # Ensure white background
            unsafe=True  # Allow more complex SVG processing
        )
        
        # Use high-quality display
        img = Image.open(io.BytesIO(png_data))
        ax.imshow(np.array(img), interpolation='nearest')  # Use nearest neighbor interpolation for sharp edges
        ax.axis('off')

    def _visualize_svg_comparison(self, generated_svg, answer_svg, scaled_svg, 
                        output_file='comparison.png', dpi=600, show_plot=False,
                        ratios=None):
        """
        Create visual comparison of entire SVG
        
        Args:
            generated_svg (str): Model-generated SVG string
            answer_svg (str): Reference answer SVG string
            scaled_svg (str): Scaled SVG string
            output_file (str): Output PNG file path
            dpi (int): Image DPI
            show_plot (bool): Whether to display image
            ratios (dict): Ratio information dictionary
            
        Returns:
            str: Output file path
        """
        # Create high-resolution image
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
            
        ax_info.text(0.5, 0.9, 'SVG Scale Comparison', ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Render original SVG
        ax_generated = plt.subplot(gs[1, 0])
        ax_generated.set_title('Generated SVG')
        self._render_svg_to_axes(generated_svg, ax_generated)
        
        # Render answer SVG
        ax_answer = plt.subplot(gs[1, 1])
        ax_answer.set_title('Answer SVG')
        self._render_svg_to_axes(answer_svg, ax_answer)
        
        # Render scaled SVG
        ax_scaled = plt.subplot(gs[1, 2])
        ax_scaled.set_title('Scaled SVG')
        self._render_svg_to_axes(scaled_svg, ax_scaled)
        
        # Create overlay comparison
        ax_overlap = plt.subplot(gs[2, :])
        ax_overlap.set_title('Overlay Comparison')
        self._render_svg_overlay(scaled_svg, answer_svg, ax_overlap)
        
        plt.tight_layout()
        
        # Save high-resolution image
        try:
            # Use high-quality save settings
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight', format='png', 
                    pil_kwargs={'quality': 95}, 
                    metadata={'CreationDate': None})  # Remove metadata to reduce file size
        except Exception as e:
            print(f"Failed to save image: {e}")
            try:
                # Save as SVG first
                svg_temp = f"{os.path.splitext(output_file)[0]}_temp.svg"
                plt.savefig(svg_temp, format='svg', dpi=dpi, bbox_inches='tight')
                # Use cairosvg for high-quality conversion
                cairosvg.svg2png(
                    url=svg_temp, 
                    write_to=output_file, 
                    dpi=dpi,
                    scale=2.0,  # Increase scale factor
                    background_color="white"
                )
                os.remove(svg_temp)  # Delete temporary SVG file
            except Exception as e2:
                print(f"SVG conversion failed: {e2}")
                # Last attempt, use basic save
                plt.savefig(output_file, dpi=dpi)
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return output_file

    def compare_and_scale_svg(self, generated_svg, answer_svg, preserve_aspect_ratio=True, 
                             output_file=None, dpi=150, show_plot=False):
        """
        Improved SVG processing workflow: center align and color standardization first, then scaling
        
        Args:
            generated_svg (str): Model-generated SVG string
            answer_svg (str): Reference answer SVG string
            preserve_aspect_ratio (bool): Whether to preserve aspect ratio, defaults to True
            output_file (str): Output PNG file path
            dpi (int): Image DPI
            show_plot (bool): Whether to display image
            
        Returns:
            dict: Dictionary containing result information
        """
        # Ensure proper font display
        configure_font()
        
        # Determine SVG viewBox
        answer_elem = self.parse_svg(answer_svg)
        target_viewbox = answer_elem.get('viewBox', "0 0 1024 1024")
        
        # 1. First perform center alignment and color standardization
        print("Step 1: Center aligning and color standardizing generated and answer SVGs...")
        normalized_gen_svg = self.center_and_normalize_svg(generated_svg, target_viewbox)
        normalized_answer_svg = self.center_and_normalize_svg(answer_svg, target_viewbox)
        
        # 2. Calculate bounds and ratios after alignment
        print("Step 2: Calculating ratio differences after standardization...")
        gen_bounds = self.get_svg_bounds(normalized_gen_svg)
        ans_bounds = self.get_svg_bounds(normalized_answer_svg)
        
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
        
        # 3. Calculate scaling factors
        scale_x = width_ratio
        scale_y = height_ratio
        
        # If aspect ratio needs to be preserved, use the smaller ratio
        if preserve_aspect_ratio:
            min_ratio = min(scale_x, scale_y)
            scale_x = scale_y = min_ratio
        
        print(f"Scaling factors - X: {scale_x:.4f}, Y: {scale_y:.4f}")
        
        # 4. Scale the standardized SVG
        print("Step 3: Scaling center-aligned SVG...")
        final_svg = self.scale_svg(normalized_gen_svg, scale_x, scale_y)
        
        # 5. Re-center align to ensure centering is maintained after scaling
        print("Step 4: Ensuring centering is maintained after scaling...")
        final_svg = self.center_and_normalize_svg(final_svg, target_viewbox)
        
        # Create visualization comparison
        print("Creating visual comparison...")
        vis_path = self._visualize_svg_comparison(
            generated_svg,  # Original generated SVG
            normalized_answer_svg,  # Standardized answer SVG
            final_svg,      # Final processed SVG
            output_file=output_file, 
            dpi=dpi, 
            show_plot=show_plot,
            ratios=ratios
        )
        
        # Return result
        result = {
            'originalSvg': generated_svg,
            'answerSvg': answer_svg,
            'normalizedGenSvg': normalized_gen_svg,
            'normalizedAnswerSvg': normalized_answer_svg,
            'finalSvg': final_svg,
            'ratios': ratios,
            'visualizationPath': vis_path
        }
        
        return result

def main():
    """Main function demonstrating SVG scaling functionality"""
    # Configure font
    configure_font()
    
    print("\n--- Complete SVG Comparison and Scaling Example ---")
    
    # Example SVG files
    generated_svg = '''<svg viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">
    <path d="M100 100 L300 100 L300 300 L100 300 Z" fill="red"/>
    <path d="M350 100 L550 100 L550 300 L350 300 Z" fill="#00FF00"/>
    <path d="M600 100 L800 100 L800 300 L600 300 Z" fill="rgb(0,0,255)"/>
    <path d="M100 350 L300 350 L300 550 L100 550 Z" fill="rgba(255,0,255,0.8)"/>
    <path d="M350 350 L550 350 L550 550 L350 550 Z" fill="hsl(60, 100%, 50%)"/>
    <path d="M600 350 L800 350 L800 550 L600 550 Z" fill="rgb(50%,50%,0%)"/>
    <path d="M100 600 L300 600 L300 800 L100 800 Z" fill="#F0F"/>
    <path d="M350 600 L550 600 L550 800 L350 800 Z" fill="navy"/>
    <path d="M600 600 L800 600 L800 800 L600 800 Z" fill="none" stroke="black" stroke-width="5"/>
</svg>'''

    answer_svg = '''<svg viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">
    <path d="M200 300 L350 280 L400 150 L450 280 L600 300 L480 380 L520 520 L400 440 L280 520 L320 380 Z" fill="gold" stroke="orange" stroke-width="5"/>
</svg>'''

    # Create SVG scaler
    scaler = SVGScaler()
    
    # Execute SVG comparison and scaling using new method
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
    
    # Print final SVG for inspection
    print("\nFinal scaled SVG:")
    print(result["finalSvg"])

if __name__ == "__main__":
    main()