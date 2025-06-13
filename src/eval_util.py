
#!/usr/bin/env python3
import re
import difflib
import os
import logging
from datetime import datetime
from typing import Optional, List
from openai import AsyncOpenAI
import base64
import traceback
def remove_whitespace(svg_content: str) -> str:
    """
    Remove all whitespace characters from the SVG content.
    """
    return re.sub(r'\s+', '', svg_content)

def compare_svg(svg1: str, svg2: str) -> float:
    """
    Compare the similarity between two SVG strings
    """
    svg1_no_space = remove_whitespace(svg1)
    svg2_no_space = remove_whitespace(svg2)

    return difflib.SequenceMatcher(None, svg1_no_space, svg2_no_space).ratio()

def calculate_change_magnitude(source_svg: str, target_svg: str) -> float:
    """
    Calculate the degree of change between SVGs, removing the influence of SVG length
    """
    source_svg_clean = remove_whitespace(source_svg)
    target_svg_clean = remove_whitespace(target_svg)
    
    edit_distance = 0
    matcher = difflib.SequenceMatcher(None, source_svg_clean, target_svg_clean)
    
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op != 'equal':
            edit_distance += max(i2 - i1, j2 - j1)
    
    return edit_distance


def setup_logger(name: str = None, 
                log_dir: str = "../logs", 
                log_filename: str = None,
                level: int = logging.INFO) -> logging.Logger:
    """
    Setup a universal logger for any module
    
    """
    os.makedirs(log_dir, exist_ok=True)
    
    if log_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"svg_evaluation_{timestamp}.log"
    
    log_file_path = os.path.join(log_dir, log_filename)
    
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'svgenius')
    
    logger = logging.getLogger(name)
    
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.setLevel(level)
    logger.propagate = False
    
    return logger

def extract_svg_from_response(response_text: str) -> Optional[str]:
    """
    Extracts the SVG code from the model response and returns the last matching SVG
    """
    answer_pattern = r'Answer:\s*(<svg[\s\S]*?<\/svg>)'
    answer_matches = re.findall(answer_pattern, response_text)
    
    if answer_matches:
        return answer_matches[-1]  
    
    svg_pattern = r'(<svg[\s\S]*?<\/svg>)'
    svg_matches = re.findall(svg_pattern, response_text)
    
    if svg_matches:
        return svg_matches[-1]  
    
    return None


def load_svg_metrics():
    """
    Attempt to import SVGMetrics class
    """
    try:
        from .metrics.merge_metrics import SVGMetrics
        print("Successfully imported SVGMetrics from standard path")
        return SVGMetrics
    except Exception as e:
        print("Failed to import SVGMetrics:", e)
        traceback.print_exc()
        return None
    
def encode_image_to_base64(image_path):
    """
    Convert image to base64 encoding

    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to convert image to base64: {e}")
        return None
