#!/usr/bin/env python3
import os
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from metrics.data_util import rasterize_svg
from eval_util import load_svg_metrics

def calculate_compression_ratio(ori_svg: str, gen_svg: str) -> float:
    """
    Calculate SVG compression ratio
    """
    ori_size = len(ori_svg.encode('utf-8'))
    gen_size = len(gen_svg.encode('utf-8'))
    
    savings_percent = (gen_size / ori_size) if ori_size > 0 else 0
    
    return savings_percent

def create_svg_metrics_for_mse():
    """
    Create an SVGMetrics instance configured only for MSE
    
    """
    SVGMetricsClass = load_svg_metrics()
    if SVGMetricsClass is None:
        return None
    
    config = {
        'LPIPS': False,
        'SSIM': False,
        'FID': False,
        'CLIPScore': False,
        'DinoScore': False,
        'AestheticScore': False,
        'MSE': True,
        'HPSv2': False,
    }
    
    try:
        return SVGMetricsClass(config=config)
    except Exception as e:
        print(f"Failed to create SVGMetrics instance: {e}")
        return None

def evaluate_svg_optimization(ori_svg: str, gen_svg: str, execution_time: float = None, opti_ratio: float = None, temp_dir: str = "./temp") -> Dict[str, Any]:
    """
    Evaluate SVG optimization quality using MSE visual metric and compression ratio
    
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    compression_ratio = calculate_compression_ratio(ori_svg, gen_svg)
    
    result = {
        "compression": {
            "ratio_percent": 1-compression_ratio,  
            "size_reduction_percent": 1 - compression_ratio 
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if execution_time is not None:
        result["execution_time_seconds"] = execution_time

    if opti_ratio is not None:
        relative_ratio = opti_ratio-compression_ratio
        result["compression"]["target_ratio_percent"] = 1-opti_ratio
        result["compression"]["relative_ratio"] = relative_ratio 

    svg_metrics = create_svg_metrics_for_mse()
    
    if svg_metrics is not None:
        try:
            gt_im=rasterize_svg(ori_svg)
            gen_im=rasterize_svg(gen_svg)
            batch = {
                "json": [{"sample_id": "current_sample"}],
                "gt_svg": [ori_svg],    
                "gen_svg": [gen_svg],   
                "gt_im": [gt_im],       
                "gen_im": [gen_im]    
            }

            avg_results, all_results = svg_metrics.calculate_metrics(batch)

            if "MSE" in avg_results:
                result["mse"] = avg_results["MSE"]
            else:
                result["visual_error"] = "MSE metric not found in results"
                
        except Exception as e:
            print(f"Error calculating MSE metric: {e}")
            result["visual_error"] = str(e)
    else:
        result["visual_error"] = "SVGMetrics could not be imported, MSE evaluation unavailable"
    
    return result