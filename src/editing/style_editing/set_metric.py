#!/usr/bin/env python3
import os
from typing import Dict, List, Any
from datetime import datetime
import math
from ...metrics.data_util import rasterize_svg
from ...eval_util import calculate_change_magnitude,load_svg_metrics

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

def evaluate_svg_edit(gen_svg: str, gt_svg: str, ori_svg: str, execution_time: float = None, temp_dir: str = "./temp") -> Dict[str, Any]:
    """
    Evaluate SVG editing quality using MSE visual metric and edit distance
    """
    os.makedirs(temp_dir, exist_ok=True)

    gen_edit_distance = calculate_change_magnitude(ori_svg, gen_svg)
    gt_edit_distance = calculate_change_magnitude(ori_svg, gt_svg)

    edit_distance_diff = gen_edit_distance - gt_edit_distance

    result = {
        "edit_distance": {
            "ori_to_gen": gen_edit_distance,
            "ori_to_gt": gt_edit_distance,
            "difference": edit_distance_diff
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if execution_time is not None:
        result["execution_time"] = {
            "seconds": execution_time,
            "processing_rate_chars_per_second": len(gen_svg) / execution_time if execution_time > 0 else None
        }

    svg_metrics = create_svg_metrics_for_mse()
    
    if svg_metrics is not None:
        try:
            gt_im = rasterize_svg(gt_svg)
            gen_im = rasterize_svg(gen_svg)
            ori_im = rasterize_svg(ori_svg)

            batch_gen_gt = {
                "json": [{"sample_id": "gen_vs_gt"}],
                "gt_svg": [gt_svg],
                "gen_svg": [gen_svg],
                "gt_im": [gt_im],
                "gen_im": [gen_im]
            }
            avg_results_gen_gt, _ = svg_metrics.calculate_metrics(batch_gen_gt)

            batch_ori_gt = {
                "json": [{"sample_id": "ori_vs_gt"}],
                "gt_svg": [gt_svg],
                "gen_svg": [ori_svg],
                "gt_im": [gt_im],
                "gen_im": [ori_im]
            }
            avg_results_ori_gt, _ = svg_metrics.calculate_metrics(batch_ori_gt)

            if "MSE" in avg_results_gen_gt and "MSE" in avg_results_ori_gt:
                gen_gt_mse = avg_results_gen_gt["MSE"]
                ori_gt_mse = avg_results_ori_gt["MSE"]

                result["mse"] = {
                    "gen_vs_gt": gen_gt_mse,
                    "ori_vs_gt": ori_gt_mse
                }

                if ori_gt_mse > 0: 
                    mse_ratio = gen_gt_mse / ori_gt_mse
                    custom_metric = math.sqrt(1 - min(mse_ratio, 1))

                    result["custom_metric"] = {
                        "value": custom_metric,
                        "formula": "sqrt(1 - min(mse(gen_svg, gt_svg) / mse(ori_svg, gt_svg), 1))"
                    }
                else:
                    result["custom_metric"] = {
                        "value": None,
                        "error": "MSE between original SVG and target SVG is 0, cannot calculate ratio"
                    }
            else:
                result["visual_error"] = "MSE metric not found in calculation results"
                
        except Exception as e:
            print(f"Error calculating MSE metric: {e}")
            result["visual_error"] = str(e)
    else:
        result["visual_error"] = "SVGMetrics could not be imported, MSE evaluation unavailable"
    
    return result