#!/usr/bin/env python3
import difflib
from typing import Dict, Any
from datetime import datetime
from ...eval_util import remove_whitespace, compare_svg, calculate_change_magnitude

def evaluate_svg_repair(model_svg: str, standard_svg: str, bug_svg: str, 
                       execution_time: float = None) -> Dict[str, Any]:
    """
    Evaluate SVG repair accuracy and efficiency
    """
    repair_accuracy = compare_svg(model_svg, standard_svg)
    if repair_accuracy !=1.0:
        repair_accuracy =0.0
    change_magnitude = calculate_change_magnitude(bug_svg, model_svg)
    change_magnitude2 = calculate_change_magnitude(bug_svg, standard_svg)

    model_svg_clean = remove_whitespace(model_svg)
    standard_svg_clean = remove_whitespace(standard_svg)
    model_standard_diff = difflib.ndiff(model_svg_clean, standard_svg_clean)
    diff_count = sum(1 for d in model_standard_diff if d.startswith('+ ') or d.startswith('- '))
    
    model_length = len(model_svg_clean)
    standard_length = len(standard_svg_clean)
    bug_length = len(remove_whitespace(bug_svg))
    
    repair_rate = None
    if execution_time is not None and execution_time > 0:

        repair_rate = len(model_svg) / execution_time
    if repair_accuracy == 1:
        change_magnitude_value = change_magnitude - change_magnitude2
    else:
        change_magnitude_value = 0
    result = {
        "repair_accuracy": repair_accuracy,
        "change_magnitude": change_magnitude_value,
        "repair_efficiency": {
            "execution_time_seconds": execution_time,
            "processing_rate_chars_per_second": repair_rate
        },
        "statistics": {
            "model_svg_length": model_length,
            "standard_svg_length": standard_length,
            "bug_svg_length": bug_length,
            "character_differences": diff_count
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return result
