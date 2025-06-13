import os
import numpy as np
from io import BytesIO
import tempfile
from .validate import  evaluate_svg  
from .normal import SVGScaler 
from .com_iou import calculate_alpha_mask_iou 
from .number import evaluate_path_count  
from .similarity import SVGComparisonSystem, extract_svg_data_from_string, example_usage  
from .sequence import (
    integrate_order_score, 
    calculate_order_similarity_from_feedback
)

import os
import json
import numpy as np
import signal
import time
from io import BytesIO
import tempfile
import traceback
from functools import wraps
from bs4 import BeautifulSoup
def timeout_handler(seconds, default_return=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            class TimeoutError(Exception):
                pass
            
            def handle_timeout(signum, frame):
                raise TimeoutError(f"function {func.__name__} timeout ({seconds}s)")
            
            original_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handle_timeout)
            
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            except TimeoutError as e:
                print(f"warning: {e}")
                return default_return
            except Exception as e:
                print(f"function {func.__name__} error: {e}")
                print(traceback.format_exc())
                return default_return
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
        
        return wrapper
    return decorator
class SVGEvaluationSystem:

    def __init__(self, weights=None):

        # default weight settings
        self.weights = weights or {
            'validity': 0.0,      
            'path_count': 0.0,  
            'global_iou': 0.5,  
            'shape': 0.25,       
            'color': 0.15,        
            'position': 0.15,    
            'order': 0.15       
        }
        
        self.scaler = SVGScaler()
        
        self.comparator = integrate_order_score(SVGComparisonSystem())
 
        self.temp_dir = tempfile.mkdtemp()
    @timeout_handler(60, default_return={"final_reward": 0.0, "valid": False, "error": "evaluation timeout"})    
    def evaluate(self, ref_svg_content, gen_svg_content, output_dir=None, return_visuals=True):
        """
       evaluate the similarity of the generated SVG to the reference SVG with error handling and timeout mechanisms
        """
        try:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = self.temp_dir
   
            result = {
                'valid': False,
                'overall_score': 0,
                'dimension_scores': {},
                'details': {},
                'visuals': {},
                'error': None
            }
 
            print("step 1: SVG validity check")
            try:
                validity_result1 = self._check_validity(gen_svg_content)
                validity_result2 = self._check_validity(ref_svg_content)
                result['valid'] = validity_result1['valid']
                result['details']['validity'] = validity_result1
            except Exception as e:
                print(f"SVG validity check failed: {e}")
                result['error'] = f"validity check failed: {str(e)}"
                return result
  
            if not result['valid'] or validity_result2['valid'] == False:
                print(f"Generated SVG is invalid: {validity_result1['message']} or provided reference SVG is invalid: {validity_result2['message']}")
                result['error'] = f"Invalid SVG: {validity_result1['message']}"
                return result
            
           # Preprocessing: remove empty fill attribute
            try:
                reference_svg = self._process_empty_fills(ref_svg_content)
                generated_svg = self._process_empty_fills(gen_svg_content)
            except Exception as e:
                print(f"failed: {e}")
                reference_svg = ref_svg_content
                generated_svg = gen_svg_content
            
            # Step 3: Pre-processing and standardization of SVGs
            print("Step 3: Pre-processing and standardization of SVGs")
            try:
                preprocessing_result = self._preprocess_svgs(reference_svg, generated_svg, output_dir)
                result['details']['final_gen_svg'] = preprocessing_result['final_gen_svg']
                result['details']['final_gt_svg'] = preprocessing_result['normalized_ref_svg']
                norm_ref_svg = preprocessing_result['normalized_ref_svg']
                norm_gen_svg = preprocessing_result['normalized_gen_svg']
                final_gen_svg = preprocessing_result['final_gen_svg']
            except Exception as e:
                print(f"Failed to preprocess and standardize SVG: {e} - continue with original SVG")
                # 如果预处理失败，使用原始SVG
                norm_ref_svg = reference_svg
                norm_gen_svg = generated_svg
                final_gen_svg = generated_svg
                result['details']['preprocessing'] = {
                    'error': f"pre-processing failed: {str(e)}"
                }
            
            # Step 4: Calculate the global IoU
            print("Step 4: Calculate the global IoU")
            try:
                global_iou_result = self._calculate_global_iou(norm_ref_svg, final_gen_svg, output_dir)
                result['dimension_scores']['global_iou'] = global_iou_result['iou']
                result['details']['global_iou'] = global_iou_result
            except Exception as e:
                print(f"Calculate the global IoU failed: {e}")
                result['dimension_scores']['global_iou'] = 0.0
                result['details']['global_iou'] = {
                    'iou': 0.0,
                    'error': f"Calculate the global IoU failed: {str(e)}"
                }
            
            # Step 5: Assess the number of paths
            print("Step 5: Assess the number of paths")
            try:
                path_count_result = self._evaluate_path_count(norm_ref_svg, final_gen_svg)
                result['dimension_scores']['path_count'] = path_count_result['score']
                result['details']['path_count'] = path_count_result
            except Exception as e:
                print(f" Assess the number of paths failed: {e}")
                result['dimension_scores']['path_count'] = 0.0
                result['details']['path_count'] = {
                    'score': 0.0,
                    'error': f" Assess the number of paths failed: {str(e)}"
                }
            
            # Step 6: Perform path-level comparisons
            print("Step 6: Perform path-level comparisons")
            try:
                comparison_result = self._compare_svgs(norm_ref_svg, final_gen_svg, output_dir)
                for key, value in comparison_result['dimension_scores'].items():
                    result['dimension_scores'][key] = value
                
                result['details']['path_comparison'] = comparison_result
            except Exception as e:
                print(f"Perform path-level comparisons failed: {e}")
                result['details']['path_comparison'] = {
                    'error': f"Perform path-level comparisons failed: {str(e)}",
                    'feedback': [],
                    'dimension_scores': {}
                }

            # Step 7: Calculate path order similarity
            print("Step 7: Calculate path order similarity")
            try:
                order_result = self._calculate_order_score(result['details'].get('path_comparison', {}))
                result['dimension_scores']['order'] = order_result['score']
                result['details']['order'] = order_result
            except Exception as e:
                print(f"Calculate path order similarity failed: {e}")
                result['dimension_scores']['order'] = 0.0
                result['details']['order'] = {
                    'score': 0.0,
                    'error': f"Calculate path order similarity failed: {str(e)}"
                }
            
            # Step 8: Calculate weighted total score
            print("Step 8: Calculate weighted total score")
            try:
                total_score = self._calculate_total_score(result['dimension_scores'])
                result['overall_score'] = int(total_score)
            except Exception as e:
                print(f"Calculate weighted total score failed: {e}")
                result['overall_score'] = 0
                result['error'] = f"Calculate weighted total score failed: {str(e)}"
            
            print(f"Assessment completed, total score: {result['overall_score']}/100")
            
            try:
                final_reward = self.calculate_final_reward(result)
                result['final_reward'] = final_reward
                print(f"final reward: {final_reward:.4f}")
            except Exception as e:
                print(f"Failure to calculate the final reward: {e}")
                result['final_reward'] = 0.0
                result['error'] = f"Failure to calculate the final reward: {str(e)}"
            # result=result.pop('details')
            final={
                'final_reward': result['final_reward'],
                # 'overall_score': result['overall_score'],
                'dimension_scores': result['dimension_scores'],
                'valid': result['valid'],
                'visuals': result['visuals'],
                'error': result['error'],
                'details': {
                    'validity': result['details']['validity'],
                    'global_iou': result['details'].get('global_iou', {}),
                    'path_count': result['details'].get('path_count', {}),
                    # 'path_comparison': result['details'].get('path_comparison', {}),
                    'order': result['details'].get('order', {}),
                    'final_gen_svg': result['details'].get('final_gen_svg', {}),
                    'final_gt_svg': result['details'].get('final_gt_svg', {}),
                },
                
            }
            return final
        except Exception as e:
            error_msg = f"Unaddressed anomalies in the assessment process: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return {
                'valid': False,
                'overall_score': 0,
                'dimension_scores': {},
                'details': {},
                'visuals': {},
                'error': error_msg,
                'final_reward': 0.0
            }
    def _calculate_order_score(self, comparison_result):
        """
        Calculate the similarity of the order in which paths are drawn
        """
        feedback_list = comparison_result.get('feedback', [])
  
        if not feedback_list:
            return {
                'score': 0.0,
                'details': 'no matching paths to calculate sequential similarity'
            }
      
        order_score = calculate_order_similarity_from_feedback(feedback_list)
        
        result = {
            'score': order_score,
            'matched_paths_count': len(feedback_list),
            'details': f'Order Similarity Score: {order_score:.4f}'
        }
    
        return result
    def _process_empty_fills(self, svg_string):
        """
        Handles empty fill attributes       
        """
        try:

           
            if not svg_string or not isinstance(svg_string, str) or not svg_string.strip().startswith('<'):
                return svg_string

            soup = BeautifulSoup(svg_string, 'xml')

            svg_element = soup.find('svg')
            if not svg_element:
                return svg_string

            path_elements = svg_element.find_all('path')

            if not path_elements:
                return svg_string

            fill_removed = False
            for path in path_elements:
                if path.has_attr('fill'):
                    if path['fill'] == '':
                        del path['fill']
                        fill_removed = True
            
            if fill_removed:
                print("remove one or more empty fill attributes")

            return str(soup)
        
        except Exception as e:
            print(f"error: {e}")
            return svg_string
    def _check_validity(self, svg_string):
        """
        Check if SVG is valid
        """
      
        svg_content = svg_string
       
        return evaluate_svg(svg_content)
    
    def _prepare_svg_files(self, reference_svg, generated_svg):
       
        if os.path.isfile(reference_svg):
            ref_path = reference_svg
        else:
            ref_path = os.path.join(self.temp_dir, "reference.svg")
            with open(ref_path, 'w') as f:
                f.write(reference_svg)
        
        if os.path.isfile(generated_svg):
            gen_path = generated_svg
        else:
            gen_path = os.path.join(self.temp_dir, "generated.svg")
            with open(gen_path, 'w') as f:
                f.write(generated_svg)
        
        return ref_path, gen_path
    
    def _preprocess_svgs(self, reference_svg, generated_svg, output_dir):
        """
        Pre-processing and standardized SVG
       
        """
        scaling_result = self.scaler.compare_and_scale_svg(
            generated_svg, 
            reference_svg,
            preserve_aspect_ratio=True,
            output_file=os.path.join(output_dir, "scaling_comparison.png"),
            show_plot=False
        )
        return {
            'normalized_ref_svg': scaling_result['normalizedAnswerSvg'],
            'normalized_gen_svg': scaling_result['normalizedGenSvg'],
            'final_gen_svg': scaling_result['finalSvg'],
            'ratios': scaling_result['ratios'],
            'visualization_path': None
        }
    
    def _calculate_global_iou(self, reference_svg, generated_svg, output_dir):
        """
        Calculate global IoU
        
        """
        iou = calculate_alpha_mask_iou(reference_svg, generated_svg,output_path=os.path.join(output_dir, "global_iou.png"))
        
        return {
            'iou': iou,
            'visual_path': None
        }
    
    def _evaluate_path_count(self, ref_svg, gen_svg):
        """
        Number of assessment paths
        
        """
        # ref_paths = extract_svg_data_with_svgpathtools(ref_path)
        # gen_paths = extract_svg_data_with_svgpathtools(gen_path)
        ref_paths = extract_svg_data_from_string(ref_svg)
        gen_paths = extract_svg_data_from_string(gen_svg)
   
        ref_count = len(ref_paths)
        gen_count = len(gen_paths)
        
        count_penalty = evaluate_path_count(ref_count, gen_count)
        count_score = 1.0 - count_penalty
        
        return {
            'reference_count': ref_count,
            'generated_count': gen_count,
            'difference': gen_count - ref_count,
            'penalty': count_penalty,
            'score': count_score,
            'match_percentage': f"{(count_score * 100):.1f}%"
        }
    
    def _compare_svgs(self, ref_svg, gen_svg, output_dir):
        """
        Performs path-level SVG comparisons
       
        """
        output_path = os.path.join(output_dir, "path_comparison.png")
        
        comparison_result = example_usage(ref_svg, gen_svg, target_size=(256, 256), output_path=output_path)
        
        comparison_result['visual_path'] = output_path
        
        return comparison_result
    def calculate_final_reward(self, result):
        """
        Calculate final reward based on evaluation results
        
        """
        if not result['valid']:
            return 0.0
  
        GLOBAL_THRESHOLD = 0.5
        IOU_THRESHOLD = 0.5
        COLOR_THRESHOLD = 0.9
        POSITION_THRESHOLD = 0.9
        ORDER_THRESHOLD = 1
        
        DETAIL_WEIGHTS = {
            'iou': 0.8,    
            'color': 0.1,  
            'position': 0.1  
        }
        
        global_iou = result['dimension_scores'].get('global_iou', 0.0)
        if global_iou < GLOBAL_THRESHOLD:
            print(f"Global IoU below threshold: {global_iou:.4f}, apply penalty")
            global_iou = 0.0
        path_count_score = result['dimension_scores'].get('path_count', 0.0)
        order_score = result['dimension_scores'].get('order', 0.0)
   
        path_comparison = result['details'].get('path_comparison', {})
        feedback_list = path_comparison.get('feedback', [])
        
        if not feedback_list:
            print("No matching paths, using global IoU and number of paths scoring")
            detail_reward = 0.0
        else:
            total_iou = 0.0
            total_color = 0.0
            total_position = 0.0
            valid_iou_count = 0
            valid_color_count = 0
            valid_position_count = 0
            
            for match in feedback_list:
                path_iou = match.get('shape_score', 0.0)
                path_color = match.get('color_score', 0.0)
                path_position = match.get('position_score', 0.0)
               
                if path_iou >= IOU_THRESHOLD:
                    total_iou += path_iou
                    valid_iou_count += 1
                
                if path_color >= COLOR_THRESHOLD:
                    total_color += path_color
                    valid_color_count += 1
                
                if path_position >= COLOR_THRESHOLD:
                    total_position += path_position
                    valid_position_count += 1
                # total_position += path_position
           
            avg_iou = total_iou / max(valid_iou_count, 1)
            # print
            avg_color = total_color / max(valid_color_count, 1)
            avg_position = total_position / max(valid_position_count, 1)
            
            detail_components = {}
            valid_weights_sum = 0.0
            
            # IoU component
            if valid_iou_count > 0:
                detail_components['iou'] = avg_iou * DETAIL_WEIGHTS['iou']
                valid_weights_sum += DETAIL_WEIGHTS['iou']
            else:
                detail_components['iou'] = 0.0
                
            # Color component
            if valid_color_count > 0:
                detail_components['color'] = avg_color * DETAIL_WEIGHTS['color']
                valid_weights_sum += DETAIL_WEIGHTS['color']
            else:
                detail_components['color'] = 0.0
                
            # Position component
             # Color component
            if valid_position_count > 0:
                detail_components['position'] = avg_position * DETAIL_WEIGHTS['position']
                valid_weights_sum += DETAIL_WEIGHTS['position']
            else:
                detail_components['position'] = 0.0
            # detail_components['position'] = avg_position * DETAIL_WEIGHTS['position']
            # valid_weights_sum += DETAIL_WEIGHTS['position']
            
            detail_reward = sum(detail_components.values())
            # if valid_weights_sum > 0:
            #     detail_reward /= valid_weights_sum
        
        if detail_reward + global_iou > 0:
            # combined_reward = 2 * (detail_reward * global_iou) / (detail_reward + global_iou)
            combined_reward = 0.5*detail_reward + 0.5*global_iou
        else:
            combined_reward = 0.0
        middle_reward = combined_reward
        
        path_count_details = result['details'].get('path_count', {})
        if path_count_details.get('difference', -1) == 0:
             combined_reward=combined_reward
        else:
            path_count_penalty = path_count_details.get('penalty', 0.0)
            combined_reward -=path_count_penalty*0.2 
        if order_score < ORDER_THRESHOLD:
            order_penalty = (ORDER_THRESHOLD - order_score) 
            combined_reward -= order_penalty
        else:
            combined_reward=combined_reward
        final_reward = combined_reward
        if final_reward < 0:
            final_reward = 0.0
        elif final_reward > 1:
            final_reward = 1.0
        
        return final_reward
    def _calculate_total_score(self, dimension_scores):
        """
        calculate weighted total score

        """
        total_score = 0
        weight_sum = 0
        
        for dim, score in dimension_scores.items():
            if dim in self.weights:
                weight = self.weights[dim]
                total_score += score * weight * 100
                weight_sum += weight
        
        if weight_sum == 0:
            return 0
        
        final_score = total_score / weight_sum
        
        return final_score

def evaluate_svg_files(reference_svg_path, generated_svg_path, output_dir="./results"):
    evaluator = SVGEvaluationSystem()
    result = evaluator.evaluate(reference_svg_path, generated_svg_path, output_dir)
    return result

def evaluate_svg_string(reference_svg, generated_svg, output_dir="./results"):
    evaluator = SVGEvaluationSystem()
    result = evaluator.evaluate(reference_svg, generated_svg, output_dir)
    return result


if __name__ == "__main__":
    import argparse
    gen_svg="""<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><path d="M132.129032 0h652.71742a132.129032 132.129032 0 0 1 87.997935 33.560774l140.056774 124.994065A132.129032 132.129032 0 0 1 1057.032258 257.123097v622.030451a132.129032 132.129032 0 0 1-132.129032 132.129033H132.129032a132.129032 132.129032 0 0 1-132.129032-132.129033V132.129032a132.129032 132.129032 0 0 1 132.129032-132.129032z" fill="#2D6AEA" />
</svg>"""
    ref_svg="""<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><path d="M132.129032 0h652.71742a132.129032 132.129032 0 0 1 87.997935 33.560774l140.056774 124.994065A132.129032 132.129032 0 0 1 1057.032258 257.123097v622.030451a132.129032 132.129032 0 0 1-132.129032 132.129033H132.129032a132.129032 132.129032 0 0 1-132.129032-132.129033V132.129032a132.129032 132.129032 0 0 1 132.129032-132.129032z" fill="#2D6AEA" />
</svg>"""
    results=evaluate_svg_string(ref_svg, gen_svg)
    print("results:")
    print(results)
   