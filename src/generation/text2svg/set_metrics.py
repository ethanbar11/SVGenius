from typing import Dict, Any
import traceback
from ...metrics.pss.merge import evaluate_svg_string   
from ...metrics.compute_fid import FIDCalculator
from ...metrics.merge_metrics import SVGMetrics

def calculate_releatemse(caption: str, reference_clip: float, generated_clip: float) -> float:
    """Calculate relative CLIP score metric"""
    try:
        if reference_clip <= 0:
            print(f"Reference SVG CLIP score is zero or negative: {reference_clip}")
            return generated_clip 

        clip_diff = reference_clip - generated_clip
        normalized_diff = clip_diff / reference_clip
        releatemse = 1.0 - max(0, normalized_diff)
        
        return releatemse
    
    except Exception as e:
        print(f"Error calculating releatemse: {e}")
        return 0.0 

def calculate_batch_fid(reference_images, generated_images):
    """Calculate FID score for batch of images"""
    try:
        if len(reference_images) < 2 or len(generated_images) < 2:
            print("FID calculation requires at least 2 reference and 2 generated images")
            return None

        batch = {
            'gt_im': reference_images,
            'gen_im': generated_images
        }

        fid_calculator = FIDCalculator(model_name='InceptionV3')
        fid_score, _ = fid_calculator.calculate_score(batch)
        
        return fid_score
        
    except Exception as e:
        print(f"Failed to calculate FID score: {e}")
        print(traceback.format_exc())
        return None
    
def evaluate_generated_svg(caption: str, reference_svg: str, generated_svg: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Evaluate generated SVG using comprehensive metrics
    """
    try:
        path_level_result = evaluate_svg_string(reference_svg, generated_svg, output_dir)
        
        samples = [{
            'caption': caption,
            'gt_svg': reference_svg,
            'gen_svg': generated_svg,
            'id': 'evaluation_sample'
        }]

        text_metrics = SVGMetrics(config={
            'LPIPS': False,
            'SSIM': False,
            'FID': False,
            'CLIPScore': True,
            'DinoScore': False,
            'AestheticScore': True,
            'MSE': False,
            'HPSv2': True,
        })

        avg_results, all_results = text_metrics.calculate_metrics_from_json_samples(samples)
        gen_clip_score = avg_results.get('CLIPScore', 0)

        gen_samples = [{
            'caption': caption,
            'gt_svg': reference_svg,
            'gen_svg': reference_svg,
            'id': 'gen_evaluation_sample'
        }]
        gen_metrics = SVGMetrics(config={
            'LPIPS': False,
            'SSIM': False,
            'FID': False,
            'CLIPScore': True,
            'DinoScore': False,
            'AestheticScore': False,
            'MSE': False,
            'HPSv2': False,
        })
        
        gen_avg_results, _ = gen_metrics.calculate_metrics_from_json_samples(gen_samples)
        ref_clip_score = gen_avg_results.get('CLIPScore', 0)

        releatemse_score = calculate_releatemse(caption, ref_clip_score, gen_clip_score)

        result = {
            'path_level_metrics': path_level_result,  
            'image_metrics': avg_results,          
            'reletiveclip': releatemse_score,           
            'clip_scores': {
                'reference_clip': ref_clip_score,
                'generated_clip': gen_clip_score
            }
        }
        
        return result
    
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print(traceback.format_exc())
        return {
            'error': f"Evaluation failed: {str(e)}",
            'path_level_metrics': {},
            'image_metrics': {},
            'reletiveclip': 0.0,
            'clip_scores': {'reference_clip': 0, 'generated_clip': 0}
        }
