import traceback
from ...metrics.pss.merge import evaluate_svg_string   
from ...metrics.compute_fid import FIDCalculator
from ...metrics.merge_metrics import SVGMetrics
from ...metrics.data_util import rasterize_svg

async def evaluate_generated_svg(caption: str, reference_svg: str, generated_svg: str, output_dir: str = None) -> Dict[str, Any]:
    """
    evaluate similarity between generated SVG and reference SVG (metrics excluding FID)

    """
    try:
        # Evaluate using path-level evaluation function
        path_level_result = evaluate_svg_string(reference_svg, generated_svg, output_dir)
        
        # Prepare sample data for image-level evaluation (excluding FID)
        samples = [{
            'caption': caption,
            'gt_svg': reference_svg,
            'gen_svg': generated_svg,
            'id': 'evaluation_sample'
        }]
        
        # Initialize SVG image-level metrics evaluator
        image_metrics = SVGMetrics(config={
            'LPIPS': True,
            'SSIM': True,
            'FID': False,
            'CLIPScore': False,
            'DinoScore': True,
            'AestheticScore': False,
            'MSE': True,
            'HPSv2': False,
        })
        
        # Calculate image-level metrics
        avg_results, all_results = image_metrics.calculate_metrics_from_json_samples(samples)
        
        # Build results
        result = {
            'path_level_metrics': path_level_result,  # Path-level evaluation results
            'image_metrics': avg_results,             # Image-level evaluation results
        }
        
        return result
    
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print(traceback.format_exc())
        return {
            'error': f"Evaluation failed: {str(e)}",
            'path_level_metrics': {},
            'image_metrics': {}
        }
    

def calculate_batch_fid(reference_images, generated_images):
    """
    Calculate batch FID score
    
    """
    try:
        # Check image count
        if len(reference_images) < 2 or len(generated_images) < 2:
            print("FID calculation requires at least 2 reference and 2 generated images")
            return None
            
        # Create batch data
        batch = {
            'gt_im': reference_images,
            'gen_im': generated_images
        }
        
        # Create FID calculator and compute score
        fid_calculator = FIDCalculator(model_name='InceptionV3')
        fid_score, _ = fid_calculator.calculate_score(batch)
        
        return fid_score
        
    except Exception as e:
        print(f"Failed to calculate FID score: {e}")
        print(traceback.format_exc())
        return None