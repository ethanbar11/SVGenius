from .compute_LPIPS import LPIPSDistanceCalculator
from .compute_SSIM import SSIMDistanceCalculator
from .compute_fid import FIDCalculator
from .compute_clip_score import CLIPScoreCalculator
from .compute_aesthetic_score import AestheticScoreMetric
from .compute_mse import MSEDistanceCalculator
from .compute_hpsv2 import HPSv2Calculator
from .data_util import rasterize_svg
from .compute_dino_score import DINOScoreCalculator
import json
import logging

class SVGMetrics: 
    def __init__(self, config=None):
        self.class_name = self.__class__.__name__

        default_config = {
            'LPIPS': True,
            'SSIM': True,
            'FID': True,
            'CLIPScore': True,
            'DinoScore': True,
            'AestheticScore': True,
            'MSE': True,
            'HPSv2': True,
        }
        self.config = config or default_config
        self.logger = logging.getLogger(self.class_name)

        self.metrics = {
            'LPIPS': LPIPSDistanceCalculator,
            'SSIM': SSIMDistanceCalculator,
            'FID': lambda: FIDCalculator(model_name='InceptionV3'),
            'CLIPScore': CLIPScoreCalculator,
            'DinoScore': DINOScoreCalculator,
            'AestheticScore': AestheticScoreMetric,
            'MSE': MSEDistanceCalculator,
            'HPSv2': HPSv2Calculator,
        }

        self.active_metrics = {k: v() for k, v in self.metrics.items() if self.config.get(k)}

    def reset(self):
        """Reset all active metrics."""
        for metric in self.active_metrics.values():
            metric.reset()

    def load_json_file(self, json_path):
        """Load data from a JSON file."""
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file {json_path}: {e}")
            raise

    def prepare_batch_from_json_samples(self, samples):
        """
        Prepare a batch from JSON samples.
        Each sample in the JSON file can have:
        1. caption and gen_svg - for CLIP score evaluation
        2. gt_svg and gen_svg - for comparing ground truth and generated SVGs
        """
        batch = {
            'json': samples,
            'gt_svg': [],
            'gen_svg': [],
            'gt_im': [],
            'gen_im': [],
            'caption': []
        }

        for sample in samples:
            # Handle SVG content
            gt_svg = sample.get('gt_svg', None)
            gen_svg = sample.get('gen_svg', None)
            
            # Handle caption for CLIP score
            caption = sample.get('caption', None)
            
            batch['gt_svg'].append(gt_svg)
            batch['gen_svg'].append(gen_svg)
            batch['caption'].append(caption)
            
            # Initialize image placeholders, they'll be rasterized later if needed
            batch['gt_im'].append(None)
            batch['gen_im'].append(None)
            
        return batch

    def process_images_for_metrics(self, batch):
        """Process and rasterize SVGs as needed for metrics calculation."""
        for i in range(len(batch['json'])):
            if batch['gt_svg'][i]:
                batch['gt_im'][i] = rasterize_svg(batch['gt_svg'][i])
            
            if batch['gen_svg'][i]:
                batch['gen_im'][i] = rasterize_svg(batch['gen_svg'][i])
        
        return batch

    def get_sample_id(self, json_item):
        """Extract sample ID from JSON item."""
        return json_item.get('id') or json_item.get('sample_id') or json_item.get('outpath_filename')

    def calculate_metrics_from_json_file(self, json_path, update=True):
        """Calculate metrics from samples in a JSON file."""
        # Load JSON data
        samples = self.load_json_file(json_path)
        return self.calculate_metrics_from_json_samples(samples, update)

    def calculate_metrics_from_json_samples(self, samples, update=True):
        """Calculate metrics from JSON samples directly."""
        # Prepare batch from samples
        batch = self.prepare_batch_from_json_samples(samples)
        
        # Process images as needed
        batch = self.process_images_for_metrics(batch)
        
        # Calculate metrics
        return self.calculate_metrics(batch, update)

    def calculate_metrics(self, batch, update=True):
        """Calculate all active metrics on the provided batch."""
        avg_results_dict = {}
        all_results_dict = {}

        # Initialize all_results_dict
        for i, json_item in enumerate(batch['json']):
            sample_id = self.get_sample_id(json_item)
            if sample_id is None:
                sample_id = f"sample_{i}"
                self.logger.warning(f"Could not find ID for sample at index {i}, using '{sample_id}' instead")
            all_results_dict[sample_id] = {}

        for metric_name, metric in self.active_metrics.items():
            print(f"Calculating {metric_name}...")
            
            # Skip metrics that can't be calculated based on available data
            if metric_name == 'CLIPScore' and not any(batch['caption']):
                self.logger.warning(f"Skipping {metric_name} - no caption available")
                continue
                
            if metric_name in [ 'LPIPS', 'SSIM', 'MSE','DinoScore'] and not all(batch['gt_im']):
                self.logger.warning(f"Skipping {metric_name} - no ground truth images available")
                continue
            
            # Handle metrics that return both average and per-sample results
            if metric_name in ['SSIM', 'CLIPScore', 'LPIPS', 'DinoScore', 'AestheticScore', 'MSE', 'HPSv2']:
                try:
                    avg_result, list_result = metric.calculate_score(batch, update=update)
                    avg_results_dict[metric_name] = avg_result
                    
                    # Store individual results
                    for i, result in enumerate(list_result):
                        sample_id = self.get_sample_id(batch['json'][i])
                        all_results_dict[sample_id][metric_name] = result
                except Exception as e:
                    self.logger.error(f"Error calculating {metric_name}: {e}")
                    avg_results_dict[metric_name] = None
            
            # Handle FID metrics that only return average
            elif metric_name in ['FID']:
                try:
                    # Skip FID if there are not enough samples
                    if len(batch['gen_im']) < 2:
                        self.logger.warning(f"Skipping {metric_name} - need at least 2 samples")
                        continue
                        
                    avg_results_dict[metric_name] = metric.calculate_score(batch)
                except Exception as e:
                    self.logger.error(f"Error calculating {metric_name}: {e}")
                    avg_results_dict[metric_name] = None
            
            # Handle ratio metrics
            else:
                try:
                    self._handle_ratio_metric(metric_name, metric, batch, avg_results_dict, all_results_dict)
                except Exception as e:
                    self.logger.error(f"Error handling ratio metric {metric_name}: {e}")
                    avg_results_dict[metric_name] = None
            
            metric.reset()
            
        print("Average results: \n", avg_results_dict)
        return avg_results_dict, all_results_dict
    
    def calculate_fid_from_json_file(self, json_path):
        """Calculate FID score from a JSON file."""
        samples = self.load_json_file(json_path)
        return self.calculate_fid_from_json_samples(samples)

    def calculate_fid_from_json_samples(self, samples):
        """Calculate FID score from JSON samples directly."""
        batch = self.prepare_batch_from_json_samples(samples)
        batch = self.process_images_for_metrics(batch)
        return self.calculate_fid(batch)

    def calculate_fid(self, batch):
        """Calculate FID score on the provided batch."""
        if 'FID' not in self.active_metrics:
            self.logger.error("FID metric is not active in the configuration.")
            return None
        
        try:
            return self.active_metrics['FID'].calculate_score(batch).item()
        except Exception as e:
            self.logger.error(f"Error calculating FID: {e}")
            return None

    def get_average_metrics(self):
        """Get average values for all metrics."""
        metrics = {}
        for metric_name, metric in self.active_metrics.items():
            if hasattr(metric, 'avg'):
                metrics[metric_name] = metric.avg
            elif hasattr(metric, 'get_average_score'):
                metrics[metric_name] = metric.get_average_score()
        return metrics

    def _handle_ratio_metric(self, metric_name, metric, batch, avg_results_dict, all_results_dict):
        """Helper method to handle ratio-based metrics."""
        metric_key = metric_name.replace('avg_', '').replace('ratio_', '')
        
        for item in batch['json']:
            sample_id = self.get_sample_id(item)
            if metric_key in item:
                value = item[metric_key]
                all_results_dict[sample_id][metric_name] = value
                metric.update(value, 1)
            else:
                self.logger.warning(f"Metric key '{metric_key}' not found in sample {sample_id}")
        
        avg_results_dict[metric_name] = metric.avg