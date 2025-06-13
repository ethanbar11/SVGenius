# -*- coding: utf-8 -*-
from collections import defaultdict
import json
import re
import os
import glob
from tqdm import tqdm
from typing import Tuple, Dict, List
import base64
import time
from openai import AsyncOpenAI
from ...eval_util import setup_logger

logger = setup_logger(name="style_trans_genjson", log_dir="../logs", log_filename="style_trans_genjson.log")

def generate_evaluation_dataset(svg_input_file: str, output_dir: str, model: str) -> None:
    """
    Generate evaluation dataset by pairing original images with style-transferred results
    """
    # Clean model name for file operations
    model_name = model.replace('/', '_').replace(':', '_')
    logger.info(f"Processing style transfer results for model: {model_name}")

    try:
        with open(svg_input_file, "r", encoding='utf-8') as f:
            samples = json.load(f)
        logger.info(f"Loaded {len(samples)} samples from {svg_input_file}")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return

    evaluation_samples = defaultdict(dict)

    # Process input samples
    logger.info("Processing input samples...")
    for i in tqdm(range(len(samples)), desc="Processing samples"):
        sample = samples[i]
        style = sample.get("question", "")
        image_path = sample.get("image_path", "")
        
        if style == "":
            logger.warning(f"Sample {i} missing style information, skipping")
            continue
            
        # Extract image name for matching
        base_name = os.path.basename(image_path)
        image_name = os.path.splitext(base_name)[0]

        evaluation_samples[image_name]["style"] = style
        evaluation_samples[image_name]["original_image_path"] = image_path
        evaluation_samples[image_name]["transferred_image_path"] = ""

    logger.info(f"Created {len(evaluation_samples)} evaluation samples")

    # Scan output directory for transferred images
    logger.info("Scanning output directory for transferred images...")
    svg_output_dir = os.path.join(output_dir)
    subdirs = glob.glob(f"{svg_output_dir}/*/")
    
    if not subdirs:
        logger.warning(f"No subdirectories found in {svg_output_dir}")
        return

    # Match transferred images
    matched_count = 0
    for subdir in tqdm(subdirs, desc="Processing output directories"):
        files = glob.glob(f"{subdir}/*")
        
        # Extract image name from directory (assumes "res_" prefix)
        match = re.search(r"res", subdir)
        if match is not None:
            image_name = subdir[match.end() + 1:-1]
        else:
            logger.warning(f"Could not extract image name from directory: {subdir}")
            continue
        
        # Find transferred image file
        for file in files:
            if file.endswith(f"{model_name}.png"):
                if image_name in evaluation_samples:
                    evaluation_samples[image_name]["transferred_image_path"] = file
                    matched_count += 1
                else:
                    logger.warning(f"Found transferred image for unknown sample: {image_name}")

    logger.info(f"Successfully matched {matched_count} transferred images")

    # Filter complete samples
    complete_samples = {k: v for k, v in evaluation_samples.items() 
                       if v["transferred_image_path"] != ""}
    
    if len(complete_samples) < len(evaluation_samples):
        missing_count = len(evaluation_samples) - len(complete_samples)
        logger.warning(f"{missing_count} samples missing transferred images")

    # Save output dataset
    output_file = os.path.join(output_dir, f"style_trans_samples_{model_name}.json")
    try:
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(complete_samples, f, ensure_ascii=False, indent=2)
        logger.info(f"Generated evaluation dataset: {output_file}")
        logger.info(f"Dataset contains {len(complete_samples)} complete sample pairs")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate SVG Style Transfer Evaluation Dataset')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input JSON file path')
    parser.add_argument('--output-dir', type=str, required=True, 
                       help='Output directory containing style transfer results')
    parser.add_argument('--model', type=str, required=True, 
                       help='Model name used for style transfer')

    args = parser.parse_args()

    try:
        # Generate dataset
        generate_evaluation_dataset(args.input, args.output_dir, args.model)
        
        logger.info("Dataset generation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Dataset generation interrupted by user")
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise

if __name__ == "__main__":
    main()