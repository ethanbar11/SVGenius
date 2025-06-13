#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import sys
import json
import logging
import re
import time
import shutil
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from openai import OpenAI, AsyncOpenAI
import traceback
from ...eval_util import setup_logger, extract_svg_from_response

logger = setup_logger(name="style_trans", log_dir="../logs", log_filename="style_trans.log")

# API Configuration
API_KEY = "your_api_key_here"  
BASE_URL = "your_base_url_here"
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

async def generate_svg_from_api(style: str, reference_svg: str, description: str, model: str = "deepseekr1") -> Tuple[
    Optional[str], float, Optional[str]]:
    """
    Generate SVG style transfer using API
  
    """
    start_time = time.time()

    try:
        system_prompt = (
            """You are a professional SVG designer with extensive experience in vector graphics creation. 
            Your task is to perform a style transfer - recreate the provided reference SVG.
            Maintain the basic structure and given semantic description of the SVG, but adjust it to the desired style.
            Provide your answer strictly in the following format: Answer:{SVG code}, without adding any explanations, comments, or other content."""
        )

        user_prompt = f"Reference SVG to transform:\n{reference_svg}\n\nDescription of the reference SVG:\n{description}\n\nthe style to transfer {style}"

        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
        )
        
        response_text = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_text += content 

        execution_time = time.time() - start_time
        svg_code = extract_svg_from_response(response_text)

        return svg_code, execution_time, response_text

    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None, time.time() - start_time, None

async def generate_svg(output_dir, index, sample, model):
    """Generate single SVG style transfer"""
    question = sample.get('question', '')
    description = sample.get('description', [''])[0] if isinstance(sample.get('description'), list) else sample.get(
        'question', '')
    gt_svg_content = sample.get('code', '')
    gt_svg_path = sample.get('image_path', '')

    style_list = ["Cartoon Style", "Pixel-art", "Line-art", "3D-style"]
    logging.info(question)

    # Validation checks
    if not question:
        logger.warning(f"Sample {index + 1} missing question description, skipping")
        return

    if not gt_svg_content and not gt_svg_path:
        logger.warning(f"Sample {index + 1} has no SVG content or path, skipping")
        return

    if question not in style_list:
        logger.warning(f"Sample {index + 1} missing transfer style, skipping")
        return

    # Load SVG content if needed
    if not gt_svg_content and gt_svg_path:
        try:
            with open(gt_svg_path, 'r', encoding='utf-8') as f:
                gt_svg_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read SVG file {gt_svg_path}: {e}")
            return

    # Determine output filename
    if gt_svg_path:
        svg_basename = os.path.basename(gt_svg_path)
        svg_filename = os.path.splitext(svg_basename)[0]
    else:
        svg_filename = f"sample_{index + 1}"

    sample_output_dir = os.path.join(output_dir, svg_filename)
    model_name = model.replace(':', '_').replace('/', '_')
    gen_svg_output_path = os.path.join(sample_output_dir, f"{model_name}.svg")

    # Skip if already exists
    if os.path.exists(gen_svg_output_path):
        logger.warning(f"Sample {gt_svg_path} already exists, skipping")
        return

    # Generate SVG
    svg, exec_time, response_text = await generate_svg_from_api(question, gt_svg_content, description, model)
    return svg, exec_time, response_text, gen_svg_output_path

async def process_svg_style_transfer(input_file: str, output_file: str, output_dir: str, thread_num: int,
                                     model: str = "deepseekr1") -> Dict[str, Any]:
    """
    Process SVG style transfer for batch of samples
   
    """
    # Load input data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            samples_data = json.load(f)

        if not isinstance(samples_data, list):
            samples_data = [samples_data]

    except Exception as e:
        logger.error(f"Failed to load JSON file {input_file}: {e}")
        return {"error": f"Failed to load JSON file: {e}"}

    if not samples_data:
        return {"error": "Sample data is empty"}

    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(samples_data), thread_num)):
        batch = samples_data[i:i + thread_num]
        tasks = [
            generate_svg(output_dir, i + j, sample, model)
            for j, sample in enumerate(batch)
        ]
        batch_responses = await asyncio.gather(*tasks)
        
        # Process batch results
        for i, response in enumerate(batch_responses):
            if response is None:
                continue
                
            generated_svg, execution_time, full_response, gen_svg_output_path = response
            description = batch[i].get('description', [''])[0] if isinstance(batch[i].get('description'), list) else (
                batch[i].get('description', ''))
            question = batch[i].get('question', '')
            gt_svg_content = batch[i].get('code', '')
            gt_svg_path = batch[i].get('image_path', '')

            # Handle generation failure
            if not generated_svg:
                result = {
                    'description': description,
                    'gt_svg_path': gt_svg_path,
                    'gt_svg': gt_svg_content,
                    'question': question,
                    'error': 'SVG generation failed',
                    'execution_time': execution_time,
                    'full_response': full_response
                }
                results.append(result)
                continue

            # Setup output paths
            if gt_svg_path:
                svg_basename = os.path.basename(gt_svg_path)
                svg_filename = os.path.splitext(svg_basename)[0]
            else:
                svg_filename = f"sample_{i + 1}"

            sample_output_dir = os.path.join(output_dir, svg_filename)
            os.makedirs(sample_output_dir, exist_ok=True)

            gt_svg_output_path = os.path.join(sample_output_dir,
                                              svg_basename if svg_basename else f"{svg_filename}.svg")

            # Save ground truth SVG
            if gt_svg_path and os.path.exists(gt_svg_path):
                shutil.copy2(gt_svg_path, gt_svg_output_path)
            else:
                with open(gt_svg_output_path, 'w', encoding='utf-8') as f:
                    f.write(gt_svg_content)

            # Save generated SVG
            with open(gen_svg_output_path, 'w', encoding='utf-8') as f:
                f.write(generated_svg.replace('\n', ''))

            result = {
                'description': description,
                'gt_svg_path': gt_svg_path,
                'gt_svg': gt_svg_content,
                'question': question,
                'generated_svg': generated_svg,
                'execution_time': execution_time,
                'full_response': full_response,
                'output_paths': {
                    'gt_svg': gt_svg_output_path,
                    'gen_svg': gen_svg_output_path
                }
            }
            results.append(result)

    # Generate summary
    summary = {
        "total_samples": len(samples_data),
        "processed_samples": len(results),
        "successful_samples": len([r for r in results if 'error' not in r]),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model
    }

    full_result = {
        "summary": summary,
        "results": results
    }

    # Save results
    try:
        model_name = model.replace('/', '_').replace(':', '_')
        with open(f"{output_file}_{model_name}", 'w', encoding='utf-8') as f:
            json.dump(full_result, f, indent=2, ensure_ascii=False)
        logger.info(f"Processing results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {e}")

    return summary

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='SVG Style Transfer')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output results JSON file path')
    parser.add_argument('--output-dir', type=str, required=True, help='Output SVG files directory')
    parser.add_argument('--model', type=str, default="Qwen2.5-72B-Instruct", help='Model name to use')
    parser.add_argument('--thread_num', type=int, default=1, help='Maximum concurrency')

    args = parser.parse_args()

    try:
        summary = asyncio.run(process_svg_style_transfer(
            args.input,
            args.output,
            args.output_dir,
            args.thread_num,
            args.model,
        ))

        if "error" in summary:
            logging.info(f"Processing failed: {summary['error']}")
        else:
            logging.info(f"Processing complete, {summary['total_samples']} total samples, {summary['successful_samples']} successful")
            logging.info(f"Detailed results saved to: {args.output}")
            logging.info(f"SVG files saved to: {args.output_dir}")

    except Exception as e:
        logging.info(f"Error occurred: {e}")
        logging.info(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()