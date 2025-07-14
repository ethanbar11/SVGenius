#!/usr/bin/env python3
import os
import re
import json
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from openai import AsyncOpenAI
from editing.bug_fixing.set_metric import evaluate_svg_repair
from eval_util import setup_logger, extract_svg_from_response 

logger = setup_logger(name="bug_fixing", log_dir="../logs", log_filename="bug_fixing.log")

API_KEY = "mock-key-123"  
BASE_URL = "http://localhost:8000/v1"
AVAILABLE_MODELS = ["Qwen2-72B-Instruct-AWQ", "gpt-4o", "deepseekr1","mock-llm"]

async def generate_svg_from_api(bug_svg: str, model: str = "deepseekr1", semaphore: asyncio.Semaphore = None) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Using the API to generate a fixed SVG based on bug_svg
    """
    start_time = time.time()
    
    system_prompt = (
        '''You are a professional SVG repair engineer with expertise in SVG standards and common
          error types. Your task is to analyze submitted SVG code, precisely locate errors, 
          and fix problems using the principle of minimal modification. Please only modify the
         parts causing errors while keeping the rest of the code unchanged. After fixing, return 
         the complete corrected code in the following strict format: Answer:{SVG code}, providing
         the complete fixed code only in the {SVG code} position, without adding any explanations,
          comments, or other content.'''
    )
    aclient = AsyncOpenAI(
                        base_url=BASE_URL,
                        api_key=API_KEY,
                    )
    async def execute_api_call():
        try:
            retries = 0
            max_retries = 10

            while retries < max_retries:
                try:
                    stream = await aclient.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": bug_svg}
                        ],
                        stream=True,
                    )
                    response = ""
                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            response+= content 
                    execution_time = time.time() - start_time
                    
                    response_text = response

                    svg_code = extract_svg_from_response(response_text)
                    if not svg_code:
                        logger.error(f"Failed to extract SVG code from response: {response_text}")
                    else:                    
                        logger.info(f"SVG code extracted successfully, length: {len(svg_code)} characters.")
                    return svg_code, execution_time, response_text
                    
                except Exception as e:
                    retries += 1
                    logger.error(f"API call failed: {e}, retry at {retries}")
                    if retries >= max_retries:
                        logger.error(f"Maximum number of retries exceeded")
                        return None, time.time() - start_time, "request error"
                    await asyncio.sleep(10) 
        
        except Exception as e:
            logger.error(f"Exception during API call: {e}")
            return None, time.time() - start_time, None

    if semaphore:
        async with semaphore:
            return await execute_api_call()
    else:
        return await execute_api_call()
    

async def process_single_sample(sample_id: str, sample_data: Dict[str, Any], model: str, semaphore: asyncio.Semaphore = None) -> Dict[str, Any]:
    
    bug_svg = sample_data.get("bug_svg", "")
    gt_svg = sample_data.get("ground_truth", "")
    
    if not bug_svg:
        return {
            "sample_id": sample_id,
            "error": "Missing bug_svg"
        }
    
    if not gt_svg:
        logger.warning(f"Sample {sample_id} missing ground_truth")
    
    logger.info(f"process sample {sample_id}...")
    
    gen_svg, execution_time, full_response = await generate_svg_from_api(bug_svg, model, semaphore)
    
    if not gen_svg:
        return {
            "sample_id": sample_id,
            "error": "SVG repair failed",
            "execution_time": execution_time,
            "full_response": full_response,
            "bug_svg": bug_svg,
            "gt_svg": gt_svg
        }
    
    evaluation_result = None
    if gt_svg and gen_svg:
        evaluation_result = evaluate_svg_repair(
            model_svg=gen_svg,
            standard_svg=gt_svg,
            bug_svg=bug_svg,
            execution_time=execution_time
        )
    
    result = {
        "sample_id": sample_id,
        "bug_svg": bug_svg,
        "gt_svg": gt_svg,
        "gen_svg": gen_svg,
        "execution_time": execution_time,
        "full_response": full_response
    }
    
    if evaluation_result:
        result["evaluation"] = evaluation_result
    
    return result

async def process_batch_from_json(json_file: str, output_file: str, model: str = "deepseekr1", max_concurrent: int = 5) -> Dict[str, Any]:

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            samples_data = json.load(f)
            
        if not isinstance(samples_data, list):
            logger.error(f"JSON data must be in list format")
            return {"error": "JSON data format error, should be list"}
        samples = samples_data

    except Exception as e:
        logger.error(f"load json file {json_file} failed: {e}")
        return {"error": f"load json file failed: {e}"}
    
    if not samples:
        return {"error": "samples list is empty"}
    
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    sample_ids = [f"sample_{i+1}" for i in range(len(samples))]
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    results = []
    total_samples = len(samples)
    processed = 0
    
    batch_size = max_concurrent * 2 
    
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_ids = sample_ids[start_idx:end_idx]
        batch_samples = samples[start_idx:end_idx]
        
        logger.info(f"processing batch {start_idx//batch_size + 1}, sample {start_idx+1}-{end_idx}...")
        
        batch_tasks = [
            process_single_sample(sample_id, sample, model, semaphore)
            for sample_id, sample in zip(batch_ids, batch_samples)
        ]

        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        processed += len(batch_results)
        
        interim_results = {
            result["sample_id"]: result
            for result in results
            if "sample_id" in result
        }
        
        interim_output = f"{os.path.splitext(output_file)[0]}_interim_{processed}.json"
        with open(interim_output, 'w', encoding='utf-8') as f:
            json.dump(interim_results, f, indent=2, ensure_ascii=False)
        logger.info(f"{processed}/{total_samples} samples have been processed, intermediate results have been saved to: {interim_output}")
    
    detailed_results = {}
    for result in results:
        if "sample_id" in result:
            detailed_results[result["sample_id"]] = result
    
    successful_samples = [r for r in results if "error" not in r]
    sample_count = len(successful_samples)
    
    perfect_repair_count = 0
    total_change_magnitude_perfect = 0.0

    total_execution_time = 0
    total_repair_accuracy = 0
    total_change_magnitude = 0
    total_model_svg_length = 0
    total_standard_svg_length = 0
    total_bug_svg_length = 0
    total_character_differences = 0
    total_processing_rate = 0
    evaluation_count = 0
    
    accuracy_ranges = {
        "excellent (>= 0.9)": 0,
        "good (0.8-0.9)": 0,
        "fair (0.6-0.8)": 0,
        "poor (< 0.6)": 0
    }
    
    for result in successful_samples:
        total_execution_time += result["execution_time"]
        
        if "evaluation" in result and "repair_accuracy" in result["evaluation"]:
            eval_data = result["evaluation"]
            total_repair_accuracy += eval_data["repair_accuracy"]
            total_change_magnitude += eval_data["change_magnitude"]
            evaluation_count += 1

            if eval_data["repair_accuracy"] == 1:
                perfect_repair_count += 1
                total_change_magnitude_perfect += eval_data["change_magnitude"]
            
            stats = eval_data["statistics"]
            total_model_svg_length += stats["model_svg_length"]
            total_standard_svg_length += stats["standard_svg_length"]
            total_bug_svg_length += stats["bug_svg_length"]
            total_character_differences += stats["character_differences"]
            
            if eval_data["repair_efficiency"]["processing_rate_chars_per_second"]:
                total_processing_rate += eval_data["repair_efficiency"]["processing_rate_chars_per_second"]
            
            acc = eval_data["repair_accuracy"]
            if acc >= 0.9:
                accuracy_ranges["excellent (>= 0.9)"] += 1
            elif acc >= 0.8:
                accuracy_ranges["good (0.8-0.9)"] += 1
            elif acc >= 0.6:
                accuracy_ranges["fair (0.6-0.8)"] += 1
            else:
                accuracy_ranges["poor (< 0.6)"] += 1

    summary = {
        "total_samples": len(samples),
        "successful_samples": sample_count,
        "detailed_results": detailed_results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if sample_count > 0:
        avg_execution_time = total_execution_time / sample_count
        summary["average_execution_time"] = avg_execution_time

    if evaluation_count > 0:
        avg_repair_accuracy = total_repair_accuracy / evaluation_count

        avg_change_magnitude_perfect = 0.0
        if perfect_repair_count > 0:
            avg_change_magnitude_perfect = total_change_magnitude_perfect / perfect_repair_count
        
        summary["evaluation_summary"] = {
            "samples_evaluated": evaluation_count,
            "average_repair_accuracy": avg_repair_accuracy,
            "average_change_magnitude_perfect": avg_change_magnitude_perfect,  
            "perfect_repair_count": perfect_repair_count,  
            "accuracy_distribution": accuracy_ranges
        }
    
    if evaluation_count > 0:
        final_avg_metrics = {
            "final_average_metrics": {
                "execution_time_seconds": total_execution_time / evaluation_count,
                "repair_accuracy": (total_repair_accuracy / evaluation_count),
                "change_magnitude": total_change_magnitude / evaluation_count,
                "change_magnitude_perfect": avg_change_magnitude_perfect,
                "model_svg_length": total_model_svg_length / evaluation_count,
                "standard_svg_length": total_standard_svg_length / evaluation_count,
                "bug_svg_length": total_bug_svg_length / evaluation_count,
                "character_differences": total_character_differences / evaluation_count,
                "processing_rate_chars_per_second": total_processing_rate / evaluation_count
            },
            "success_rate": sample_count / len(samples) if len(samples) > 0 else 0,
            "evaluation_rate": evaluation_count / len(samples) if len(samples) > 0 else 0,
            "perfect_repair_rate": perfect_repair_count / evaluation_count 
        }
        
        summary["final_evaluation_metrics"] = final_avg_metrics
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"results have been saved to: {output_file}")
    except Exception as e:
        logger.error(f"saving results to {output_file} failed: {e}")
    
    return summary

async def main_async():
    """
    Asynchronous main function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Repair SVG using API and save results')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file containing bug_svg and ground_truth fields')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file for results')
    parser.add_argument('--model', type=str, default="deepseekr1", help='Model name to use')
    parser.add_argument('--concurrent', type=int, default=5, help='Maximum concurrent requests')
    
    args = parser.parse_args()
    
    # Process samples
    summary = await process_batch_from_json(
        args.input,
        args.output,
        args.model,
        args.concurrent
    )
    
    # Output summary results
    if "error" in summary:
        print(f"Processing failed: {summary['error']}")
    else:
        print(f"Processing completed, total {summary['total_samples']} samples, {summary['successful_samples']} successful")
        
        # if "average_execution_time" in summary:
        #     print(f"Average execution time: {summary['average_execution_time']:.2f} seconds")
        
        # # Output evaluation metrics
        # if "evaluation_summary" in summary:
        #     eval_summary = summary["evaluation_summary"]
        #     print(f"Average repair accuracy: {eval_summary['average_repair_accuracy']:.2%}")
        #     print(f"Average change magnitude for perfect repairs: {eval_summary['average_change_magnitude_perfect']:.2%}")
            
        #     print("Accuracy distribution:")
        #     for range_name, count in eval_summary["accuracy_distribution"].items():
        #         percentage = count / eval_summary["samples_evaluated"] * 100 if eval_summary["samples_evaluated"] > 0 else 0
        #         print(f"  {range_name}: {count} samples ({percentage:.1f}%)")
        
        # # If there are final evaluation metrics, output them too
        # if "final_evaluation_metrics" in summary:
        #     final_metrics = summary["final_evaluation_metrics"]["final_average_metrics"]
        #     print("\nFinal average evaluation metrics:")
        #     print(f"  Execution time: {final_metrics['execution_time_seconds']:.2f} seconds")
        #     print(f"  Repair accuracy: {final_metrics['repair_accuracy']:.2%}")
        #     print(f"  Change magnitude: {final_metrics['change_magnitude']:.2%}")
        #     print(f"  Perfect repair change magnitude: {final_metrics['change_magnitude_perfect']:.2%}")
        #     print(f"  Processing rate: {final_metrics['processing_rate_chars_per_second']:.2f} chars/sec")
        
        print(f"Detailed results saved to: {args.output}")

def main():
    """
    Command line entry function
    """
    asyncio.run(main_async())

if __name__ == "__main__":
    main()