#!/usr/bin/env python3
import os
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from openai import AsyncOpenAI
import aiofiles
from .set_metric import evaluate_svg_edit 
from ...eval_util import setup_logger,extract_svg_from_response

logger = setup_logger(name="style_editing",log_dir="../logs",log_filename="style_editing.log")

API_KEY = "your_api_key_here"  # Replace with your actual API key
BASE_URL = "your_base_url_here"  # Replace with your API base URL
AVAILABLE_MODELS = [ "Qwen2-72B-Instruct-AWQ", "gpt-4o",  "deepseekr1"]


async def edit_svg_from_api(original_svg: str, edit_command: str, model: str = "deepseekr1", semaphore: asyncio.Semaphore = None) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Edit SVG using API
    """
    start_time = time.time()
   
    system_prompt = (
        '''You are a professional SVG editing engineer with extensive experience in SVG editing. Your task is to receive SVG code and modification requests from users, and make precise modifications according to their instructions. Please only modify the parts specified by the user, keeping the rest of the code unchanged.After fixing, return 
         the complete corrected code in the following strict format: Answer:{SVG code}, providing
         the complete fixed code only in the {SVG code} position, without adding any explanations,
          comments, or other content.'''
    )
 
    user_prompt = f"""Here is the original SVG:{original_svg}.Edit command: {edit_command}"""
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
                            {"role": "user", "content": user_prompt}
                        ],
                        stream=True,
                    )
                    response = ""
                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            response += content  

                    execution_time = time.time() - start_time
                    
                    response_text = response
                    
                    svg_code = extract_svg_from_response(response_text)
                    
                    return svg_code, execution_time, response_text
                    
                except Exception as e:
                    retries += 1
                    logger.error(f"API call failed: {e}, retry {retries}")
                    if retries >= max_retries:
                        logger.error(f"Exceeded maximum retries, returning error")
                        return None, time.time() - start_time, "request error"
                    await asyncio.sleep(10) 

        except Exception as e:
            logger.error(f"Exception during API call: {e}")
            return None, time.time() - start_time, None
    
    # Execute API call with semaphore control if provided
    if semaphore:
        async with semaphore:
            return await execute_api_call()
    else:
        # Otherwise execute directly
        return await execute_api_call()

async def process_single_edit_sample(sample_id: str, sample_data: Dict[str, Any], model: str, output_file: str, semaphore: asyncio.Semaphore = None) -> Dict[str, Any]:
    """
    Process single SVG edit sample and save generated SVG, original SVG and target SVG to specified directory.
    """
    if not sample_data:
        return {
            "sample_id": sample_id,
            "error": "Sample data is empty"
        }

    svg_filename = list(sample_data.keys())[0]
    svg_content = sample_data[svg_filename]

    original_svg = svg_content.get("original", "")
    modified_svg = svg_content.get("modified", "")
    edit_command = svg_content.get("command", "")

    if not original_svg:
        return {
            "sample_id": sample_id,
            "error": "Missing original field"
        }

    logger.info(f"Processing sample {sample_id} ({svg_filename})...")

    generated_svg, execution_time, full_response = await edit_svg_from_api(original_svg, edit_command, model, semaphore)

    if not generated_svg:
        return {
            "sample_id": sample_id,
            "svg_filename": svg_filename,
            "error": "SVG edit failed",
            "execution_time": execution_time,
            "full_response": full_response,
            "original_svg": original_svg,
            "modified_svg": modified_svg,
            "edit_command": edit_command
        }
    
    save_dir = os.path.join(os.path.dirname(output_file), svg_filename)
    os.makedirs(save_dir, exist_ok=True)

    ori_path = os.path.join(save_dir, "ori.svg")
    async with aiofiles.open(ori_path, 'w', encoding='utf-8') as f:
        await f.write(original_svg)

    if modified_svg:
        gt_path = os.path.join(save_dir, "gt.svg")
        async with aiofiles.open(gt_path, 'w', encoding='utf-8') as f:
            await f.write(modified_svg)

    gen_path = os.path.join(save_dir, "gen.svg")
    async with aiofiles.open(gen_path, 'w', encoding='utf-8') as f:
        await f.write(generated_svg)

    evaluation_result = None
    if modified_svg and generated_svg:
        evaluation_result = evaluate_svg_edit(
            ori_svg=original_svg,
            gt_svg=modified_svg,
            gen_svg=generated_svg,
            execution_time=execution_time
        )

    result = {
        "sample_id": sample_id,
        "svg_filename": svg_filename,
        "original_svg": original_svg,
        "modified_svg": modified_svg,
        "generated_svg": generated_svg,
        "edit_command": edit_command,
        "execution_time": execution_time,
        "full_response": full_response
    }

    if evaluation_result:
        result["evaluation"] = evaluation_result

    async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
        import json
        await f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return result

async def process_edit_batch_from_json(json_file: str, output_file: str, model: str = "deepseekr1", max_concurrent: int = 5) -> Dict[str, Any]:
    """
    Batch process SVG edit samples from JSON file

    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            samples_data = json.load(f)

        if not isinstance(samples_data, list):
            logger.error(f"JSON data must be in list format")
            return {"error": "Invalid JSON data format, should be list"}

        samples = samples_data
    except Exception as e:
        logger.error(f"Failed to load JSON file {json_file}: {e}")
        return {"error": f"Failed to load JSON file: {e}"}
    
    if not samples:
        return {"error": "Sample data is empty"}

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
        
        logger.info(f"Processing batch {start_idx//batch_size + 1}, samples {start_idx+1}-{end_idx}...")

        batch_tasks = [
            process_single_edit_sample(sample_id, sample, model, output_file, semaphore)
            for sample_id, sample in zip(batch_ids, batch_samples)
        ]

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
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
        logger.info(f"Processed {processed}/{total_samples} samples, interim results saved to: {interim_output}")

    detailed_results = {}
    for result in results:
        if "sample_id" in result:
            detailed_results[result["sample_id"]] = result

    successful_samples = [r for r in results if "error" not in r]
    sample_count = len(successful_samples)

    total_execution_time = 0

    total_edit_distance_original_to_generated = 0
    total_edit_distance_original_to_modified = 0
    total_edit_distance_diff = 0

    mse_better_samples_count = 0
    mse_better_edit_distance_original_to_generated = 0
    mse_better_edit_distance_original_to_modified = 0
    mse_better_edit_distance_diff = 0

    total_mse_gen_gt = 0
    total_mse_ori_gt = 0
    mse_count = 0
    total_custom_metric = 0
    custom_metric_count = 0
    
    evaluation_count = 0
    
    for result in successful_samples:
        total_execution_time += result["execution_time"]
        
        if "evaluation" in result:
            eval_data = result["evaluation"]
            evaluation_count += 1

            ori_to_gen = eval_data["edit_distance"]["ori_to_gen"]
            ori_to_gt = eval_data["edit_distance"]["ori_to_gt"]
            edit_diff = eval_data["edit_distance"]["difference"]

            total_edit_distance_original_to_generated += ori_to_gen
            total_edit_distance_original_to_modified += ori_to_gt
            total_edit_distance_diff += edit_diff

            mse_condition_met = False
            if "mse" in eval_data and isinstance(eval_data["mse"], dict):
                if "gen_vs_gt" in eval_data["mse"] and "ori_vs_gt" in eval_data["mse"]:
                    gen_gt_mse = eval_data["mse"]["gen_vs_gt"]
                    ori_gt_mse = eval_data["mse"]["ori_vs_gt"]

                    total_mse_gen_gt += gen_gt_mse
                    total_mse_ori_gt += ori_gt_mse
                    mse_count += 1

                    if gen_gt_mse <= ori_gt_mse:
                        mse_condition_met = True

            if mse_condition_met:
                mse_better_samples_count += 1
                mse_better_edit_distance_original_to_generated += ori_to_gen
                mse_better_edit_distance_original_to_modified += ori_to_gt
                mse_better_edit_distance_diff += edit_diff

            if "custom_metric" in eval_data and isinstance(eval_data["custom_metric"], dict):
                if "value" in eval_data["custom_metric"] and eval_data["custom_metric"]["value"] is not None:
                    total_custom_metric += eval_data["custom_metric"]["value"]
                    custom_metric_count += 1

    summary = {
        "total_samples": len(samples),
        "successful_samples": sample_count,
        "samples_evaluated": evaluation_count,
        "detailed_results": detailed_results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if sample_count > 0:
        summary["average_execution_time"] = total_execution_time / sample_count

    if evaluation_count > 0:
        summary["evaluation_summary"] = {
            "average_edit_distance": {
                "original_to_generated": total_edit_distance_original_to_generated / evaluation_count,
                "original_to_modified": total_edit_distance_original_to_modified / evaluation_count,
                "difference": total_edit_distance_diff / evaluation_count
            }
        }

        if mse_better_samples_count > 0:
            summary["evaluation_summary"]["relate_edit_distance"] = {
                "count": mse_better_samples_count,
                "percentage": mse_better_samples_count / evaluation_count * 100,
                "average_edit_distance": {
                    "original_to_generated": mse_better_edit_distance_original_to_generated / mse_better_samples_count,
                    "original_to_modified": mse_better_edit_distance_original_to_modified / mse_better_samples_count,
                    "difference": mse_better_edit_distance_diff / mse_better_samples_count
                }
            }

        if mse_count > 0:
            summary["evaluation_summary"]["average_mse"] = {
                "gen_vs_gt": total_mse_gen_gt / mse_count,
                "ori_vs_gt": total_mse_ori_gt / mse_count,
                "ratio": total_mse_gen_gt / total_mse_ori_gt if total_mse_ori_gt > 0 else None
            }

        if custom_metric_count > 0:
            summary["evaluation_summary"]["average_custom_metric"] = total_custom_metric / custom_metric_count

    if evaluation_count > 0:
        final_avg_metrics = {
            "final_average_metrics": {
                "execution_time_seconds": total_execution_time / sample_count if sample_count > 0 else 0,
                "edit_distance_original_to_generated": total_edit_distance_original_to_generated / evaluation_count,
                "edit_distance_original_to_modified": total_edit_distance_original_to_modified / evaluation_count,
                "edit_distance_difference": total_edit_distance_diff / evaluation_count
            },
            "success_rate": sample_count / len(samples) if len(samples) > 0 else 0,
            "evaluation_rate": evaluation_count / len(samples) if len(samples) > 0 else 0
        }

        if mse_better_samples_count > 0:
            final_avg_metrics["final_average_metrics"]["mse_better_samples_percentage"] = mse_better_samples_count / evaluation_count * 100
            final_avg_metrics["final_average_metrics"]["mse_better_edit_distance_original_to_generated"] = mse_better_edit_distance_original_to_generated / mse_better_samples_count
            final_avg_metrics["final_average_metrics"]["mse_better_edit_distance_original_to_modified"] = mse_better_edit_distance_original_to_modified / mse_better_samples_count
            final_avg_metrics["final_average_metrics"]["mse_better_edit_distance_difference"] = mse_better_edit_distance_diff / mse_better_samples_count

        if mse_count > 0:
            final_avg_metrics["final_average_metrics"]["mse_gen_vs_gt"] = total_mse_gen_gt / mse_count
            final_avg_metrics["final_average_metrics"]["mse_ori_vs_gt"] = total_mse_ori_gt / mse_count
            final_avg_metrics["final_average_metrics"]["mse_ratio"] = total_mse_gen_gt / total_mse_ori_gt if total_mse_ori_gt > 0 else None

        if custom_metric_count > 0:
            final_avg_metrics["final_average_metrics"]["custom_metric"] = total_custom_metric / custom_metric_count

        summary["final_evaluation_metrics"] = final_avg_metrics
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Processing results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {e}")
    
    return summary

async def main_async():
    """
    Asynchronous main function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Edit SVG using API and save results')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path containing original, modified and command fields')
    parser.add_argument('--output', type=str, required=True, help='Output result JSON file path')
    parser.add_argument('--model', type=str, default="deepseekr1", help='Model name to use')
    parser.add_argument('--concurrent', type=int, default=5, help='Maximum concurrent requests')
    
    args = parser.parse_args()

    summary = await process_edit_batch_from_json(
        args.input,
        args.output,
        args.model,
        args.concurrent
    )

    if "error" in summary:
        print(f"Processing failed: {summary['error']}")
    else:
        print(f"Processing completed, total {summary['total_samples']} samples, {summary['successful_samples']} successful")
        
        # if "average_execution_time" in summary:
        #     print(f"Average execution time: {summary['average_execution_time']:.2f} seconds")

        # if "evaluation_summary" in summary:
        #     eval_summary = summary["evaluation_summary"]
        #     print(f"Average edit distance difference: {eval_summary['average_edit_distance']['difference']:.4f}")
            
        #     if "relate_edit_distance" in eval_summary:
        #         mbs = eval_summary["relate_edit_distance"]
        #         print(f"MSE improved samples: {mbs['count']} ({mbs['percentage']:.1f}%)")
        #         print(f"  MSE improved samples average edit distance difference: {mbs['average_edit_distance']['difference']:.4f}")
            
        #     if "average_mse" in eval_summary:
        #         mse = eval_summary["average_mse"]
        #         print(f"Average MSE(gen_vs_gt): {mse['gen_vs_gt']:.4f}")
        #         print(f"Average MSE(ori_vs_gt): {mse['ori_vs_gt']:.4f}")
        #         if mse['ratio'] is not None:
        #             print(f"MSE ratio: {mse['ratio']:.4f}")
            
        #     if "average_custom_metric" in eval_summary:
        #         print(f"Average custom metric: {eval_summary['average_custom_metric']:.4f}")

        # if "final_evaluation_metrics" in summary:
        #     final_metrics = summary["final_evaluation_metrics"]["final_average_metrics"]
        #     print("\nFinal average evaluation metrics:")
        #     print(f"  Execution time: {final_metrics['execution_time_seconds']:.2f} seconds")
        #     print(f"  Edit distance difference: {final_metrics['edit_distance_difference']:.4f}")
            
        #     if "mse_better_samples_percentage" in final_metrics:
        #         print(f"  MSE improved samples percentage: {final_metrics['mse_better_samples_percentage']:.1f}%")
        #         print(f"  MSE improved samples average difference: {final_metrics['mse_better_edit_distance_difference']:.4f}")
            
        #     if "mse_gen_vs_gt" in final_metrics:
        #         print(f"  MSE(gen_vs_gt): {final_metrics['mse_gen_vs_gt']:.4f}")
        #         print(f"  MSE(ori_vs_gt): {final_metrics['mse_ori_vs_gt']:.4f}")
        #         if final_metrics.get('mse_ratio') is not None:
        #             print(f"  MSE ratio: {final_metrics['mse_ratio']:.4f}")
            
        #     if "custom_metric" in final_metrics:
        #         print(f"  Custom metric: {final_metrics['custom_metric']:.4f}")
        
        print(f"Detailed results saved to: {args.output}")

def main():
    """
    Command line entry function
    """
    asyncio.run(main_async())

if __name__ == "__main__":
    main()