# -*- coding: utf-8 -*-
from collections import defaultdict
from datetime import datetime
import base64
import json
import os
from openai import AsyncOpenAI
from tqdm import tqdm
from typing import Optional, Tuple
import time
import sys
import traceback
import logging
import asyncio
from ...eval_util import setup_logger, extract_svg_from_response

logger = setup_logger(name="style_trans_eval", log_dir="../logs", log_filename="style_trans_eval.log")

API_KEY = "your_api_key_here"
BASE_URL = "your_base_url_here"
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

async def generate_score(style: str, original_image_path: str, transferred_image_path: str, 
                        score_rubric: dict, model: str = "gpt-4o") -> Tuple[str, str, float]:
    """
    Generate evaluation score for style transfer result using LLM as judge
    """
    # Handle case where style transfer failed
    if transferred_image_path == "":
        return "SVG style transfer failed - no output image provided.", "0", 0.0

    start_time = time.time()
    
    try:
        with open(original_image_path, "rb") as f:
            base64_image1 = base64.b64encode(f.read()).decode("utf-8")
        with open(transferred_image_path, "rb") as f:
            base64_image2 = base64.b64encode(f.read()).decode("utf-8")
        
        logger.info(f"Evaluating style transfer to: {style}")

        instruction = f"Transfer the provided Image 1 to {style} style"
    
        ABS_SYSTEM_PROMPT = """You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."""
        ABSOLUTE_PROMPT = """Task Description:
            An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
            1. Make a brief description of the style transfer that how it modifies the image 1 to image 2.
            2. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
            3. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
            4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
            5. Please do not generate any other opening, closing, and explanations.


            The instruction to evaluate:
            {instruction}

            Response to evaluate:
            the given Image 2
            Score Rubrics:
            {rubric}

            Feedback: """

        prompt = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(
            instruction=instruction,
            rubric=score_rubric
        )

        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image1}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image2}"
                            },
                        },
                    ]
                }
            ],
            temperature=0.05,
            top_p=0.95,
            stream=True
        )
        response_text = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_text += content

        execution_time = time.time() - start_time

        feedback = response_text[:response_text.find("[RESULT]")]
        score = response_text[response_text.find("[RESULT]") + len("[RESULT]"):].replace("</s>", "").strip()

        return feedback, score, execution_time

    except Exception as e:
        logger.error(f"API call failed: {e}")
        return f"Evaluation failed due to API error: {str(e)}", "0", time.time() - start_time

async def evaluate_svg_style_transfer(input_file: str, output_dir: str, thread_num: int, 
                                    model: str = "gpt-4o"):
    """
    Main evaluation function for SVG style transfer assessment
    """
    try:
        with open(input_file, "r", encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Loaded dataset with {len(dataset)} images from {input_file}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    rubric_path = './style_trans/score_rubrics.json'
    try:
        with open(rubric_path, "r", encoding='utf-8') as f:
            score_rubrics = json.load(f)
        logger.info(f"Loaded {len(score_rubrics)} evaluation criteria from rubrics")
    except Exception as e:
        logger.error(f"Failed to load scoring rubrics: {e}")
        return

    style_list = ["Cartoon Style", "Pixel-art", "Line-art", "3D-style"]
    
    # Prepare evaluation samples
    evaluation_samples = []
    for image_name, image_data in dataset.items():
        if 'style' not in image_data:
            logger.warning(f"Skipping {image_name}: missing style information")
            continue
            
        evaluation_samples.append({
            "image_name": image_name,
            "style": image_data.get("style", ""),
            "original_image_path": image_data.get("original_image_path", ""),
            "transferred_image_path": image_data.get("transferred_image_path", ""),
        })

    logger.info(f"Prepared {len(evaluation_samples)} samples for evaluation")

    overall_results = {}
    detailed_results = {}
 
    for criterion_idx, rubric in enumerate(score_rubrics):
        criterion_name = rubric.get("criteria_description", f"Criterion_{criterion_idx}")
        logger.info(f"Evaluating criterion: {criterion_name}")
        
        results = []
        sum_score = defaultdict(int)
        count = defaultdict(int)

        for batch_start in tqdm(range(0, len(evaluation_samples), thread_num), 
                              desc=f"Evaluating {model} - {criterion_name}"):
            batch = evaluation_samples[batch_start:batch_start + thread_num]
            
            # Create concurrent tasks for batch
            tasks = [
                generate_score(
                    sample['style'],
                    sample["original_image_path"],
                    sample["transferred_image_path"],
                    rubric,
                    model
                )
                for sample in batch
            ]

            batch_responses = await asyncio.gather(*tasks)
            for sample_idx, response in enumerate(batch_responses):
                sample = batch[sample_idx]
                feedback, score, execution_time = response

                if feedback is None or score is None:
                    logger.warning(f"Evaluation failed for image {sample['image_name']}")
                    continue

                result = {
                    "image_name": sample["image_name"],
                    "style": sample['style'],
                    "original_image_path": sample["original_image_path"],
                    "transferred_image_path": sample["transferred_image_path"],
                    "feedback": feedback,
                    "score": score,
                    "execution_time": execution_time
                }

                results.append(result)
               
                try:
                    numeric_score = int(score)
                    sum_score[sample['style']] += numeric_score
                    count[sample['style']] += 1
                except ValueError:
                    logger.error(f"Invalid score format - Feedback: {feedback}, Score: {score}")

                log_msg = f"Image: {sample['image_name']}, Style: {sample['style']}, "
                log_msg += f"Score: {score}, Time: {execution_time:.2f}s"
                logger.info(log_msg)

        criterion_averages = {}
        for style in style_list:
            if style in count and count[style] > 0:
                criterion_averages[style] = round(sum_score[style] / count[style], 2)
                logger.info(f"{criterion_name} - {style}: {criterion_averages[style]} "
                           f"({count[style]} samples)")

        overall_results[criterion_name] = criterion_averages
        detailed_results[criterion_name] = results

    total_results = {
        "overall_result": overall_results,
        "detailed_result": detailed_results
    }

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(input_file)
    file_name = os.path.splitext(base_name)[0]
    result_file = os.path.join(output_dir, f"{file_name}_evaluation_{timestamp}.json")
    
    try:
        with open(result_file, "w", encoding='utf-8') as f:
            json.dump(total_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation results saved to: {result_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def main():
    """
    Main entry point for SVG style transfer evaluation system
    """
    import argparse
    parser = argparse.ArgumentParser(description='Transform SVG style using API and save results')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input JSON dataset file path')
    parser.add_argument('--output-dir', type=str, required=True, 
                       help='Output directory for evaluation results')
    parser.add_argument('--model', type=str, default="gpt-4o", 
                       help='Model name to use for evaluation (default: gpt-4o)')
    parser.add_argument('--thread_num', type=int, default=1, 
                       help='Maximum concurrent evaluation threads (default: 1)')

    args = parser.parse_args()

    try:
        # Run evaluation
        asyncio.run(evaluate_svg_style_transfer(
            args.input,
            args.output_dir,
            args.thread_num,
            args.model,
        ))
        logger.info("Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()