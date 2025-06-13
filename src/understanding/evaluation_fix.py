#!/usr/bin/env python3

import base64
import os
import json
import re
import random
import asyncio
import time
from openai import AsyncOpenAI
import logging
from datetime import datetime
from ..eval_util import setup_logger

logger = setup_logger(name="understanding", log_dir="./logs", log_filename="understanding.log")


API_KEY = "your_api_key_here"  # Replace with your actual API key
BASE_URL = "your_base_url_here"  # Replace with your API base URL
AVAILABLE_MODELS = [ "Qwen2-72B-Instruct-AWQ", "gpt-4o",  "deepseekr1"]

async def generate_response(query, request_id, semaphore, model):
    """
    Generate response from language model with error handling and retries
    
    """
    async with semaphore:
        retries = 0
        max_retries = 10
        
        while retries < max_retries:
            try:
                aclient = AsyncOpenAI(
                    base_url=BASE_URL,
                    api_key=API_KEY,
                )
                
                response = await aclient.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                )
                
                return request_id, response.choices[0].message.content
                
            except Exception as e:
                retries += 1
                logger.error(f"Request ID {request_id} error: {e}, retry attempt {retries}")
                if retries >= max_retries:
                    logger.error(f"Request ID {request_id} exceeded max retries, returning error")
                    return request_id, "request error"
                await asyncio.sleep(10) 

def extract_answer(response_text):
    """
    Extract answer choice (A, B, C, D) from model response using multiple strategies

    """
    if not response_text:
        return None

    # Strategy 1: Look for explicit "Answer: X" format
    pattern = r'Answer:\s*([A-D])'
    matches = re.findall(pattern, response_text)
    if matches:
        return matches[-1]  # Return the last match
    
    # Strategy 2: Parse line by line from bottom up
    lines = response_text.split('\n')
    for line in reversed(lines):  
        line = line.strip()
        
        # Single letter on its own line
        if line and re.match(r'^[A-D]$', line):
            return line

        # "The answer is X" patterns
        answer_match = re.search(r'(?:The\s+)?answer(?:\s+is)?(?:\s*[:=]?\s*)([A-D])', line, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1)

        # "I choose X" patterns
        choice_match = re.search(
            r'(?:I|my)?\s*(?:choose|select|pick|opt\s+for)(?:\s+option)?\s*(?:is|:)?\s*([A-D])', 
            line, re.IGNORECASE
        )
        if choice_match:
            return choice_match.group(1)

        # "Option X is correct" patterns
        option_match = re.search(
            r'option\s*([A-D])(?:\s+is\s+(?:correct|right|the\s+answer))', 
            line, re.IGNORECASE
        )
        if option_match:
            return option_match.group(1)

    logger.warning(f"Unable to extract answer from response:\n{response_text}")
    return None

def calculate_accuracy(results):
    """
    Calculate comprehensive accuracy statistics from evaluation results

    """
    if not results:
        return {
            "overall_accuracy": 0,
            "total": 0,
            "correct": 0,
            "answer_is_null": 0,
            "question_types": {}
        }

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    overall_accuracy = correct / total if total > 0 else 0
    answer_is_null = sum(1 for r in results if r["model_answer"] is None)

    # Calculate per-question-type accuracy
    question_types = {}
    for result in results:
        q_type = result.get("question_key", "unknown")
        if q_type not in question_types:
            question_types[q_type] = {"total": 0, "correct": 0}

        question_types[q_type]["total"] += 1
        if result["is_correct"]:
            question_types[q_type]["correct"] += 1

    # Calculate accuracy percentages for each question type
    for q_type in question_types:
        type_total = question_types[q_type]["total"]
        type_correct = question_types[q_type]["correct"]
        question_types[q_type]["accuracy"] = type_correct / type_total if type_total > 0 else 0

    return {
        "overall_accuracy": overall_accuracy,
        "total": total,
        "correct": correct,
        "answer_is_null": answer_is_null,
        "question_types": question_types
    }

def save_results(results, stats, output_file):
    """
    Save evaluation results and statistics to JSON file
  
    """
    output = {
        "overall_stats": stats,
        "detailed_results": results
    }

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation results saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False

async def evaluate_svg_questions(input_file, model_name, output_dir, concurrency_limit, 
                                num_samples=None, random_seed=None):
    """
    Main evaluation function for SVG question-answering assessment
    
    """
    if random_seed is not None:
        random.seed(random_seed)

    os.makedirs(output_dir, exist_ok=True)

    # Generate unique output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_trans_name = model_name.replace(':', '_').replace('/', '_')
    output_file = os.path.join(output_dir, f"{model_trans_name}_results_{timestamp}.json")

    # Load dataset
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Loaded dataset: {input_file}, containing {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

    # Parse dataset and create evaluation samples
    evaluation_samples = []

    if isinstance(dataset, list):
        # Handle list format
        for item in dataset:
            image_path = item.get("image_path", "")
            questions = item.get("questions", {})

            for question_key, question_data in questions.items():
                evaluation_samples.append({
                    "image_name": os.path.basename(image_path),
                    "image_path": image_path,
                    "question_key": question_key,
                    "question": question_data.get("question", ""),
                    "options": question_data.get("option", {}),
                    "correct_answer": question_data.get("answer", "")
                })
    else:
        # Handle dictionary format
        for image_name, image_data in dataset.items():
            image_path = image_data.get("image_path", "")
            questions = image_data.get("questions", {})

            for question_key, question_data in questions.items():
                evaluation_samples.append({
                    "image_name": image_name,
                    "image_path": image_path,
                    "question_key": question_key,
                    "question": question_data.get("question", ""),
                    "options": question_data.get("option", {}),
                    "correct_answer": question_data.get("answer", "")
                })

    # Sample subset if requested
    if num_samples and num_samples < len(evaluation_samples):
        evaluation_samples = random.sample(evaluation_samples, num_samples)

    logger.info(f"Evaluating {len(evaluation_samples)} question samples using model: {model_name}")

    semaphore = asyncio.Semaphore(concurrency_limit)
    start_time = time.time()
    
    # Create concurrent tasks
    tasks = []
    for idx, sample in enumerate(evaluation_samples):
        try:
            # Load SVG file (convert PNG path to SVG path)
            svg_path = sample["image_path"].replace('.png', '.svg')
            
            with open(svg_path, "r", encoding='utf-8') as f:
                svg_content = f.read()
        except Exception as e:
            logger.error(f"Error reading SVG file: {svg_path}, {e}")
            continue

        # Format options string
        options_str = "; ".join([f"{key}) {value}" for key, value in sample["options"].items()])

        # Create comprehensive prompt
        prompt = f"""You are an SVG analysis expert. Follow these steps carefully to answer the given multiple-choice question.

        Task Instruction
        1. Analyze the given SVG code to understand the visual content and structure.
        2. Answer the multiple-choice question based on your analysis of the SVG.
        3. Output your answer in the exact format 'Answer: X' in the last line, where X is one of A, B, C, or D.

        SVG Code
        {svg_content}

        Question
        {sample["question"]}

        Options
        {options_str}

        Important Notes
        - Carefully analyze the SVG code to understand what it represents visually.
        - Answer exactly the given multiple-choice question - do not create new questions.
        - Your final answer must be in the exact format 'Answer: X' where X is A, B, C, or D.
        - Base your answer solely on the SVG content provided.

        Now, analyze the SVG and answer the question:
        """

        task = asyncio.create_task(
            generate_response(prompt, idx, semaphore, model_name)
        )
        tasks.append((idx, sample, task))

    # Process results as they complete
    results = []
    total = len(tasks)
    completed = 0

    for future in asyncio.as_completed([task for _, _, task in tasks]):
        request_id, response = await future
       
        # Find corresponding sample
        sample = None
        for idx, s, _ in tasks:
            if idx == request_id:
                sample = s
                break
        
        if not sample:
            logger.error(f"Unable to find sample for request ID {request_id}")
            continue
        
        completed += 1
        if completed % 10 == 0 or completed == total:
            logger.info(f"Progress: {completed}/{total} queries processed ({completed/total:.1%})")
    
        # Extract and validate answer
        model_answer = extract_answer(response)
        correct_answer = sample["correct_answer"]
        is_correct = (model_answer == correct_answer) if model_answer else False
        
        # Create detailed result record
        result = {
            "image_name": sample["image_name"],
            "image_path": sample["image_path"],
            "question_key": sample["question_key"],
            "question": sample["question"],
            "options": sample["options"],
            "correct_answer": correct_answer,
            "model_answer": model_answer,
            "is_correct": is_correct,
            "model_response": response
        }
        
        results.append(result)

        # Log individual result
        log_msg = f"Image: {sample['image_name']}, Question: {sample['question_key']}, "
        log_msg += f"Correct: {correct_answer}, Model: {model_answer}, Correct: {is_correct}"
        logger.info(log_msg)
    
    end_time = time.time()
    logger.info(f"Total evaluation time: {end_time - start_time:.2f} seconds")

    # Calculate and log statistics
    stats = calculate_accuracy(results)
    logger.info(f"Overall accuracy: {stats['overall_accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    for q_type, type_stats in stats["question_types"].items():
        logger.info(f"Question type '{q_type}' accuracy: {type_stats['accuracy']:.2%} "
                     f"({type_stats['correct']}/{type_stats['total']})")

    # Save results
    save_results(results, stats, output_file)
    return stats


async def main():
    """
    Main entry point for the SVG question-answering evaluation system
   
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Understanding qa using API and save results')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input JSON dataset file')
    parser.add_argument('--model', type=str, default='gpt-4o', 
                       help='Model name to evaluate (default: gpt-4o)')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results', 
                       help='Output directory for evaluation results (default: ./evaluation_results)')
    parser.add_argument('--concurrency', type=int, default=5, 
                       help='Maximum concurrent API requests (default: 5)')
    parser.add_argument('--samples', type=int, default=None, 
                       help='Number of samples to evaluate (default: all samples)')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for reproducible sampling (default: system time)')

    args = parser.parse_args()


    try:
        # Run evaluation
        stats = await evaluate_svg_questions(
            args.input,
            args.model,
            args.output_dir,
            args.concurrency,
            args.samples,
            args.seed
        )
        
        if stats:
            logger.info("=== Evaluation Complete ===")
            logger.info(f"Final accuracy: {stats['overall_accuracy']:.2%}")
            logger.info(f"Questions answered: {stats['total']}")
            logger.info(f"Extraction failures: {stats['answer_is_null']}")
        else:
            logger.error("Evaluation failed")
            
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())