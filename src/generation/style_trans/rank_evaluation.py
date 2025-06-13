# -*- coding: utf-8 -*-
import argparse
import asyncio
import base64
import glob
import json
import logging
import os
import random
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from typing import Tuple
from PIL import Image
from openai import AsyncOpenAI

# Configuration
base_model = "deepseekr1"
API_KEY = "your_api_key_here"  
BASE_URL = "your_base_url_here"

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

async def generate_score(style: str, original_image_path: str, transferred_image_path1: str,
                         transferred_image_path2: str, model: str = "deepseekr1") -> Tuple[str, str]:
    """
    Generate relative score between two style transfer results

    """
    # Load evaluation criteria
    rubric = [
        "Style Consistency: Compared to the reference style, how well does the conversion match the target style?",
        "Content Preservation: To what extent does the SVG conversion preserve the basic content and main elements of the original image?",
        "Visual Quality: How is the visual effect of the conversion result?"
    ]

    # Encode images as base64
    with open(original_image_path, "rb") as f:
        base64_image1 = base64.b64encode(f.read()).decode("utf-8")
    with open(transferred_image_path1, "rb") as f:
        base64_image2 = base64.b64encode(f.read()).decode("utf-8")
    with open(transferred_image_path2, "rb") as f:
        base64_image3 = base64.b64encode(f.read()).decode("utf-8")

    print(original_image_path, style)
    instruction = f"Transform Image 1 into {style}"

    REL_SYSTEM_PROMPT = "As an impartial judge assistant, your role is to provide detailed comparative analysis between two style transfer outputs, focusing on their adherence to the specified criteria."

    RELATIVE_PROMPT = """Evaluation Task:
    You will compare two style-transformed images (Response A/image 2 and Response B/image 3) against the original Image 1. Follow these steps:
        
    1. CRITERIA-BASED ANALYSIS:
       - Evaluate both transformed images strictly using the provided scoring rubric
       - Focus on how each image interprets the style transfer differently

    2. COMPARATIVE FEEDBACK:
       - Directly compare Response A and Response B toward the style transform of Image 1
       - Highlight key similarities and differences in their style execution
       - Note which aspects better capture the target style characteristics

    3. DECISION:
       - Conclude with a clear preference: "[RESULT] A" or "[RESULT] B"
       - Base your decision solely on the rubric criteria
    
    4. Please do not generate any other opening, closing, and explanations.
    
    The instruction to evaluate:
    {instruction}

    Response to evaluate:
    the given image 2 and image 3

    Score Rubrics:
    {rubric}

    Output format (strictly follow):
    Feedback: [Your detailed comparison analysis focusing on rubric criteria] 
    [RESULT] [A/B]"""

    prompt = REL_SYSTEM_PROMPT + "\n\n" + RELATIVE_PROMPT.format(
        instruction=instruction,
        rubric=rubric)

    # Retry mechanism for API calls
    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
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
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image3}"
                                },
                            },
                        ]
                    }
                ],
                temperature=0.05,
                stream=True
            )
            
            response_text = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response_text += content  

            # Parse feedback and result
            feedback = response_text[:response_text.find("[RESULT]")]
            score = response_text[response_text.find("[RESULT]") + len("[RESULT]"):].replace("</s>", "").strip()
            print(f"Feedback:{feedback}")
            print(f"Winner:{score}")

            if score == "A":
                result = transferred_image_path1
            else:
                result = transferred_image_path2
            return feedback, result

        except Exception as e:
            retries += 1
            logging.error(f"Request error: {e}, retry attempt {retries}")
            if retries >= max_retries:
                logging.error(f"Request exceeded max retries, returning error")
                return None, None
            await asyncio.sleep(10)

def extract_model_from_file(image_name, file_name):
    """Extract model name from file name"""
    pattern = re.escape(image_name) if not re.search(r'[.^$*+?{}[\]\\|()]', image_name) else image_name
    match = re.search(r'.*' + pattern, file_name)
    if match is None:
        return None
    end_pos = match.end()
    model_name = file_name[end_pos + 1:].removesuffix('.png') if end_pos + 1 < len(file_name) else None
    return model_name

async def get_rank(image_name, style, original_path, image_paths, output_dir, model, max_concurrency=5):
    """Get ranking for images through pairwise comparison"""
    base_path = f"{original_path.removesuffix('.png')}_{base_model}.png"
    output_path = os.path.join(output_dir, 'results.json')

    if not os.path.exists(base_path):
        return
    
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_image(image_path):
        async with semaphore: 
            if image_path == base_path:
                return None
            # Randomize comparison order
            if random.random() < 0.5:
                process_result = await generate_score(style, original_path, base_path, image_path, model=model)
            else:
                process_result = await generate_score(style, original_path, image_path, base_path, model=model)
            return process_result

    tasks = [process_image(image_path) for image_path in image_paths]
    results = await asyncio.gather(*tasks)

    # Collect results
    logs = []
    for image_path, result in zip(image_paths, results):
        if result is None:
            continue
        feedback, winner = result
        if winner is not None:
            logs.append({
                "Base model": base_model,
                "Compare model": extract_model_from_file(image_name, image_path),
                "feedback": feedback,
                "winner": extract_model_from_file(image_name, winner)
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def process_files(svg_dir, model):
    """Process all files and generate rankings"""
    model_list = [
        "Qwen3-32B-AWQ",
        "qwen_qwq-32b_free",
        "deepseekr1",
        "openai_gpt-4o",
        "google_gemini-2.0-flash-001",
        "anthropic_claude-3.7-sonnet"
    ]

    # Load all SVG captions
    svg_captions = []
    
    caption_files = [
        './style_trans/easy_icons/svg_captions.json',
        './style_trans/complex_icons/svg_captions.json',
        './style_trans/illustrations/svg_captions.json'
    ]
    
    for svg_path in caption_files:
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_captions.extend(json.load(f))

    evaluation_samples = {}

    # Prepare evaluation samples
    for data in svg_captions:
        image_path = data['image_path']
        style = data['question']
        basename = os.path.basename(image_path)
        name = os.path.splitext(basename)[0]

        svg_path = os.path.join(svg_dir, name)
        png_path = os.path.join(svg_path, basename)

        if os.path.exists(image_path):
            shutil.copy2(image_path, png_path)
        evaluation_samples[name] = {
            "style": style,
            "original_image_path": png_path,
        }

    subdirs = glob.glob(f"{svg_dir}/*/")
    count = defaultdict(int)
    score = defaultdict(int)

    # Process each subdirectory
    for subdir in subdirs:
        files = glob.glob(f"{subdir}/*")
        image_paths = []

        # Collect relevant PNG files
        for file in files:
            if any(re.search(model, file) for model in model_list):
                if file.endswith(".png"):
                    image_paths.append(file)

        image_name = os.path.basename(subdir.removesuffix('\\'))
        image_name = os.path.splitext(image_name)[0]
        result_dir = os.path.join(svg_dir, image_name)

        if image_name not in evaluation_samples:
            continue

        sample = evaluation_samples[image_name]
        
        # Run ranking evaluation
        asyncio.run(
            get_rank(image_name, sample['style'], sample['original_image_path'], image_paths, result_dir, model))

        # Process results
        with open(os.path.join(result_dir, 'results.json'), 'r', encoding='utf-8') as f:
            result_log = json.load(f)

        for result in result_log:
            count[result['Base model']] += 1
            count[result['Compare model']] += 1
            score[result['winner']] += 1

    # Calculate win rates
    win_rate = {}
    for model_name, model_score in score.items():
        win_rate[model_name] = model_score / count[model_name]

    # Save overall results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{svg_dir}/overall_results_{timestamp}.json"
    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(win_rate, f, ensure_ascii=False, indent=2)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SVG Style Transfer Relative Evaluation')
    parser.add_argument('--svg-dir', type=str,
                        default="./style_trans/human_eval/easy/process",
                        help='SVG directory path')
    parser.add_argument('--model', type=str,
                        default="deepseekr1",
                        help='Evaluation model name')
    
    args = parser.parse_args()

    process_files(args.svg_dir, args.model)

if __name__ == "__main__":
    main()