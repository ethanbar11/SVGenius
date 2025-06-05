# 🎨 SVGenius:  Benchmarking LLMs in SVG Understanding, Editing and Generation


<div align="center">
    <a href="https:https://arxiv.org/abs/2506.03139" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-SVGenius-red?logo=arxiv" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/xiaoooobai/SVGenius" target="_blank">
        <img alt="SVGenius" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-SVGenius-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://zju-real.github.io/SVGenius/" target="_blank">
        <img alt="Webpage" src="https://img.shields.io/badge/%F0%9F%8C%8E_Website-SVGenius-green.svg" height="20" />
    </a>
</div>


## 🌟 Overview

SVGenius evaluates (M)LLMs capabilities across three progressive dimensions: **Understanding** (perceptua and semantic QA), **Editing** (bug fixing, code optimization, style editing), and **Generation** (text-to-SVG, image-to-SVG, style transfer).  Built on real-world data from **24** application domains with systematic complexity stratification, SVGenius evaluates models through **8** task categories and **18** metrics. We assess **22** mainstream models spanning different scales, architectures, training paradigms, and accessibility levels
<img src="docs/static/images/overview.png" width="100%"/>

## ✨ Key Contributions
Our contributions can be summarized as:

🔍 Problem Identification: We identify key limitations in existing SVG evaluation approaches and propose a comprehensive solution
🎯 Benchmark Innovation: We introduce SVGenius, the first large-scale, complexity-stratified benchmark for SVG processing with real-world data
📊 Extensive Evaluation: We provide comprehensive evaluation of 22 models, establishing performance baselines and identifying key factors influencing SVG processing capabilities

## 📁 Repository Structure

```
SVGenius/
├── 📂 data/                    # Hierarchical Dataset
│   ├── easy/                   # Easy level data
│   ├── medium/               # Moderate level data
│   └── hard/                # Complex level data
├── 📂 tasks/                  # Eight task subcategories
│   ├── understanding/          # Understanding dimension includes semantic QA and perception QA
│   ├── editing/                # Editing dimension includes code optimization, style editing and bug fixing
│   └── generation/             # Generation dimension includes text-to-svg, multimodel-to-svg and style transferr
├── 📂 supplementary/          # Additional materials
│   └── appendix.pdf           # Appendix includes data construction, tasks, metrics and more details
└── 📄 README.md               # This file
```

## 🎯 Task Categories

### 🔍 Understanding Dimension
- **Perceptual QA**: Visual understanding of SVG elements and layouts
- **Semantic QA**: Comprehension of symbolic meanings and relationships

### ✏️ Editing Dimension  
- **Bug Fixing**: Identification and correction of SVG code errors
- **Code Optimization**: Performance and efficiency improvements
- **Style Editing**: Visual appearance modifications

### 🎨 Generation Dimension
- **Text-to-SVG Generation**: Creating SVG from textual descriptions
- **Image-to-SVG Generation**: SVG creation from multiple input modalities
- **Style Transfer**: Applying artistic styles to existing SVGs

## 📊 Benchmark Statistics

| Metric | Count |
|--------|-------|
| Total Samples | 2377 |
| Application Domains | 24 |
| Task Categories | 8 |
| Difficulty Levels | 3 (Easy/Medium/Hard) |
| Evaluated Models | 22 |

🧪 Model Evaluation
We evaluate a diverse set of models on SVGenius to assess SVG processing capabilities across different architectures, scales, and training paradigms.
Evaluated Models
Our evaluation encompasses:

🔒 Proprietary Models: GPT-4o, Gemini-2.0-Flash, Claude 3.7-Sonnet
🌐 Open-Source Models: Representative models spanning 1.5B to 72B parameters
  ·DeepSeek-R1, Qwen2.5/3, Llama-3.2, Mistral-Small
🎨 SVG-Specialized Systems: Iconshop, StarVector, LLM4SVG

Evaluation Protocol
  ·Zero-shot Settings: All models evaluated using default configurations
  ·Complexity Levels: Three difficulty tiers (Easy, Medium, Hard)
  ·Statistical Robustness: Three independent runs per setting

*Detailed results available in the [supplementary materials](./supplementary/supplementary.pdf).*
