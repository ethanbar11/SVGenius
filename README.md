# 🎨 SVGenius:  Benchmarking LLMs in SVG Understanding, Editing and Generation

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange?style=for-the-badge)](https://arxiv.org/abs/2506.03139)
[![Dataset](https://img.shields.io/badge/Dataset-Available-green?style=for-the-badge)](#data)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](#license)

## 🌟 Overview

SVGenius is a pioneering benchmark designed to systematically evaluate the capabilities of Large Language Models in SVG processing. Our benchmark encompasses **24 diverse application domains** and provides comprehensive evaluation across three core dimensions: **Understanding**, **Editing** and **Generation**.

## ✨ Key Contributions
Our contributions can be summarized as:

🔍 Problem Identification: We identify key limitations in existing SVG evaluation approaches and propose a comprehensive solution
🎯 Benchmark Innovation: We introduce SVGenius, the first large-scale, complexity-stratified benchmark for SVG processing with real-world data
📊 Extensive Evaluation: We provide comprehensive evaluation of 24 models, establishing performance baselines and identifying key factors influencing SVG processing capabilities

## 📁 Repository Structure

```
SVGenius/
├── 📂 data/                    # Hierarchical Dataset
│   ├── easy/                   # Easy level data
│   ├── moderate/               # Moderate level data
│   └── complex/                # Complex level data
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
- **Multimodal-to-SVG Generation**: SVG creation from multiple input modalities
- **Style Transfer**: Applying artistic styles to existing SVGs

## 📊 Benchmark Statistics

| Metric | Count |
|--------|-------|
| Total Samples | 2377 |
| Application Domains | 24 |
| Task Categories | 8 |
| Difficulty Levels | 3 (Easy/Moderate/Complex) |
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
  ·Complexity Levels: Three difficulty tiers (Easy, Moderate, Complex)
  ·Statistical Robustness: Three independent runs per setting


*Detailed results available in the [supplementary materials](./supplementary/).*


## 📄 License

This project is made available for research purposes. Please refer to the license terms for usage guidelines.

## 🔗 Anonymous Submission

This repository contains supplementary materials for an anonymous conference submission. All identifying information has been removed to maintain anonymity during the review process.

## 🙏 Acknowledgments

We thank the anonymous reviewers and the research community for their valuable feedback.

