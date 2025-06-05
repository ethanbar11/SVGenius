# 🎨 SVGenius:  Benchmarking LLMs in SVG Understanding, Editing and Generation


<div align="center">
    <a href="https://arxiv.org/abs/2506.03139" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-SVGenius-red?logo=arxiv" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/xiaoooobai/SVGenius" target="_blank">
        <img alt="SVGenius" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-SVGenius-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://zju-real.github.io/SVGenius/" target="_blank">
        <img alt="Webpage" src="https://img.shields.io/badge/%F0%9F%8C%8E_Website-SVGenius-green.svg" height="20" />
    </a>
</div>

## 🔥🔥🔥 News !!

- [2025/06/04] 👋 Release Datasets. 🤗[Dataset](https://huggingface.co/datasets/xiaoooobai/SVGenius).
- [2025/06/04] 👋 Unpload paper. [Arxiv](https://arxiv.org/abs/2506.03139).

## 🌟 Overview

SVGenius evaluates (M)LLMs capabilities across three progressive dimensions: **Understanding** (perceptua and semantic QA), **Editing** (bug fixing, code optimization, style editing), and **Generation** (text-to-SVG, image-to-SVG, style transfer).  Built on real-world data from **24** application domains with systematic complexity stratification, SVGenius evaluates models through **8** task categories and **18** metrics. We assess **22** mainstream models spanning different scales, architectures, training paradigms, and accessibility levels.

<img src="docs/static/images/overview.jpg" width="100%"/>


Comparison of SVGenius with existing SVG processing benchmarks. 

<img src="docs/static/images/compare.png" width="100%"/>

## ✨ Data Construct

Current SVG benchmarks face critical limitations in data quality and diversity.To address these limitations, we construct a comprehensive dataset with principled complexity stratification as show in the following figure: Left: systematic pipeline from data collection, processing, human filtering to complexity stratification. Center: 24-domain coverage across diverse applications. Right: validation of complexity stratification showing clear hierarchical separation across Easy, Medium, and Hard levels through feature distributions and complexity scores.

<img src="docs/static/images/data_construct.jpg" width="100%"/>

## 📊 Benchmark Statistics

🧪 Model Evaluation
We evaluate a diverse set of models on SVGenius to assess SVG processing capabilities across different architectures, scales, and training paradigms:

🔒 Proprietary Models: GPT-4o, Gemini-2.0-Flash, Claude 3.7-Sonnet

🌐 Open-Source Models: Representative models spanning 1.5B to 72B parameters
  ·DeepSeek-R1, Qwen2.5/3, Llama-3.2, Mistral-Small (16 models)
  
🎨 SVG-Specialized Systems: Iconshop, StarVector, LLM4SVG

*Detailed results available in the [supplementary materials](./supplementary/supplementary.pdf).*

## 📁 QucikStart

```
SVGenius/
├── 📂 docs/                    # project page code
├── 📂 src/                    # data construct and evaluate code (coming soon)
├── 📂 supplementary/          # Additional materials
│   └── appendix.pdf           # Appendix includes data construction, tasks, metrics and more details
└── 📄 README.md               # This file
```

## Citation

```bibtex
@misc{chen2025svgeniusbenchmarkingllmssvg,
      title={SVGenius: Benchmarking LLMs in SVG Understanding, Editing and Generation}, 
      author={Siqi Chen and Xinyu Dong and Haolei Xu and Xingyu Wu and Fei Tang and Hang Zhang and Yuchen Yan and Linjuan Wu and Wenqi Zhang and Guiyang Hou and Yongliang Shen and          Weiming Lu and Yueting Zhuang},
      year={2025},
      eprint={2506.03139},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.03139}, 
}
```
