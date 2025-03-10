# KneeXNet 🦵🏻 <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/> [![arXiv](https://img.shields.io/badge/arXiv-2403.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2403.XXXXX) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## A State-of-the-Art Deep Learning Framework for Knee MRI Analysis

<div align="center">
<img src="https://img.shields.io/badge/Status-Research-blue"/> <img src="https://img.shields.io/badge/Version-1.0.0-green"/> <img src="https://img.shields.io/badge/Journal-Under_Review-orange"/>
</div>



## 📋 Table of Contents
- [Overview](#overview)
- [Performance](#performance)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 🔍 Overview

KneeXNet is an innovative deep learning framework specifically designed for the analysis of knee MRI scans. It combines graph convolutional networks, multi-scale feature fusion, and contrastive learning to effectively detect knee joint abnormalities, ACL tears, and meniscal tears with state-of-the-art accuracy.

**Key Features:**
- 📊 Superior performance compared to traditional ML and other deep learning methods
- 🧠 Novel architecture that captures complex patterns in knee MRI data
- 🔄 Multi-scale feature fusion for better representation learning
- 📈 Robust evaluation on multiple datasets
- 🛠️ Comprehensive implementation using PyTorch

## 🏆 Performance

<p align="center">
  <img src="kneeXNet_performance_comparison.png" alt="KneeXNet Performance" width="700px"/>
</p>

KneeXNet significantly outperforms existing methods:

| Method | Abnormality | ACL Tear | Meniscal Tear |
|--------|-------------|----------|---------------|
| SVM | 0.872 ± 0.013 | 0.841 ± 0.016 | 0.836 ± 0.015 |
| CNN | 0.923 ± 0.008 | 0.896 ± 0.010 | 0.889 ± 0.009 |
| SENet | 0.956 ± 0.005 | 0.934 ± 0.006 | 0.928 ± 0.006 |
| **KneeXNet** | **0.985 ± 0.003** | **0.972 ± 0.004** | **0.968 ± 0.004** |

## 📂 Repository Structure

```
KneeXNet/
├── datasets/            # Dataset loading and preprocessing
├── networks/            # Model architectures and components
├── out_dir/             # Output directory for results and checkpoints
├── scripts/             # Training, evaluation, and utility scripts
├── src/                 # Core source code
└── manage.py            # Main entry point for running experiments
```

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/username/KneeXNet.git
cd KneeXNet

# Create a new conda environment
conda create -n kneeXNet python=3.8
conda activate kneeXNet

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## 🚀 Usage

### Training

```bash
python manage.py train --config configs/kneeXNet_train.yaml
```

### Evaluation

```bash
python manage.py evaluate --model_path out_dir/checkpoints/best_model.pth --test_data path/to/test/data
```

### Inference

```bash
python manage.py predict --model_path out_dir/checkpoints/best_model.pth --input path/to/mri/scan --output path/to/results
```

## 📊 Dataset

The KneeXNet framework was trained and evaluated on a comprehensive knee MRI dataset:

| Set | Number of cases | Abnormalities | ACL tears | Meniscal tears |
|-----|-----------------|---------------|-----------|----------------|
| Training | 1,130 (82.5%) | 743 (65.8%) | 213 (18.8%) | 436 (38.6%) |
| Validation | 120 (8.8%) | 79 (65.8%) | 23 (19.2%) | 47 (39.2%) |
| Test | 120 (8.8%) | 78 (65.0%) | 22 (18.3%) | 45 (37.5%) |
| Total | 1,370 (100%) | 900 (65.7%) | 258 (18.8%) | 528 (38.5%) |

## 🧠 Model Architecture

<p align="center">
  <img src="kneeXNet_detailed_architecture.png" alt="KneeXNet Detailed Architecture" width="750px"/>
</p>

KneeXNet consists of several key components:

- **Feature Extraction**: Multi-scale convolutional layers
- **Spatial Reasoning**: Graph convolutional networks
- **Attention Mechanism**: Channel and spatial attention
- **Contrastive Learning**: For better feature representation

## 📈 Results

<div align="center">
<table>
  <tr>
    <td><img src="roc_curves.png" alt="ROC Curves" width="400px"/></td>
    <td><img src="confusion_matrix.png" alt="Confusion Matrix" width="400px"/></td>
  </tr>
  <tr>
    <td align="center">ROC Curves</td>
    <td align="center">Confusion Matrix</td>
  </tr>
</table>
</div>

## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{author2024kneeXNet,
  title={KneeXNet: A Novel Deep Learning Framework for Knee MRI Analysis},
  author={Author, A. and Researcher, B. and Scientist, C.},
  journal={arXiv preprint arXiv:2403.XXXXX},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

Thanks for Tongji University and University of Shanghai for Science and Technology.

---

<div align="center">
<p>
  <img src="https://img.shields.io/badge/Made_with-❤️-red"/> 
  <img src="https://img.shields.io/badge/Powered_by-Science-blue"/>
</p>

<p>
  <a href="mailto:contact@example.com"><img src="https://img.shields.io/badge/Email-contact%40example.com-blue?style=flat-square&logo=gmail"></a>
  <a href="https://twitter.com/username"><img src="https://img.shields.io/badge/Twitter-@username-blue?style=flat-square&logo=twitter"></a>
</p>
</div>
