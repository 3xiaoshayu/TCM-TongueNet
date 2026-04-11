<div align="center">

# TCM-TongueNet




<p align="center">
  <img src="tongue.png" alt="tongue" width="150" />
</p>

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/ResNet-101-red.svg)](https://example.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-4285F4.svg?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1xU24ulsjo5fBjQbofMpXyseOV7X5jnhn?usp=sharing)


An Interpretable Deep Learning Framework for Tongue Diagnosis: From Spatial Localization to Feature–Organ Association


</div>

---

## Overview


A PyTorch-based open-source toolkit for AI-assisted TCM tongue diagnosis, supporting dual-task decoupled modeling, Grad-CAM visualization, dual-matrix consistency validation, and TCM prior-guided attention learning.

---

## Project Structure

```
├── scripts/                    
│
├── split_dataset.py             # Dataset deduplication & re-split
├── train_5class_zangfu.py       # 5-organ classification training
├── inference.py                 # Batch inference & evaluation plots
├── adaptive_segmentation.py     # Heatmap-based tongue region segmentation
├── gradcam_visualization.py     # Grad-CAM heatmap visualization
├── region_quanification.py      # Region attention & TCM consistency
├── train_13class_zangfu_tongue.py # 13-class (tongue + organs) training
├── generate_all_maps.py         # Save feature & CAM maps to .npz
├── attention_matrix_from_image.py   # Attention matrix from images (MLP)
├── attention_matrix_from_path.py    # Attention matrix from CSV data
├── consistency_analysis_importance_matching.py  # Importance matching score
└── consistency_analysis_weighted_fusion.py     # Weighted fusion consistency
```

---

## Key Highlights

- **Dual-Task Decoupled Modeling**：Independent networks for five-viscera classification and multi-label feature learning effectively reduce feature interference and performance degradation.
- **Dual-Dimensional Interpretability**：Spatial localization via Grad-CAM and feature attribution via attention matrices reveal the underlying connections between tongue features and visceral abnormalities.
- **Data-Driven Adaptive Region Partitioning**：Tongue regions are statistically determined from large-sample attention distributions, reducing errors caused by fixed-ratio partitioning.
- **Dual-Matrix Rank Consistency Verification**：Cross-validation of two attention matrices via rank correlation significantly improves result reliability and reproducibility.
- **Quantitative TCM Prior Validation**：TCM priors serve as attention guidance without circular reasoning; core patterns are learned from data, ensuring scientific rigor.

---

## Installation

### Prerequisites
- Python 3.13.11
- CUDA 12.9 compatible GPU (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/3xiaoshayu/TCM-TongueNet.git
cd TCM-TongueNet

# Install dependencies
pip install torch torchvision opencv-python pandas numpy scipy matplotlib seaborn pillow scikit-learn pytorch-grad-cam
```

**Key Dependencies:**
```txt
torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=9.0.0
scikit-learn>=1.0.0
pytorch-grad-cam>=1.4.0
```

---

## Usage Rules

This project is open-sourced under the MIT License. You can freely use, modify, and distribute the code for academic and non-commercial research purposes.

If you use this project in your research or publications, please cite this repository and the corresponding paper:
- Liu X, Wang J, Yu H, Xu P, Li B, Teng W, Han X*, Liu J*. An Interpretable Deep Learning Framework for Tongue Diagnosis: From Spatial Localization to Feature–Organ Association.

- Commercial use, unauthorized redistribution, and closed-source integration are not permitted without prior written permission from the authors.

- The authors are not responsible for any medical or clinical use of this project; it is designed for research only, not for clinical diagnosis or treatment.
---

## Acknowledgments

This work was supported by:
- **Natural Science Foundation of Shandong Province** (Grant No. ZR2024QF228, ZR2024QA176)
- **National Natural Science Foundation of China** (Grant No. 82301666)

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

- **Corresponding Author**: Xiaoqing Han (hanxiaoqing@sdsmu.edu.cn)
- **Institution**: Shandong Second Medical University
- **Issues**: [GitHub Issues](https://github.com/HeanLiu/Worm-PostureNet/issues)

---
