<div align="center">

# TCM-TongueNet




<p align="center">
  <img src="docs/your_image.png" alt="project logo" width="150" />
</p>

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/ResNet-101-red.svg)](https://example.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Your%20Link-4285F4.svg?logo=googledrive&logoColor=white)](https://example.com)


An Interpretable Deep Learning Framework for Tongue Diagnosis: From Spatial Localization to Feature–Organ Association


</div>

---

## Overview


A PyTorch-based open-source toolkit for AI-assisted TCM tongue diagnosis, supporting dual-task decoupled modeling, Grad-CAM visualization, dual-matrix consistency validation, and TCM prior-guided attention learning.

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
- Python 3.13
- CUDA 12.x compatible GPU (recommended)

## Usage Rules

This project is open-sourced under the MIT License. You can freely use, modify, and distribute the code for academic and non-commercial research purposes.

If you use this project in your research or publications, please cite this repository and the corresponding paper:
- Liu X, Wang J, Yu H, Xu P, Li B, Teng W, Han X*, Liu J*. An Interpretable Deep Learning Framework for Tongue Diagnosis: From Spatial Localization to Feature–Organ Association.

- Commercial use, unauthorized redistribution, and closed-source integration are not permitted without prior written permission from the authors.

- The authors are not responsible for any medical or clinical use of this project; it is designed for research only, not for clinical diagnosis or treatment.
