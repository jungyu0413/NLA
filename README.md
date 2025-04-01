# Navigating Label Ambiguity for Facial Expression Recognition in the Wild (AAAI 2025)

This is the official repository for the paper:  
[**"Navigating Label Ambiguity for Facial Expression Recognition in the Wild"**](https://arxiv.org/abs/2502.09993), accepted at **AAAI 2025**.

---

## Abstract

Facial expression recognition (FER) remains a challenging task due to label ambiguity caused by the subjective nature of facial expressions and noisy samples. Additionally, class imbalance, which is common in real-world datasets, further complicates FER. Although many studies have shown impressive improvements, they typically address only one of these issues, leading to suboptimal results. To tackle both challenges simultaneously, we propose a novel framework called Navigating Label Ambiguity (NLA), which is robust under real-world conditions. The motivation behind NLA is that dynamically estimating and emphasizing ambiguous samples at each iteration helps mitigate noise and class imbalance by reducing the model’s bias toward majority classes. To achieve this, NLA consists of two main components: Noiseaware Adaptive Weighting (NAW) and consistency regularization. Specifically, NAW adaptively assigns higher importance to ambiguous samples and lower importance to noisy ones, based on the correlation between the intermediate prediction scores for the ground truth and the nearest negative. Moreover, we incorporate a regularization term to ensure consistent latent distributions. Consequently, NLA enables the model to progressively focus on more challenging ambiguous samples, which primarily belong to the minority class, in the later stages of training. Extensive experiments demonstrate that NLA outperforms existing methods in both overall and mean accuracy, confirming its robustness against noise and class imbalance. To the best of our knowledge, this is the first framework to address both problems simultaneously.

---

## Framework Overview

![Image](https://github.com/user-attachments/assets/a1e549f0-dff8-40b8-ba90-dc14e04f57e9)

---

## Results

### Comparison with other methods on RAF-DB  
![Image](https://github.com/user-attachments/assets/24b0f829-1cb6-4475-a928-0839f6579ebf)

### Comparison with other methods on AffectNet  
![Image](https://github.com/user-attachments/assets/ca4b0e87-049d-466a-96fa-c64a53ea4802)

### Evaluation of overall accuracy across different noise ratios and class imbalance conditions
![Image](<img width="918" alt="Image" src="https://github.com/user-attachments/assets/02e0e0e7-6a5e-41bd-9a40-16301da7b37c" />)

---

## 🚀 How to Use

### 1. Download Datasets  
- [RAF-DB](http://www.whdeng.cn/RAF/model1.html)  
- [AffectNet](https://www.affectnet.org/)  
- [FERPlus](https://www.microsoft.com/en-us/research/project/ferplus-dataset/)

### 2. Prepare Pretrained Weights  
Download the pretrained weights from the following link:  
[Google Drive - NLA Pretrained Weights](https://drive.google.com/file/d/12NY75DwMUnXFbRYDQso4eXIxLAV2PP63/view?usp=drive_link)

### 3. Set Dataset Paths  
In `main.py`, modify the following lines:

```python
dataset_path = "your_dataset_path"
label_path = "your_label_path"
