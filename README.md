# Navigating Label Ambiguity for Facial Expression Recognition in the Wild (AAAI 2025)

[![Paper](https://img.shields.io/badge/Paper-AAAI%202025-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/32476/34631)
[![arXiv](https://img.shields.io/badge/arXiv-2502.09993-b31b1b.svg)](https://arxiv.org/abs/2502.09993)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation for the paper **"Navigating Label Ambiguity for Facial Expression Recognition in the Wild"**, accepted at the **AAAI 2025**.

> **Abstract:** Facial expression recognition (FER) remains a challenging task due to label ambiguity caused by the subjective nature of facial expressions and noisy samples. Additionally, class imbalance, which is common in real-world datasets, further complicates FER. To tackle both challenges simultaneously, we propose a novel framework called **Navigating Label Ambiguity (NLA)**, which is robust under real-world conditions. NLA consists of two main components: Noise-aware Adaptive Weighting (NAW) and consistency regularization. Extensive experiments demonstrate that NLA outperforms existing methods in both overall and mean accuracy, confirming its robustness against noise and class imbalance.

---

## 📌 Updates
- **[2025.03]** Official training and evaluation codes are released!
- **[2025.02]** The pre-print is now available on [arXiv](https://arxiv.org/abs/2502.09993).
- **[2024.12]** Our paper has been accepted to **AAAI 2025**!

---

## 📂 Repository Structure

~~~text
NLA/
│
├── src/
│   ├── dataset.py         # Dataloaders for FER datasets
│   ├── loss.py            # Implementation of NAW and Consistency Regularization
│   ├── model.py           # Network architecture
│   ├── resnet.py          # ResNet base functions
│   ├── resnet18.py        # ResNet-18 backbone 
│   ├── test.py            # Evaluation functions
│   ├── train.py           # Training loops and optimization
│   └── utils.py           # Evaluation metrics and helper functions
│
├── main_rafdb.py          # Main execution script for RAF-DB
└── README.md
~~~

---

## ⚙️ Environment Setup

We recommend using Anaconda to manage the environment.

~~~bash
# Clone the repository
git clone https://github.com/jungyu0413/NLA.git
cd NLA

# Create and activate a conda environment
conda create -n nla_fer python=3.8
conda activate nla_fer

# Install PyTorch and other dependencies
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
~~~
*(Note: If you have a `requirements.txt`, you can simply run `pip install -r requirements.txt`)*

---

## 💾 Data Preparation

Download the in-the-wild FER datasets used in this paper:
1. **[RAF-DB](http://www.whdeng.cn/RAF/model1.html)**
2. **[AffectNet](http://mohammadmahoor.com/affectnet/)**
3. **[FERPlus](https://github.com/microsoft/FERPlus)**

After downloading and extracting the datasets, modify the data paths inside your main execution script (e.g., `main_rafdb.py`):

~~~python
# main_rafdb.py
dataset_path = "/path/to/your/dataset"
label_path = "/path/to/your/labels"
~~~

---

## 🚀 Pre-trained Weights

To easily reproduce our reported results, download the pre-trained checkpoints from the link below:
- 🔗 **[Google Drive - NLA Pretrained Weights](https://drive.google.com/file/d/12NY75DwMUnXFbRYDQso4eXIxLAV2PP63/view?usp=drive_link)**

Place the downloaded `.pth` files into a `./checkpoints/` directory at the root of this project.

---

## 💻 Training & Evaluation

### Training
To train the NLA framework from scratch, run the execution script. 

~~~bash
python main_rafdb.py --mode train --batch_size 32 --epochs 60
~~~

### Evaluation
To evaluate the model using the pre-trained weights:

~~~bash
python main_rafdb.py --mode test --resume ./checkpoints/nla_rafdb_best.pth
~~~

---

## 📊 Framework & Results

### Overview of NLA
![NLA Framework](https://github.com/user-attachments/assets/a1e549f0-dff8-40b8-ba90-dc14e04f57e9)

### Quantitative Results

**1. Comparison on RAF-DB & AffectNet**
| RAF-DB | AffectNet |
| :---: | :---: |
| ![RAF-DB Results](https://github.com/user-attachments/assets/24b0f829-1cb6-4475-a928-0839f6579ebf) | ![AffectNet Results](https://github.com/user-attachments/assets/ca4b0e87-049d-466a-96fa-c64a53ea4802) |

**2. Robustness under Noise & Class Imbalance**
<img width="918" alt="result" src="https://github.com/user-attachments/assets/9186d792-0c07-41a3-be0b-ebf8f39a1da4" />

---

## 📜 Citation

If you find this repository or our paper useful for your research, please consider citing:

~~~bibtex
@inproceedings{lee2025navigating,
  title={Navigating Label Ambiguity for Facial Expression Recognition in the Wild},
  author={Lee, JunGyu and Choi, Yeji and Kim, Haksub and Kim, Ig-Jae and Nam, Gi Pyo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
~~~

## 🙏 Acknowledgements
This research was supported by the Field-oriented Technology Development Project for Customs Administration through the National Research Foundation of Korea (NRF) funded by the Ministry of Science & ICT and Korea Customs Service (2022M311A1095154), and supported by the KIST Institutional Program (Project No. 2E33001).


